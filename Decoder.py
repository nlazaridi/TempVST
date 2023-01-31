import torch.nn as nn
import torch
from token_performer import Token_performer
from Transformer import saliency_token_inference, contour_token_inference, token_TransformerEncoder
from einops import rearrange

class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1,attn_drop=0.1, proj_drop=0.1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((saliency_tokens, fea), dim=1)
        #fea = torch.cat((fea, contour_tokens), dim=1)
        # [B, 1 + H*W + 1, 384]

        fea = self.encoderlayer(fea)
        # fea [B, 1 + H*W + 1, 384]
        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        #contour_tokens = fea[:, -1, :].unsqueeze(1)

        saliency_fea = self.saliency_token_pre(fea)
        # saliency_fea [B, H*W, 384]
        #contour_fea = self.contour_token_pre(fea)
        # contour_fea [B, H*W, 384]

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        #contour_fea = self.mlp2_c(self.norm2_c(contour_fea))

        return saliency_fea, fea, saliency_tokens


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5,dp1=0.1,dp2=0.1)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        #dec_fea = dec_fea.unsqueeze(0)
        #dec_fea = dec_fea.squeeze(dim=1)
        #if self.fuse:
        #    enc_fea = enc_fea.unsqueeze(0)
        #    enc_fea = enc_fea.squeeze(dim=1)
        if len(dec_fea.size())==2:
            dec_fea = dec_fea.unsqueeze(0)
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            if len(dec_fea.size())==2:
                dec_fea = dec_fea.unsqueeze(0)
            if len(enc_fea.size())==2:
                enc_fea = enc_fea.unsqueeze(0)
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)

        return dec_fea


class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        
        '''
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
    def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, rgb_fea_1_8, rgb_fea_1_4, len_snippet, batch_size):
        # saliency_fea_1_16 [B, 14*14, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # token_fea_1_16  [B, 1 + 14*14 + 1, 384] (contain saliency token and contour token)

        # saliency_tokens [B, 1, 384]
        # contour_tokens [B, 1, 384]

        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 56*56, 64]

        B, _, _, = token_fea_1_16.size()

        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # saliency_fea_1_16 [B, 14*14, 64]
        saliency_fea_1_16 = rearrange(saliency_fea_1_16, 'b (f p) d -> b f p d', f=len_snippet)
        mask_1_16 = self.pre_1_16(saliency_fea_1_16[:,-1,:,:].squeeze())
        if len(mask_1_16.size()) == 2:
            mask_1_16 = mask_1_16.unsqueeze(dim=0)
        mask_1_16 = mask_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        zz = rearrange(token_fea_1_16[:,1:,:], 'b (f p) d ->  b f p d', f=len_snippet)
        rgb_fea_1_8 = rearrange(rgb_fea_1_8, '(b f) p d -> b f p d', f=len_snippet)
        fea_1_8 = self.decoder1(zz[:,-1,:,:].squeeze(), rgb_fea_1_8[:,-1,:,:].squeeze())

        # token prediction
        saliency_fea_1_8, token_fea_1_8, saliency_tokens = self.token_pre_1_8(fea_1_8, saliency_tokens)

        # predict saliency maps and contour maps
        mask_1_8 = self.pre_1_8(saliency_fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)


        # 1/8 -> 1/4
        rgb_fea_1_4 = rearrange(rgb_fea_1_4, '(b f) p d -> b f p d', f=len_snippet)
        rgb_fea_1_4_new = rgb_fea_1_4[:,-1,:,:].squeeze()
        if len(rgb_fea_1_4_new.size()) == 2:
            rgb_fea_1_4_new = rgb_fea_1_4_new.unsqueeze(dim=0)
        fea_1_4 = self.decoder2(token_fea_1_8[:, 1:, :], rgb_fea_1_4_new)

        # token prediction
        saliency_fea_1_4, token_fea_1_4, saliency_tokens = self.token_pre_1_4(fea_1_4, saliency_tokens)

        # predict saliency maps and contour maps
        mask_1_4 = self.pre_1_4(saliency_fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        saliency_fea_1_1 = self.decoder3(saliency_fea_1_4)

        mask_1_1 = self.pre_1_1(saliency_fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)



        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1]

