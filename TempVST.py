import torch
from torch import nn, einsum
import torch.nn as nn 
from t2t_vit import T2t_vit_t_14
from Transformer import Transformer
from Transformer import token_Transformer
from Decoder import Decoder
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

class TempVST(nn.Module):
    def __init__(self, args):
        super(TempVST, self).__init__()

        self.batch_size = args.batch_size
        self.len_snippet = args.len_snippet


        #VST Timesformer 
        self.timesformer = TimeSformer(
            dim = 512,
            image_size = 224,
            patch_size = 16, #was 16
            num_frames = 8,
            num_classes = 10,
            depth = 12,
            heads = 8,
            dim_head =  64,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            rotary_emb = False,
            num_gpu = args.num_gpu,
            batch_size = args.batch_size,
            len_snippet = args.len_snippet
        )
        self.len_snippet = args.len_snippet
        self.batch_size = args.batch_size

        # VST Convertor
        self.transformer = Transformer(
            embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.0
        )

        # VST Decoder
        self.token_trans = token_Transformer(
            embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.0
        )
        self.decoder = Decoder(
            embed_dim=384, token_dim=64, depth=2, img_size=args.img_size
        )
        
        self.apply(self._init_weights)

        #VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    

    def forward(self, video_Input):

        B, _, _, _, _ = video_Input.shape

        video_Input = rearrange(video_Input, 'b f c h w -> (b f) c h w')

        # VST Encoder
        #with torch.no_grad():
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(video_Input)

        # VST TimeSformer
        rgb_fea_1_16 = rearrange(rgb_fea_1_16, '(b f) p d -> b (f p) d', b=B)
        rgb_fea_1_16 = self.timesformer(rgb_fea_1_16)
        rgb_fea_1_16 = rearrange(rgb_fea_1_16, 'b (f p) d -> (b f) p d', f = self.len_snippet)

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)

        # VST Decoder
        rgb_fea_1_16 = rearrange(rgb_fea_1_16, '(b f) p d -> b (f p) d', f = self.len_snippet)
        (
            saliency_fea_1_16,
            fea_1_16,
            saliency_tokens,
           
        ) = self.token_trans(rgb_fea_1_16)

        outputs = self.decoder(
            saliency_fea_1_16,
            fea_1_16,
            saliency_tokens,
            rgb_fea_1_8,
            rgb_fea_1_4,
            self.len_snippet,
            self.batch_size
        )

        return outputs


class TimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        num_gpu,
        image_size=224,
        patch_size = 16,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True,
        shift_tokens = False,
        batch_size = 16,
        len_snippet = 5
    ):

        super().__init__()
        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        #patch_dim = channels * patch_size ** 2
        patch_dim = 384

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.batch_size = batch_size
        self.len_snippet = len_snippet
        self.num_gpu = num_gpu


        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)


        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.to_out = nn.Linear(dim, patch_dim)

        trunc_normal_(self.cls_token, std=0.2)
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def forward(self, video, mask = None):
        
        device = video.device
        tokens = self.to_patch_embedding(video)

        x = tokens 
        # positional embedding

        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(tokens.shape[0], device = device)
            image_pos_emb = self.image_rot_emb(1, 1, device = device)


        # calculate masking for uneven number of frames

        frame_mask = None
        cls_attn_mask = None
        if mask is not None:
            #mask_with_cls = F.pad(mask, (1, 0), value = True)

            frame_mask = repeat(mask, 'b f -> (b h n) () f', n = n, h = self.heads)

            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n = n, h = self.heads)
            #cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value = True)

        # time and space attention

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f p) d', '(b p) f d', num_gpu = self.num_gpu, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb, batch_size = self.batch_size, len_snippet = self.len_snippet) + x
            #x = time_attn(x, '(b f) n d', '(b n) f d',  mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
            #x = spatial_attn(x, '(b f) n d', '(b f) n d', cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            x = spatial_attn(x, 'b (f p) d', '(b f) p d', num_gpu = self.num_gpu, cls_mask = cls_attn_mask, rot_emb = image_pos_emb, batch_size = self.batch_size, len_snippet = self.len_snippet) + x
            x = ff(x) + x
        
        x = self.to_out(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def forward(
        self,
        x,
        einops_from,
        einops_to,
        num_gpu,
        mask=None,
        cls_mask=None,
        rot_emb=None,
        batch_size = 16,
        len_snippet = 5,
        **einops_dims
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        q = q * self.scale

        # splice out classification token at index 1
        #(cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #    lambda t: (t[:, :1], t[:, 1:]), (q, k, v)
        #)

        # let classification token attend to key / values of all patches across time and space
        #cls_out = attn(cls_q, k, v, mask=cls_mask)

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f"{einops_from} -> {einops_to}", b=int(batch_size*h/num_gpu), f=len_snippet, **einops_dims),
            (q, k, v),
        )

        # add rotary embeddings, if applicable
        if rot_emb is not None:
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)

        # expand cls token keys and values across time or space and concat
        #r = q_.shape[0] // cls_k.shape[0]
        #cls_k, cls_v = map(
        #    lambda t: repeat(t, "b () d -> (b r) () d", r=r), (cls_k, cls_v)
        #)

        #k_ = torch.cat((cls_k, k_), dim=1)
        #v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_, mask=mask)

        # merge back time or space
        out = rearrange(out, f"{einops_to} -> {einops_from}", b=int(batch_size*h/num_gpu), f=len_snippet, **einops_dims)

        # concat back the cls token
        #out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        # combine heads out
        return self.to_out(out)


def attn(q, k, v, mask=None):
    sim = einsum("b i d, b j d -> b i j", q, k)

    if mask is not None:
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)
    out = einsum("b i j, b j d -> b i d", attn, v)
    return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )
    def forward(self,x):
        return self.net(x)
        

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
