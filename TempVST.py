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
            dim = 384,
            image_size = 224,
            patch_size = 16,
            num_frames = args.len_snippet,
            depth = 4,
            heads = 6,
            dim_head = 64,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            mlp_ratio = 3
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
        image_size=224,
        patch_size = 16,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        mlp_ratio = 3
    ):

        super().__init__()
        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches

        self.heads = heads
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.pos_emb = nn.Embedding(num_positions + 1, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, mult=mlp_ratio, dropout = ff_dropout)
            time_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def forward(self, x):

        x += self.pos_emb(torch.arange(x.shape[1], device=x.device))

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f p) d', '(b p) f d', p=self.num_patches) + x
            x = spatial_attn(x, 'b (f p) d', '(b f) p d', f=self.num_frames) + x
            x = ff(x) + x

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
        mask=None,
        **einops_dims
    ):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f"{einops_from} -> {einops_to}", **einops_dims),
            (q, k, v),
        )

        # attention
        out = attn(q_, k_, v_, mask=mask)

        # merge back time or space
        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)

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
