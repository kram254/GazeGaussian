import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)
        
    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        return shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, cond_dim=None):
        super().__init__()
        self.use_conditioning = cond_dim is not None
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
        if self.use_conditioning:
            self.adaLN = AdaptiveLayerNorm(dim, cond_dim)
        
    def forward(self, x, cond=None):
        if self.use_conditioning and cond is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(x, cond)
            x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa)
            x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=256, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Unpatchify(nn.Module):
    def __init__(self, patch_size=8, embed_dim=768, out_chans=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, out_chans * patch_size * patch_size)
        self.out_chans = out_chans
        
    def forward(self, x, H, W):
        B, N, _ = x.shape
        x = self.proj(x)
        
        p = self.patch_size
        h = H // p
        w = W // p
        
        x = x.reshape(B, h, w, p, p, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_chans, H, W)
        return x
