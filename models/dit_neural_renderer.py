import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

if __name__ == "__main__":
    from dit_block import DiTBlock, PatchEmbed, Unpatchify
    from pixel_shuffle_upsample import Blur, PixelShuffleUpsample
    from vae_module import VAE
else:
    from models.dit_block import DiTBlock, PatchEmbed, Unpatchify
    from models.pixel_shuffle_upsample import Blur, PixelShuffleUpsample
    from models.vae_module import VAE


class DiTNeuralRenderer(nn.Module):
    def __init__(
        self,
        bg_type="white",
        feat_nc=256,
        out_dim=3,
        final_actvn=True,
        min_feat=32,
        featmap_size=32,
        img_size=256,
        patch_size=8,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        cond_dim=144,
        use_vae=False,
        vae_z_channels=4,
        freeze_vae=True,
        predict_noise=False,
        **kwargs
    ):
        super().__init__()
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        self.n_feat = feat_nc
        self.out_dim = out_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.cond_dim = cond_dim
        self.use_vae = use_vae
        self.predict_noise = predict_noise
        
        embed_dim = 512
        
        self.patch_embed = PatchEmbed(
            img_size=featmap_size,
            patch_size=patch_size,
            in_chans=feat_nc,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_dim=cond_dim
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.unpatchify = Unpatchify(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_chans=feat_nc
        )
        
        n_blocks = int(log2(img_size) - log2(featmap_size))
        self.feat_upsample_list = nn.ModuleList([
            PixelShuffleUpsample(max(feat_nc // (2 ** i), min_feat))
            for i in range(n_blocks)
        ])
        
        self.rgb_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Blur()
        )
        
        self.feat_2_rgb_list = nn.ModuleList(
            [nn.Conv2d(feat_nc, out_dim, 1, 1, padding=0)]
            + [
                nn.Conv2d(
                    max(feat_nc // (2 ** (i + 1)), min_feat),
                    out_dim,
                    1,
                    1,
                    padding=0,
                )
                for i in range(n_blocks)
            ]
        )
        
        self.feat_layers = nn.ModuleList([
            nn.Conv2d(
                max(feat_nc // (2 ** i), min_feat),
                max(feat_nc // (2 ** (i + 1)), min_feat),
                1,
                1,
                padding=0,
            )
            for i in range(n_blocks)
        ])
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        self.n_blocks = n_blocks
        
        self._build_bg_featmap()
        self._init_weights()
        
        if self.use_vae:
            self.vae = VAE(
                in_channels=feat_nc,
                out_channels=feat_nc,
                z_channels=vae_z_channels,
                ch=128,
                ch_mult=(1, 2, 4),
                num_res_blocks=2
            )
            if freeze_vae:
                for param in self.vae.parameters():
                    param.requires_grad = False
        
    def _build_bg_featmap(self):
        if self.bg_type == "white":
            bg_featmap = torch.ones(
                (1, self.n_feat, self.featmap_size, self.featmap_size),
                dtype=torch.float32,
            )
        elif self.bg_type == "black":
            bg_featmap = torch.zeros(
                (1, self.n_feat, self.featmap_size, self.featmap_size),
                dtype=torch.float32,
            )
        else:
            bg_featmap = None
            print("Error bg_type")
            exit(0)
            
        self.register_parameter("bg_featmap", torch.nn.Parameter(bg_featmap))
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def get_bg_featmap(self):
        return self.bg_featmap
        
    def forward(self, x, shape_code):
        B, C, H, W = x.shape
        
        if self.use_vae:
            vae_mean, vae_logvar = self.vae.encode(x)
            x_vae = vae_mean
            x = self.patch_embed(x_vae)
        else:
            x = self.patch_embed(x)
            
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x, shape_code)
            
        x = self.norm(x)
        x = self.unpatchify(x, self.featmap_size, self.featmap_size)
        
        if self.use_vae:
            x = torch.tanh(x)
            x = self.vae.decode(x)
        
        if self.n_blocks == 0:
            rgb = x[:, :3]
            return rgb
            
        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x))
        net = x
        
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
                
        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
            
        return rgb


if __name__ == "__main__":
    model = DiTNeuralRenderer(
        feat_nc=256,
        featmap_size=128,
        img_size=512,
        patch_size=8,
        depth=6,
        num_heads=8,
        cond_dim=144
    )
    
    x = torch.rand(2, 256, 128, 128)
    cond = torch.rand(2, 144)
    
    out = model(x, cond)
    print(f"Input: {x.shape}")
    print(f"Condition: {cond.shape}")
    print(f"Output: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
