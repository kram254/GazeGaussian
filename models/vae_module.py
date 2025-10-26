import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels=256, z_channels=4, ch=128, ch_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down_block = nn.Module()
            down_block.block = block
            if i_level != self.num_resolutions - 1:
                down_block.downsample = Downsample(block_in)
            self.down.append(down_block)
        
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in),
            ResnetBlock(block_in, block_in),
        ])
        
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, padding=1)
    
    def forward(self, x):
        h = self.conv_in(x)
        
        for i_level in range(self.num_resolutions):
            for i_block in range(len(self.down[i_level].block)):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        
        for block in self.mid:
            h = block(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VAEDecoder(nn.Module):
    def __init__(self, out_channels=256, z_channels=4, ch=128, ch_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[-1]
        
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)
        
        self.mid = nn.ModuleList([
            ResnetBlock(block_in, block_in),
            ResnetBlock(block_in, block_in),
        ])
        
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up_block = nn.Module()
            up_block.block = block
            if i_level != 0:
                up_block.upsample = Upsample(block_in)
            self.up.insert(0, up_block)
        
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        
        for block in self.mid:
            h = block(h)
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(len(self.up[i_level].block)):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VAE(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, z_channels=4, ch=128, ch_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, z_channels, ch, ch_mult, num_res_blocks)
        self.decoder = VAEDecoder(out_channels, z_channels, ch, ch_mult, num_res_blocks)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self, x, sample=True):
        mean, logvar = self.encode(x)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        return self.decode(z), mean, logvar
