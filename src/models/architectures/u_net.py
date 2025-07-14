import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        emb_out = self.emb_proj(F.silu(emb))[:, :, None, None]
        h = h + emb_out
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.res_conv(x)


class DenoiserUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        n_layers: int,
        num_classes: int,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        chs = [base_channels * (2**i) for i in range(n_layers)]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # initial conv
        self.init_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # downsampling
        for i in range(n_layers):
            in_ch = chs[i]
            out_ch = chs[i] if i == 0 else chs[i]
            self.downs.append(ResBlock(in_ch, chs[i], emb_dim))
            if i < n_layers - 1:
                self.downs.append(nn.Conv2d(chs[i], chs[i + 1], kernel_size=4, stride=2, padding=1))

        # bottleneck
        self.bot1 = ResBlock(chs[-1], chs[-1] * 2, emb_dim)
        self.bot2 = ResBlock(chs[-1] * 2, chs[-1], emb_dim)

        # upsampling
        for i in reversed(range(n_layers)):
            in_ch = chs[i] * 2
            out_ch = chs[i]
            self.ups.append(ResBlock(in_ch, out_ch, emb_dim))
            if i > 0:
                self.ups.append(nn.ConvTranspose2d(out_ch, chs[i - 1], kernel_size=4, stride=2, padding=1))

        # final conv
        self.final_res = ResBlock(base_channels, base_channels, emb_dim)
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t, class_labels, **model_kwargs):
        """
        x: (B, C, H, W) noised input
        t: (B,) noise level
        class_labels: (B,) integer labels in [0..num_classes]
        """
        B, _, H, W = x.shape
        # time embedding
        t_emb = self.time_mlp(t)

        # class embedding
        c_emb = self.class_emb(class_labels)
        emb = t_emb + c_emb  # (B, emb_dim)

        # down path
        h = self.init_conv(x)
        hs = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                hs.append(h)
            else:  # downsample conv
                h = layer(h)

        # bottleneck
        h = self.bot1(h, emb)
        h = self.bot2(h, emb)

        # up path
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h_skip = hs.pop()
                h = torch.cat([h, h_skip], dim=1)
                h = layer(h, emb)
            else:  # upsample ConvTranspose2d
                h = layer(h)

        h = self.final_res(h, emb)
        out = self.final_conv(h)
        return out
