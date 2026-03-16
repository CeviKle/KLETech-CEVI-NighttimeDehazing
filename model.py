"""
NightDehazeNet — Advanced Multi-Scale Attention U-Net for Nighttime Image Dehazing.

Architecture highlights:
  - Multi-scale context blocks (dilated convolutions at rates 1, 2, 4)
  - Channel attention (SE-style squeeze-and-excite)
  - Spatial attention (CBAM-style)
  - Restormer-style transposed self-attention
  - Gated depthwise-conv feed-forward
  - Gated skip fusion instead of plain concat
  - Optional illumination-aware auxiliary head
  - Global residual learning (output + input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


# ============================================================
# BUILDING BLOCKS
# ============================================================

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 2D feature maps."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sigma = x.std(1, keepdim=True)
        return (x - mu) / (sigma + self.eps) * self.gamma + self.beta


class ChannelAttention(nn.Module):
    """SE-style squeeze-and-excite channel attention."""
    def __init__(self, dim, reduction=16):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention using max+avg pool."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        gate = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * gate


class MultiScaleContext(nn.Module):
    """
    Parallel dilated convolutions at rates 1, 2, 4 to capture
    multi-scale haze patterns, fused via 1×1 projection.
    """
    def __init__(self, dim):
        super().__init__()
        branch_dim = dim // 3
        remainder = dim - branch_dim * 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, branch_dim, 3, padding=1, dilation=1, bias=False),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, branch_dim, 3, padding=2, dilation=2, bias=False),
            nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, branch_dim + remainder, 3, padding=4, dilation=4, bias=False),
            nn.GELU()
        )
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b4], dim=1)
        return self.fuse(out) + x  # Residual


class GatedDconvFeedForward(nn.Module):
    """Gated depthwise-conv feed-forward from Restormer."""
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransposedAttention(nn.Module):
    """
    Restormer-style Multi-DConv Head Transposed Attention.
    Operates on CxHW instead of HWxC for memory efficiency.
    """
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        temp = self.temperature.clamp(max=5.0)
        attn = (q @ k.transpose(-2, -1)) * temp
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x → LN → Attention → + → LN → FFN → +
    With optional channel and spatial attention after FFN.
    """
    def __init__(self, dim, num_heads, expansion_factor=2.66,
                 use_channel_attn=True, use_spatial_attn=True):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = TransposedAttention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedDconvFeedForward(dim, expansion_factor)

        self.ca = ChannelAttention(dim) if use_channel_attn else nn.Identity()
        self.sa = SpatialAttention() if use_spatial_attn else nn.Identity()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        y = self.ffn(self.norm2(x))
        y = self.ca(y)
        y = self.sa(y)
        return x + y


# ============================================================
# ENCODER / DECODER COMPONENTS
# ============================================================

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.body = nn.Conv2d(in_ch, in_ch * 2, kernel_size=4, stride=2,
                              padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.body = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2,
                                       stride=2, bias=False)

    def forward(self, x):
        return self.body(x)


class GatedSkipFusion(nn.Module):
    """Learnable gating for skip connection fusion."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, decoder_feat, skip_feat):
        cat = torch.cat([decoder_feat, skip_feat], dim=1)
        return self.conv(cat) * self.gate(cat)


class PlainSkipFusion(nn.Module):
    """Simple 1×1 conv for skip connection merging."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, decoder_feat, skip_feat):
        return self.conv(torch.cat([decoder_feat, skip_feat], dim=1))


# ============================================================
# MAIN NETWORK
# ============================================================

class NightDehazeNet(nn.Module):
    """
    Multi-Scale Attention U-Net for nighttime image dehazing.

    Args:
        inp_channels: Input image channels (default 3)
        out_channels: Output image channels (default 3)
        dim: Base feature dimension (default 48)
        num_blocks: Transformer blocks per encoder level
        num_refinement_blocks: Blocks in refinement stage
        heads: Attention heads per level
        expansion_factor: FFN expansion
        use_multiscale_context: Enable multi-scale dilated conv
        use_channel_attention: Enable SE attention
        use_spatial_attention: Enable CBAM spatial attention
        use_gated_skip: Gated vs plain skip fusion
        use_illumination_head: Auxiliary illumination branch
    """

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=None,
                 num_refinement_blocks=4,
                 heads=None,
                 expansion_factor=2.66,
                 use_multiscale_context=True,
                 use_channel_attention=True,
                 use_spatial_attention=True,
                 use_gated_skip=True,
                 use_illumination_head=True,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.use_illumination_head = use_illumination_head

        # --- Stem ---
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False)

        # Optional multi-scale context on input features
        self.context = (MultiScaleContext(dim)
                        if use_multiscale_context else nn.Identity())

        # --- Encoder ---
        self.encoder_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            level_dim = dim * (2 ** i)
            blocks = nn.Sequential(*[
                TransformerBlock(level_dim, heads[i], expansion_factor,
                                 use_channel_attention, use_spatial_attention)
                for _ in range(num_blocks[i])
            ])
            self.encoder_layers.append(blocks)
            if i < 3:
                self.downsamples.append(DownSample(level_dim))

        # --- Decoder ---
        self.decoder_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        SkipFusionClass = GatedSkipFusion if use_gated_skip else PlainSkipFusion

        for i in range(3):
            # Level indices: decode from bottom to top
            # i=0: level 3→2, i=1: level 2→1, i=2: level 1→0
            up_dim = dim * (2 ** (3 - i))       # Input dim to upsample
            skip_dim = dim * (2 ** (2 - i))     # Skip connection dim

            self.upsamples.append(UpSample(up_dim))
            self.skip_fusions.append(SkipFusionClass(skip_dim))
            blocks = nn.Sequential(*[
                TransformerBlock(skip_dim, heads[2 - i], expansion_factor,
                                 use_channel_attention, use_spatial_attention)
                for _ in range(num_blocks[2 - i])
            ])
            self.decoder_layers.append(blocks)

        # --- Refinement ---
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], expansion_factor,
                             use_channel_attention, use_spatial_attention)
            for _ in range(num_refinement_blocks)
        ])

        # --- Output projection ---
        self.output_proj = nn.Conv2d(dim, out_channels, 3, padding=1, bias=False)

        # --- Auxiliary illumination head ---
        if use_illumination_head:
            self.illum_head = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 4, 1, 1, bias=False),
                nn.Sigmoid()
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inp_img):
        # Stem + multi-scale context
        x0 = self.context(self.patch_embed(inp_img))

        # Encoder
        enc_feats = []
        feat = x0
        for i, layer in enumerate(self.encoder_layers):
            if self.use_checkpoint and self.training:
                feat = checkpoint(layer, feat, use_reentrant=False)
            else:
                feat = layer(feat)
            enc_feats.append(feat)
            if i < 3:
                feat = self.downsamples[i](feat)

        # Decoder
        x = enc_feats[-1]  # Bottleneck
        for i, (layer, up, fuse) in enumerate(
                zip(self.decoder_layers, self.upsamples, self.skip_fusions)):
            x = up(x)
            skip = enc_feats[-(i + 2)]
            x = fuse(x, skip)
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Refinement
        if self.use_checkpoint and self.training:
            x = checkpoint(self.refinement, x, use_reentrant=False)
        else:
            x = self.refinement(x)

        # Output
        out = self.output_proj(x)
        restored = out + inp_img  # Global residual

        if self.use_illumination_head and self.training:
            illum_map = self.illum_head(x)
            return restored, illum_map

        return restored


def build_model(cfg=None):
    """Build model from config."""
    if cfg is None:
        from config import ModelConfig
        cfg = ModelConfig()

    return NightDehazeNet(
        inp_channels=cfg.inp_channels,
        out_channels=cfg.out_channels,
        dim=cfg.dim,
        num_blocks=cfg.num_blocks,
        num_refinement_blocks=cfg.num_refinement_blocks,
        heads=cfg.heads,
        expansion_factor=cfg.expansion_factor,
        use_multiscale_context=cfg.use_multiscale_context,
        use_channel_attention=cfg.use_channel_attention,
        use_spatial_attention=cfg.use_spatial_attention,
        use_gated_skip=cfg.use_gated_skip,
        use_illumination_head=cfg.use_illumination_head,
        use_checkpoint=getattr(cfg, 'use_checkpoint', False),
    )


if __name__ == "__main__":
    model = NightDehazeNet(dim=48)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_params:.2f}M | Trainable: {trainable:.2f}M")

    # Test forward
    x = torch.randn(1, 3, 256, 256)
    model.train()
    out, illum = model(x)
    print(f"Train output: {out.shape}, illum: {illum.shape}")

    model.eval()
    out = model(x)
    print(f"Eval output: {out.shape}")
