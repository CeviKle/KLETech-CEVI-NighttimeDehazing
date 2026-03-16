"""
Composite loss suite for nighttime image dehazing.

Includes:
  - Charbonnier L1 (robust pixel loss)
  - MS-SSIM (multi-scale structural similarity)
  - VGG Perceptual (multi-layer feature matching)
  - Frequency (FFT domain high-frequency preservation)
  - Color Consistency (channel-wise mean/variance alignment)
  - Illumination Smoothness (for auxiliary illumination map)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# PIXEL-LEVEL LOSSES
# ============================================================

class CharbonnierLoss(nn.Module):
    """Robust L1 alternative: sqrt(diff^2 + eps^2)."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ============================================================
# STRUCTURAL LOSSES
# ============================================================

def _gaussian_kernel(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_window(window_size, channel):
    _1d = _gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    return window.expand(channel, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (returns 1 - SSIM)."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", _create_window(window_size, 3))

    def _ssim(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ch = img1.size(1)
        win = self.window
        if win.device != img1.device:
            win = win.to(img1.device)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, win, padding=pad, groups=ch)
        mu2 = F.conv2d(img2, win, padding=pad, groups=ch)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, win, padding=pad, groups=ch) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, win, padding=pad, groups=ch) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, win, padding=pad, groups=ch) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        # Normalize to [0, 1] if in [-1, 1]
        return 1.0 - self._ssim(pred, target)


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM loss."""
    def __init__(self, weights=None, window_size=11):
        super().__init__()
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.ssim_module = SSIMLoss(window_size)

    def forward(self, pred, target):
        levels = len(self.weights)
        loss = 0.0
        for i in range(levels):
            ssim_val = 1.0 - self.ssim_module(pred, target)  # Get SSIM value
            loss += self.weights[i] * (1.0 - ssim_val) if i < levels - 1 else self.weights[i] * ssim_val
            if i < levels - 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                # Update buffer device
        return loss


# ============================================================
# PERCEPTUAL LOSS
# ============================================================

class PerceptualLoss(nn.Module):
    """Multi-layer VGG-19 perceptual loss with proper ImageNet normalization."""
    def __init__(self, layer_indices=None):
        super().__init__()
        if layer_indices is None:
            layer_indices = [2, 7, 12, 21, 30]  # conv1_2, conv2_2, conv3_2, conv4_2, conv5_2

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in layer_indices:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
            prev = idx + 1

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Convert from [-1,1] to ImageNet-normalized."""
        x = x * 0.5 + 0.5  # → [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        h_pred, h_target = pred, target
        for s in self.slices:
            h_pred = s(h_pred)
            h_target = s(h_target)
            loss += F.l1_loss(h_pred, h_target)
        return loss


# ============================================================
# FREQUENCY LOSS
# ============================================================

class FrequencyLoss(nn.Module):
    """FFT-domain loss for preserving high-frequency detail."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Phase spectrum
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)

        mag_loss = F.l1_loss(pred_mag, target_mag)
        phase_loss = F.l1_loss(pred_phase, target_phase)

        return mag_loss + 0.1 * phase_loss


# ============================================================
# COLOR CONSISTENCY LOSS
# ============================================================

class ColorConsistencyLoss(nn.Module):
    """Channel-wise mean and variance alignment."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_mean = pred.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        pred_var = pred.var(dim=[2, 3])
        target_var = target.var(dim=[2, 3])

        mean_loss = F.l1_loss(pred_mean, target_mean)
        var_loss = F.l1_loss(pred_var, target_var)
        return mean_loss + var_loss


class ColorAngleLoss(nn.Module):
    """Penalizes hue/saturation shifts by comparing RGB vector angles."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Add epsilon to prevent div by zero
        pred_norm = F.normalize(pred + 1e-6, dim=1)
        target_norm = F.normalize(target + 1e-6, dim=1)
        # Cosine similarity -> 1 is best. We want to minimize (1 - cos_sim)
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        return 1.0 - cos_sim.mean()


# ============================================================
# ILLUMINATION SMOOTHNESS
# ============================================================

class IlluminationSmoothnessLoss(nn.Module):
    """Total variation for the illumination map to enforce smoothness."""
    def __init__(self):
        super().__init__()

    def forward(self, illum_map):
        tv_h = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :]).mean()
        tv_w = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1]).mean()
        return tv_h + tv_w


# ============================================================
# COMPOSITE LOSS
# ============================================================

class CompositeLoss(nn.Module):
    """
    Aggregated loss function with configurable weights.
    Handles both main restoration loss and auxiliary illumination loss.
    """
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            from config import LossConfig
            cfg = LossConfig()

        self.char_loss = CharbonnierLoss(eps=cfg.charbonnier_eps)
        self.ssim_loss = SSIMLoss()
        self.w_char = cfg.charbonnier_weight
        self.w_ssim = cfg.ssim_weight

        self.use_perceptual = cfg.use_perceptual
        self.use_frequency = cfg.use_frequency
        self.use_color = cfg.use_color

        if self.use_perceptual:
            self.perc_loss = PerceptualLoss()
            self.w_perc = cfg.perceptual_weight

        if self.use_frequency:
            self.freq_loss = FrequencyLoss()
            self.w_freq = cfg.frequency_weight

        if self.use_color:
            self.color_loss = ColorConsistencyLoss()
            self.color_angle_loss = ColorAngleLoss()
            self.w_color = cfg.color_weight

        self.illum_loss = IlluminationSmoothnessLoss()

    def forward(self, pred, target, illum_map=None):
        loss = 0.0
        
        if self.w_char > 0:
            loss += self.w_char * self.char_loss(pred, target)
        
        if self.w_ssim > 0:
            loss += self.w_ssim * self.ssim_loss(pred, target)

        if self.use_perceptual and self.w_perc > 0:
            loss += self.w_perc * self.perc_loss(pred, target)
            
        if self.use_frequency and self.w_freq > 0:
            loss += self.w_freq * self.freq_loss(pred, target)
            
        if self.use_color and self.w_color > 0:
            loss += self.w_color * self.color_loss(pred, target)
            loss += 0.1 * self.color_angle_loss(pred, target)  # Added angular constraint

        if illum_map is not None:
            loss += 0.01 * self.illum_loss(illum_map)

        return loss
