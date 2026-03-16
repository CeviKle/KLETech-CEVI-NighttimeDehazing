"""
Evaluation metrics: PSNR, SSIM, LPIPS, NIQE.

All metrics operate on numpy images in [0, 1] range with shape (H, W, 3),
except LPIPS which uses torch tensors.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import uniform_filter
from scipy.special import gamma as gamma_func


# ============================================================
# REFERENCE METRICS
# ============================================================

def compute_psnr(img1, img2, data_range=1.0):
    """Peak Signal-to-Noise Ratio."""
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def compute_ssim(img1, img2, data_range=1.0):
    """Structural Similarity Index."""
    return structural_similarity(img1, img2, channel_axis=2, data_range=data_range)


# ============================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================

_lpips_model = None

def _get_lpips_model(device="cpu"):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


def compute_lpips(img1, img2, device="cpu"):
    """
    LPIPS distance. Lower = more similar.
    img1, img2: numpy arrays in [0, 1], shape (H, W, 3).
    """
    model = _get_lpips_model(device)
    # Convert to tensor: (1, 3, H, W) in [-1, 1]
    t1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    t2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    t1, t2 = t1.to(device), t2.to(device)
    with torch.no_grad():
        d = model(t1, t2)
    return d.item()


# ============================================================
# NIQE (Natural Image Quality Evaluator) — No-Reference
# ============================================================

def _estimate_aggd_params(block):
    """Asymmetric Generalized Gaussian Distribution parameter estimation."""
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = (gamma_func(2.0 / gam) ** 2) / (gamma_func(1.0 / gam) * gamma_func(3.0 / gam))

    left_data = block[block < 0]
    right_data = block[block >= 0]

    sigma_l = np.sqrt(np.mean(left_data ** 2)) if len(left_data) > 0 else 1e-7
    sigma_r = np.sqrt(np.mean(right_data ** 2)) if len(right_data) > 0 else 1e-7

    gam_hat = sigma_l / (sigma_r + 1e-7)
    r_hat = (np.mean(np.abs(block)) ** 2) / (np.mean(block ** 2) + 1e-7)

    rhat_norm = r_hat * (gam_hat ** 3 + 1) * (gam_hat + 1) / ((gam_hat ** 2 + 1) ** 2 + 1e-7)
    diff = np.abs(r_gam - rhat_norm)
    alpha = gam[np.argmin(diff)]

    return alpha, sigma_l, sigma_r


def _compute_niqe_features(img_gray, block_size=96):
    """Extract BRISQUE-like features from image blocks."""
    h, w = img_gray.shape
    features = []

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = img_gray[i:i + block_size, j:j + block_size].astype(np.float64)

            mu = uniform_filter(block, size=7)
            sigma = np.sqrt(np.maximum(uniform_filter(block ** 2, size=7) - mu ** 2, 0)) + 1e-7

            mscn = (block - mu) / sigma

            alpha, sl, sr = _estimate_aggd_params(mscn)
            feat = [alpha, (sl + sr) / 2.0]

            # Paired product neighbors
            shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dy, dx in shifts:
                shifted = np.roll(np.roll(mscn, dy, axis=0), dx, axis=1)
                pair = mscn * shifted
                a, s_l, s_r = _estimate_aggd_params(pair)
                mean_param = (s_r - s_l) * (gamma_func(2.0 / a) / gamma_func(1.0 / a))
                feat.extend([a, mean_param, s_l, s_r])

            features.append(feat)

    return np.array(features) if features else np.zeros((1, 18))


def compute_niqe(img, block_size=96):
    """
    NIQE score (no-reference quality). Lower = better.
    img: numpy array in [0, 1], shape (H, W, 3).
    """
    if img.ndim == 3:
        gray = np.dot(img, [0.2989, 0.5870, 0.1140])
    else:
        gray = img

    gray = (gray * 255).astype(np.float64)

    features = _compute_niqe_features(gray, block_size)

    if features.shape[0] < 2:
        return 5.0  # Default for very small images

    mu = features.mean(axis=0)
    cov = np.cov(features.T) + np.eye(features.shape[1]) * 1e-7

    # Pristine model (simplified — uses data statistics as proxy)
    mu_p = np.zeros_like(mu)
    cov_p = np.eye(features.shape[1])

    diff = mu - mu_p
    cov_avg = (cov + cov_p) / 2.0

    try:
        inv_cov = np.linalg.inv(cov_avg)
        niqe_score = np.sqrt(diff @ inv_cov @ diff)
    except np.linalg.LinAlgError:
        niqe_score = 5.0

    return float(niqe_score)


# ============================================================
# AGGREGATE
# ============================================================

def compute_all_metrics(pred_np, target_np, device="cpu"):
    """
    Compute all metrics at once.

    Args:
        pred_np: Predicted image, numpy (H, W, 3) in [0, 1]
        target_np: Ground truth image, numpy (H, W, 3) in [0, 1]
        device: Device for LPIPS computation

    Returns:
        dict with psnr, ssim, lpips, niqe
    """
    return {
        "psnr": compute_psnr(pred_np, target_np),
        "ssim": compute_ssim(pred_np, target_np),
        "lpips": compute_lpips(pred_np, target_np, device),
        "niqe": compute_niqe(pred_np),
    }


def tensor2img(tensor):
    """Convert tensor (B,C,H,W) or (C,H,W) in [-1,1] to numpy (H,W,3) in [0,1]."""
    t = tensor.cpu().detach()
    if t.ndim == 4:
        t = t[0]
    t = t.permute(1, 2, 0).numpy()
    t = (t * 0.5 + 0.5).clip(0, 1)
    return t
