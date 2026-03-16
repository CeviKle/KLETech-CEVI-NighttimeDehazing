"""
Production inference for NightDehazeNet.

Features:
  - Tiled inference with Gaussian-weighted overlap blending
  - Self-ensemble: average over 8 geometric transforms (4 rotations × 2 flips)
  - Optional TTA for competition submissions
"""

import os
import math
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from torchvision.utils import save_image

from config import get_config
from model import build_model
from metrics import compute_psnr, compute_ssim, compute_lpips, compute_niqe, tensor2img
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# GAUSSIAN WEIGHT MAP
# ============================================================

def create_gaussian_weight(tile_size, sigma_factor=0.25):
    """Create 2D Gaussian weight map for tile blending."""
    sigma = tile_size * sigma_factor
    center = tile_size / 2.0
    y = torch.arange(tile_size, dtype=torch.float32)
    x = torch.arange(tile_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    weight = torch.exp(-((yy - center) ** 2 + (xx - center) ** 2) / (2 * sigma ** 2))
    weight = weight / weight.max()
    weight = weight.clamp(min=0.01)  # Prevent zeros at edges
    return weight  # (H, W)


# ============================================================
# TILED INFERENCE
# ============================================================

class SeamlessTiledInference:
    """
    Advanced sliding-window inference with:
      - Gaussian-weighted overlap blending
      - Global Mean Matching (to fix the 'darkness trap')
    """

    def __init__(self, model, tile_size=512, overlap=128, device='cuda'):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.weight = create_gaussian_weight(tile_size).to(device)

    @torch.no_grad()
    def __call__(self, img_tensor, mean_match=True):
        """
        Args:
            img_tensor: (1, 3, H, W) normalized to [-1, 1]
            mean_match: Adjust output mean to match input mean (preserves brightness)
        """
        b, c, h, w = img_tensor.shape
        ts = self.tile_size
        
        # Calculate hazy input mean for brightness reference
        hazy_mean = (img_tensor * 0.5 + 0.5).mean()

        # If image fits in one tile
        if h <= ts and w <= ts:
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16
            
            if pad_h > 0 or pad_w > 0:
                img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                img_padded = img_tensor
                
            result = self.model(img_padded)
            pred_padded = result[0] if isinstance(result, tuple) else result
            
            # Strip the padding
            pred = pred_padded[:, :, :h, :w]
            
            if mean_match:
                # Denormalize, match mean, re-normalize
                p = pred * 0.5 + 0.5
                curr_mean = p.mean()
                # We don't want to fully match (haze is brighter), so we use a target shift
                # target = hazy_mean - 0.05 is a safe bet for nighttime
                target = max(hazy_mean - 0.08, curr_mean) 
                p = (p * (target / (curr_mean + 1e-6))).clamp(0, 1)
                pred = (p - 0.5) / 0.5
            return pred

        stride = ts - self.overlap
        output = torch.zeros_like(img_tensor)
        weight_map = torch.zeros((1, 1, h, w), device=self.device)

        # Coordinate calculation
        h_starts = []
        for i in range(0, h - ts + 1, stride):
            h_starts.append(i)
        if not h_starts:
            h_starts.append(0)
        elif h_starts[-1] + ts < h: 
            h_starts.append(h - ts)
        
        w_starts = []
        for j in range(0, w - ts + 1, stride):
            w_starts.append(j)
        if not w_starts:
            w_starts.append(0)
        elif w_starts[-1] + ts < w: 
            w_starts.append(w - ts)

        for h_start in h_starts:
            for w_start in w_starts:
                # Calculate end pos ensuring we don't exceed image bounds
                h_end = min(h_start + ts, h)
                w_end = min(w_start + ts, w)
                
                # If the image dimension is smaller than tile size, crop it but also adjust the inference
                crop = img_tensor[:, :, h_start:h_end, w_start:w_end]
                
                # U-Net architecture requires dimensions to be divisible by 16 (or 32 depending on depth)
                # If this is an edge tile, its size might be an odd number, causing skip connection size mismatches.
                _, _, ch, cw = crop.shape
                pad_h = (16 - (ch % 16)) % 16
                pad_w = (16 - (cw % 16)) % 16
                
                if pad_h > 0 or pad_w > 0:
                    crop_padded = F.pad(crop, (0, pad_w, 0, pad_h), mode='reflect')
                else:
                    crop_padded = crop
                
                result = self.model(crop_padded)
                pred_padded = result[0] if isinstance(result, tuple) else result
                
                # Strip the padding to get the actual prediction
                pred = pred_padded[:, :, :ch, :cw]

                # We need to reshape weight map to match the crop if it's smaller than tile_size
                crop_weight = self.weight[:ch, :cw]

                # Apply weight
                output[:, :, h_start:h_end, w_start:w_end] += pred * crop_weight.unsqueeze(0).unsqueeze(0)
                weight_map[:, :, h_start:h_end, w_start:w_end] += crop_weight.unsqueeze(0).unsqueeze(0)

        merged = output / weight_map.clamp(min=1e-8)
        
        if mean_match:
            # Final global Mean Matching
            p_final = merged * 0.5 + 0.5
            curr_mean = p_final.mean()
            # Empirical target: keep the prediction clean but dont let it get 20% darker than hazy
            target = max(hazy_mean - 0.10, curr_mean)
            p_final = (p_final * (target / (curr_mean + 1e-6))).clamp(0, 1)
            merged = (p_final - 0.5) / 0.5
            
        return merged


# ============================================================
# SELF-ENSEMBLE
# ============================================================

def self_ensemble(model, img_tensor, tiled_inf=None, mean_match=True):
    """
    Average predictions over 8 geometric transforms.
    """
    predictions = []
    
    def _do_infer(x):
        if tiled_inf:
            return tiled_inf(x, mean_match=mean_match)
        res = model(x)
        return res[0] if isinstance(res, tuple) else res

    for rot in range(4):
        # Rotate
        x = torch.rot90(img_tensor, rot, dims=[2, 3])
        pred = _do_infer(x)
        pred = torch.rot90(pred, -rot, dims=[2, 3])
        predictions.append(pred)

        # Flip
        x_f = torch.flip(x, dims=[3])
        pred_f = _do_infer(x_f)
        pred_f = torch.flip(pred_f, dims=[3])
        pred_f = torch.rot90(pred_f, -rot, dims=[2, 3])
        predictions.append(pred_f)

    return torch.stack(predictions).mean(dim=0)


# ============================================================
# MAIN INFERENCE
# ============================================================

def run_inference(args):
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if args.dim:
        cfg.model.dim = args.dim
    model = build_model(cfg.model).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
              f"PSNR={ckpt.get('psnr', '?')})")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Setup inference
    tiled_inf = SeamlessTiledInference(
        model, tile_size=args.tile_size, overlap=args.overlap, device=device
    )

    # Input preprocessing
    transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    # Find images
    input_paths = sorted(
        glob(os.path.join(args.input_dir, "*.png")) +
        glob(os.path.join(args.input_dir, "*.jpg"))
    )
    print(f"Processing {len(input_paths)} images...")

    os.makedirs(args.output_dir, exist_ok=True)

    # GT paths for metric computation (if available)
    gt_paths = None
    if args.gt_dir and os.path.isdir(args.gt_dir):
        gt_paths = sorted(
            glob(os.path.join(args.gt_dir, "*.png")) +
            glob(os.path.join(args.gt_dir, "*.jpg"))
        )

    all_metrics = {"psnr": [], "ssim": [], "lpips": [], "niqe": []}
    mean_match = not args.no_mean_match

    with torch.no_grad():
        for i, path in enumerate(tqdm(input_paths, desc="Inference")):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            inp = transform(image=img)['image'].unsqueeze(0).to(device)

            if args.self_ensemble:
                pred = self_ensemble(model, inp, tiled_inf, mean_match=mean_match)
            else:
                pred = tiled_inf(inp, mean_match=mean_match)

            # Denormalize to [0, 1]
            pred = (pred * 0.5 + 0.5).clamp(0, 1)

            # Save
            out_name = os.path.basename(path)
            save_image(pred, os.path.join(args.output_dir, out_name))

            # Metrics
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            niqe_score = compute_niqe(pred_np)
            all_metrics["niqe"].append(niqe_score)

            if gt_paths and i < len(gt_paths):
                gt_img = cv2.cvtColor(cv2.imread(gt_paths[i]), cv2.COLOR_BGR2RGB)
                gt_np = gt_img.astype(np.float32) / 255.0
                all_metrics["psnr"].append(compute_psnr(pred_np, gt_np))
                all_metrics["ssim"].append(compute_ssim(pred_np, gt_np))
                all_metrics["lpips"].append(compute_lpips(pred_np, gt_np, device))

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Mean Correction: {'ENABLED' if mean_match else 'DISABLED'}")
    if all_metrics["psnr"]:
        print(f"  PSNR:  {np.mean(all_metrics['psnr']):.2f} ± {np.std(all_metrics['psnr']):.2f}")
        print(f"  SSIM:  {np.mean(all_metrics['ssim']):.4f} ± {np.std(all_metrics['ssim']):.4f}")
        print(f"  LPIPS: {np.mean(all_metrics['lpips']):.4f} ± {np.std(all_metrics['lpips']):.4f}")
    print(f"  NIQE:  {np.mean(all_metrics['niqe']):.2f} ± {np.std(all_metrics['niqe']):.2f}")
    print(f"\nSaved {len(input_paths)} results to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NightDehaze_v3 Inference")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--gt_dir", type=str, default=None, help="GT dir for metrics")
    parser.add_argument("--checkpoint", type=str, default="./experiments/checkpoints/best_model.pth")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--self_ensemble", action="store_true")
    parser.add_argument("--no_mean_match", action="store_true", help="Disable brightness/color fix")
    args = parser.parse_args()
    run_inference(args)
