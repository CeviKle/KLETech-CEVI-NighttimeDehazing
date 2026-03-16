"""
Final Submission Generator for NightDehazeNet v3.

After running calibrate_brightness.py to find the best --mean_offset,
use this script to generate the final 25 test predictions with the
calibrated brightness correction applied.

Usage (on SSH):
    python generate_submission.py \
        --input /NTIRE2026/C10_NightTimeDehazing/test_inp \
        --output ./NTIRE_CALIBRATED_SUBMISSION \
        --checkpoint experiments/checkpoints/stage_3/best_model.pth \
        --overlap 160 \
        --mean_offset -0.08       # <-- from calibrate_brightness.py output
        --self_ensemble
"""

import os
import cv2
import torch
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from torchvision.utils import save_image

from model import build_model
from config import get_config
import albumentations as A
from albumentations.pytorch import ToTensorV2
from inference import SeamlessTiledInference, self_ensemble


def apply_manual_offset(pred_np, hazy_np, offset):
    """Apply the calibrated brightness offset."""
    pred_mean = pred_np.mean()
    hazy_mean = hazy_np.mean()
    target = max(hazy_mean + offset, 1e-3)
    scale = target / (pred_mean + 1e-6)
    return np.clip(pred_np * scale, 0, 1).astype(np.float32)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = get_config()
    if args.dim:
        cfg.model.dim = args.dim
    if args.refinement_blocks:
        cfg.model.num_refinement_blocks = args.refinement_blocks
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, PSNR={ckpt.get('psnr','?')})")
    model.eval()

    tiled_inf = SeamlessTiledInference(
        model, tile_size=args.tile_size, overlap=args.overlap, device=device
    )

    transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    input_paths = sorted(
        glob(os.path.join(args.input, "*.png")) +
        glob(os.path.join(args.input, "*.jpg"))
    )
    print(f"Processing {len(input_paths)} images...")
    print(f"  Brightness offset: {args.mean_offset if args.mean_offset is not None else 'DISABLED'}")
    print(f"  Self-ensemble: {args.self_ensemble}")

    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for path in tqdm(input_paths, desc="Generating submission"):
            hazy_bgr = cv2.imread(path)
            hazy_rgb = cv2.cvtColor(hazy_bgr, cv2.COLOR_BGR2RGB)
            hazy_np = hazy_rgb.astype(np.float32) / 255.0

            inp = transform(image=hazy_rgb)['image'].unsqueeze(0).to(device)

            if args.self_ensemble:
                pred = self_ensemble(model, inp, tiled_inf, mean_match=False)
            else:
                pred = tiled_inf(inp, mean_match=False)

            pred_np = (pred * 0.5 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

            # Apply calibrated offset
            if args.mean_offset is not None:
                pred_np = apply_manual_offset(pred_np, hazy_np, args.mean_offset)

            # Save
            out_name = os.path.basename(path)
            out_path = os.path.join(args.output, out_name)
            pred_bgr = cv2.cvtColor((pred_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, pred_bgr)

    print(f"\nSaved {len(input_paths)} predictions to '{args.output}'")
    print("Ready to zip and submit!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate calibrated NTIRE submission")
    parser.add_argument("--input", type=str, required=True, help="Path to test input images")
    parser.add_argument("--output", type=str, default="./NTIRE_CALIBRATED_SUBMISSION")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument("--dim", type=int, default=48,
                        help="Model base dimension. Must match the checkpoint (default: 48)")
    parser.add_argument("--refinement_blocks", type=int, default=4,
                        help="Number of refinement blocks. Must match the checkpoint (default: 4)")
    parser.add_argument("--mean_offset", type=float, default=None,
                        help="Calibrated offset from calibrate_brightness.py. None = no correction.")
    parser.add_argument("--self_ensemble", action="store_true")
    args = parser.parse_args()
    main(args)
