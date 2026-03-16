"""
Advanced data pipeline for nighttime image dehazing.

Features:
  - Strong geometric + photometric augmentations
  - MixUp and CutMix for pair-consistent blending
  - Synthetic haze injection (atmospheric scattering model)
  - Progressive patch resizing
  - Self-supervised masked image modeling
  - High repeat factor for small datasets
"""

import os
import random
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# DATA LOADING UTILS
# ============================================================

def create_internal_split(csv_path, split_ratio=0.8, seed=42):
    """Split paired data into train/val sets using CSV file (NFS-safe)."""
    import csv as csv_mod

    hazy_imgs = []
    gt_imgs = []
    trans_imgs = []

    with open(csv_path, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            hazy_imgs.append(row['hazy_path'])
            gt_imgs.append(row['gt_path'])
            # Get trans if it exists, else 'None'
            trans_imgs.append(row.get('trans_path', 'None'))

    assert len(hazy_imgs) > 0, f"No entries found in {csv_path}"

    rng = np.random.RandomState(seed)
    indices = np.arange(len(hazy_imgs))
    rng.shuffle(indices)

    split_idx = int(len(hazy_imgs) * split_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_h = [hazy_imgs[i] for i in train_idx]
    train_g = [gt_imgs[i] for i in train_idx]
    train_t = [trans_imgs[i] for i in train_idx]
    
    val_h = [hazy_imgs[i] for i in val_idx]
    val_g = [gt_imgs[i] for i in val_idx]
    val_t = [trans_imgs[i] for i in val_idx]

    print(f"CSV: {csv_path} → {len(train_h)} train, {len(val_h)} val")
    return (train_h, train_g, train_t), (val_h, val_g, val_t)


def get_all_paths_from_csv(csv_path):
    """Get all image paths from CSV (for self-supervised phase)."""
    import csv as csv_mod
    paths = []
    with open(csv_path, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            paths.append(row['hazy_path'])
            paths.append(row['gt_path'])
            if row.get('trans_path', 'None') != 'None':
                paths.append(row['trans_path'])
    return sorted(set(paths))


# ============================================================
# SYNTHETIC HAZE INJECTION
# ============================================================

def inject_synthetic_haze(img, beta_range=(0.3, 1.2), A_range=(0.6, 0.95)):
    """
    Atmospheric scattering model: I(x) = J(x)*t(x) + A*(1 - t(x))
    where t(x) = exp(-beta * d(x)), d(x) is a random depth map.
    Applies to a [0, 1] float image.
    """
    h, w = img.shape[:2]
    # Generate random depth map at very small resolution (e.g. 1/32)
    small_h, small_w = max(h // 32, 2), max(w // 32, 2)
    depth_small = np.random.uniform(0.3, 1.5, (small_h, small_w)).astype(np.float32)

    # Blur at the small scale (drastically faster)
    depth_small = cv2.GaussianBlur(depth_small, (0, 0), sigmaX=3, sigmaY=3)

    # Resize back to full resolution
    depth = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_CUBIC)

    beta = np.random.uniform(*beta_range)
    A = np.random.uniform(*A_range)

    transmission = np.exp(-beta * depth)[..., np.newaxis]
    hazy = img * transmission + A * (1 - transmission)

    # Add subtle glow around bright regions (night-specific)
    # Downsample for faster processing
    glow_h, glow_w = max(h // 8, 4), max(w // 8, 4)
    img_small = cv2.resize(img, (glow_w, glow_h), interpolation=cv2.INTER_LINEAR)
    bright_mask_small = (np.mean(img_small, axis=2) > 0.6).astype(np.float32)

    if bright_mask_small.any():
        glow_small = cv2.GaussianBlur(bright_mask_small, (0, 0), sigmaX=15)
        glow = cv2.resize(glow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        glow_intensity = np.random.uniform(0.02, 0.08)
        glow_color = np.random.uniform(0.7, 1.0, 3).astype(np.float32)
        hazy = hazy + glow_intensity * glow[..., np.newaxis] * glow_color

    # Add Poisson shot noise (high-ISO nighttime camera simulation)
    # Scale controls ISO intensity. Lower scale = more noise chunkiness.
    signal = np.clip(hazy * 255.0, 0, 255.0)
    noise_scale = np.random.uniform(0.5, 2.5) 
    hazy = np.random.poisson(signal / noise_scale) * noise_scale / 255.0

    return np.clip(hazy, 0, 1).astype(np.float32)


# ============================================================
# MIXUP / CUTMIX
# ============================================================

def mixup_pair(hazy1, gt1, hazy2, gt2, alpha_low=0.7):
    """MixUp two paired samples."""
    lam = np.random.uniform(alpha_low, 1.0)
    hazy_mix = lam * hazy1 + (1 - lam) * hazy2
    gt_mix = lam * gt1 + (1 - lam) * gt2
    return hazy_mix.astype(np.float32), gt_mix.astype(np.float32)


def cutmix_pair(hazy1, gt1, hazy2, gt2, min_ratio=0.2, max_ratio=0.5):
    """CutMix: paste rectangular region from pair 2 into pair 1."""
    h, w = hazy1.shape[:2]
    ratio = np.random.uniform(min_ratio, max_ratio)
    cut_h = int(h * ratio)
    cut_w = int(w * ratio)
    y = np.random.randint(0, h - cut_h + 1)
    x = np.random.randint(0, w - cut_w + 1)

    hazy_out = hazy1.copy()
    gt_out = gt1.copy()
    hazy_out[y:y + cut_h, x:x + cut_w] = hazy2[y:y + cut_h, x:x + cut_w]
    gt_out[y:y + cut_h, x:x + cut_w] = gt2[y:y + cut_h, x:x + cut_w]
    return hazy_out, gt_out


def apply_nightfall(img):
    """
    Transform daytime clear image to look like a night scene.
    1. Inverse Gamma (darken)
    2. Color Temperature Shift (sodium vapor hint)
    3. Point Light Glows
    """
    # 1. Darken (Random Gamma between 1.8 and 2.8)
    gamma = random.uniform(1.8, 2.8)
    img = np.power(img, gamma)

    # 2. Color Shift (Classic sodium vapor tint)
    tint = np.array([1.0, 0.85, 0.6], dtype=np.float32)
    img = img * tint

    # 3. Simulate localized light sources (if bright spots exist)
    h, w = img.shape[:2]
    # Use a high threshold for "potential lights"
    bright_mask = (np.mean(img, axis=2) > 0.4).astype(np.float32)
    if bright_mask.any():
        tint = np.array([1.0, 0.85, 0.6], dtype=np.float32)
        num_glows = random.randint(1, 3)
        for _ in range(num_glows):
            # Choose a random bright spot
            coords = np.argwhere(bright_mask > 0)
            if len(coords) > 0:
                center = coords[random.randint(0, len(coords)-1)]
                glow_size = random.randint(30, 80)
                glow_mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(glow_mask, (center[1], center[0]), glow_size, 1.0, -1)
                glow_mask = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=glow_size/2)
                intensity = random.uniform(0.1, 0.3)
                img = img + (glow_mask[..., np.newaxis] * intensity * tint)

    return np.clip(img, 0, 1).astype(np.float32)


# ============================================================
# NIGHT HAZE DATASET
# ============================================================

class NightHazeDataset(Dataset):
    """
    Advanced paired dataset for nighttime dehazing with:
      - Stage-specific augmentations (Nightfall for Stage 1)
      - Pair-consistent geometric + photometric augmentations
      - MixUp / CutMix
      - High-efficiency loading (no caching for >500 images)
      - Transmission Map support
    """

    def __init__(self, hazy_paths, gt_paths, trans_paths=None, split='train', 
                  patch_size=256, repeat=1, cfg=None, stage=1):
        super().__init__()
        self.hazy_paths = hazy_paths
        self.gt_paths = gt_paths
        self.trans_paths = trans_paths # List of paths or None
        self.split = split
        self.patch_size = patch_size
        self.repeat = repeat
        self.stage = stage

        # Config for augmentation toggles
        self.use_mixup = getattr(cfg, 'use_mixup', True) if cfg else True
        self.mixup_alpha = getattr(cfg, 'mixup_alpha', 0.7) if cfg else 0.7
        self.use_cutmix = getattr(cfg, 'use_cutmix', True) if cfg else True
        self.cutmix_prob = getattr(cfg, 'cutmix_prob', 0.3) if cfg else 0.3
        self.use_haze_synth = getattr(cfg, 'use_haze_synth', True) if cfg else True

        # Build transform pipelines
        # Use image, image0 (gt), and image2 (trans) for consistency
        self.train_transform = A.Compose([
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            ], p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        ], additional_targets={'image0': 'image', 'image2': 'mask'}) # Crop + Geometric

        self.normalize_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ], additional_targets={'image0': 'image', 'image2': 'mask'})

        self.val_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ], additional_targets={'image0': 'image', 'image2': 'mask'})

        # Pre-load images ONLY for tiny datasets
        self._cache = {}
        if len(self.hazy_paths) <= 50:
            print(f"Pre-caching {len(self.hazy_paths)} image triplets...")
            for i in range(len(self.hazy_paths)):
                self._cache[i] = self._load_pair_no_cache(i)

    def _load_pair_no_cache(self, idx):
        try:
            h = cv2.imread(self.hazy_paths[idx])
            if h is None:
                raise ValueError(f"Hazy image {self.hazy_paths[idx]} is corrupted.")
            
            g_path = self.gt_paths[idx]
            if g_path != "None" and os.path.exists(g_path):
                g = cv2.imread(g_path)
                if g is not None:
                    g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
                else:
                    g = np.zeros_like(h)
            else:
                g = np.zeros_like(h)
            
            t_path = self.trans_paths[idx] if self.trans_paths else "None"
            if t_path != "None" and os.path.exists(t_path):
                t = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
                if t is None: t = np.ones((h.shape[0], h.shape[1]), dtype=np.uint8) * 255
            else:
                t = np.ones((h.shape[0], h.shape[1]), dtype=np.uint8) * 255 
            
            h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
            return h, g, t
        except Exception as e:
            raise e

    def _load_pair(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        return self._load_pair_no_cache(idx)

    def __len__(self):
        return len(self.hazy_paths) * self.repeat

    def __getitem__(self, index):
        try:
            return self._get_single_item(index)
        except Exception as e:
            idx = index % len(self.hazy_paths)
            print(f"\n[!] Error loading sample {idx} ({self.hazy_paths[idx]}): {e}")
            # Try a random different index to avoid infinite crash loops
            new_idx = random.randint(0, len(self.hazy_paths) - 1)
            return self.__getitem__(new_idx)

    def _get_single_item(self, index):
        idx = index % len(self.hazy_paths)
        hazy_img, gt_img, trans_img = self._load_pair(idx)

        if self.split == 'train':
            # 1. Apply geometric transforms + crop FIRST (Crucial for speed!)
            augmented = self.train_transform(image=hazy_img, image0=gt_img, image2=trans_img)
            h_crop, g_crop, t_crop = augmented['image'], augmented['image0'], augmented['image2']

            # --- STAGE 1 & 2 SPECIFIC: NIGHTFALL (on CROP only) ---
            # For Stage 1: Always apply nightfall (RESIDE is daytime)
            # For Stage 2: Detect if image is already nighttime (GTA5) or daytime (NH-Haze)
            #   - If GT mean brightness > 0.35 → daytime → apply nightfall + synthetic haze
            #   - If GT mean brightness <= 0.35 → already night (GTA5) → use real pair as-is
            if self.stage in [1, 2]:
                gt_float = g_crop.astype(np.float32) / 255.0
                gt_brightness = gt_float.mean()

                # Only apply nightfall to daytime images; nighttime pairs (GTA5) pass through
                is_daytime = (self.stage == 1) or (gt_brightness > 0.35)

                if is_daytime:
                    gt_night = apply_nightfall(gt_float)
                    
                    # Synthetic haze on crop
                    if self.trans_paths and self.trans_paths[idx] != "None":
                        t_float = t_crop.astype(np.float32) / 255.0
                        beta = random.uniform(0.5, 1.5)
                        A = random.uniform(0.7, 1.0)
                        t_p = np.exp(-beta * (1.0 - t_float))[..., np.newaxis]
                        hazy_night = gt_night * t_p + A * (1.0 - t_p)
                    else:
                        hazy_night = inject_synthetic_haze(gt_night)
                    
                    h_crop = (hazy_night * 255).astype(np.uint8)
                    g_crop = (gt_night * 255).astype(np.uint8)
                # else: GTA5 nighttime pair → use h_crop and g_crop as-is from CSV

            # --- GENERAL AUGMENTATIONS ---
            # NOTE: For Stage 1 we prioritize speed over complex MixUp

            # Final Normalize and tensorize
            final = self.normalize_transform(image=h_crop, image0=g_crop, image2=t_crop)
            h, g, t = final['image'], final['image0'], final['image2']
            if t.dim() == 2: t = t.unsqueeze(0)
            return h, g, t
        else:
            augmented = self.val_transform(image=hazy_img, image0=gt_img, image2=trans_img)
            h, g, t = augmented['image'], augmented['image0'], augmented['image2']
            if t.dim() == 2: t = t.unsqueeze(0)
            name = os.path.basename(self.hazy_paths[idx])
            return h, g, t, name

    def update_patch_size(self, new_size):
        """For progressive training — update the crop size."""
        self.patch_size = new_size
        # Rebuild train transform with new size
        self.train_transform.transforms[0] = A.PadIfNeeded(
            min_height=new_size, min_width=new_size, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
        self.train_transform.transforms[1] = A.RandomCrop(
            height=new_size, width=new_size, p=1.0)


# ============================================================
# SELF-SUPERVISED DATASET (Masked Image Modeling)
# ============================================================

class MaskedImageDataset(Dataset):
    """
    Self-supervised pretext: mask random patches of hazy images,
    train the encoder to reconstruct masked regions.
    """

    def __init__(self, image_paths, patch_size=256, mask_ratio=0.5, repeat=40):
        super().__init__()
        self.paths = image_paths
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.repeat = repeat

        self.transform = A.Compose([
            A.RandomCrop(height=patch_size, width=patch_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

        # Cache
        self._cache = {}
        if len(self.paths) <= 50:
            for i, p in enumerate(self.paths):
                self._cache[i] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, index):
        idx = index % len(self.paths)
        if idx in self._cache:
            img = self._cache[idx]
        else:
            img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)

        aug = self.transform(image=img)
        clean = aug['image']  # (3, H, W)

        # Create random block mask
        _, h, w = clean.shape
        mask_size = 16  # Block size
        num_blocks_h = h // mask_size
        num_blocks_w = w // mask_size
        num_mask = int(num_blocks_h * num_blocks_w * self.mask_ratio)

        mask = torch.ones(1, h, w)
        indices = list(range(num_blocks_h * num_blocks_w))
        random.shuffle(indices)
        for k in indices[:num_mask]:
            bi = k // num_blocks_w
            bj = k % num_blocks_w
            mask[:, bi * mask_size:(bi + 1) * mask_size,
                 bj * mask_size:(bj + 1) * mask_size] = 0

        masked_input = clean * mask
        return masked_input, clean, mask
