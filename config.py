"""
Centralized configuration for NightDehaze_v3 pipeline.
All hyperparameters in one place for easy ablation and experiment management.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    inp_channels: int = 3
    out_channels: int = 3
    dim: int = 48                           # Base feature dimension
    num_blocks: List[int] = field(default_factory=lambda: [4, 6, 6, 8])
    num_refinement_blocks: int = 4
    heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    expansion_factor: float = 2.66
    use_multiscale_context: bool = True     # Multi-scale dilated conv block
    use_channel_attention: bool = True      # SE-style channel attention
    use_spatial_attention: bool = True       # CBAM-style spatial attention
    use_gated_skip: bool = True             # Gated skip fusion (vs plain concat)
    use_illumination_head: bool = True      # Auxiliary illumination map branch
    use_checkpoint: bool = False            # Gradient Checkpointing (Save VRAM)


@dataclass
class LossConfig:
    charbonnier_weight: float = 1.0
    charbonnier_eps: float = 1e-3
    ssim_weight: float = 0.05               # Restored stable weight for structural metrics
    perceptual_weight: float = 0.01        # Reduced from 0.04
    frequency_weight: float = 0.1          # FFT domain loss
    color_weight: float = 0.05             # Restored stable weight for color preservation
    use_perceptual: bool = True
    use_frequency: bool = True
    use_color: bool = True


@dataclass
class AugmentConfig:
    use_mixup: bool = True
    mixup_alpha: float = 0.7               # MixUp blending range [alpha, 1.0]
    use_cutmix: bool = True
    cutmix_prob: float = 0.3
    use_haze_synth: bool = True            # Synthetic haze injection
    use_channel_shuffle: bool = True
    gamma_limit: tuple = (80, 120)
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15


@dataclass
class TrainConfig:
    # Paths
    train_hazy: str = "/NTIRE2026/C10_NightTimeDehazing/train_inp"
    train_gt: str = "/NTIRE2026/C10_NightTimeDehazing/train_gt"
    val_hazy: str = "/NTIRE2026/C10_NightTimeDehazing/val_inp"
    save_dir: str = "./experiments/checkpoints"
    sample_dir: str = "./experiments/samples"
    log_dir: str = "./experiments/logs"

    # Training schedule
    epochs: int = 300
    batch_size: int = 8                    # 24GB GPU can handle bs=8 with 256 patches
    lr: float = 2e-4
    weight_decay: float = 1e-3
    betas: tuple = (0.9, 0.999)
    warmup_epochs: int = 15
    eta_min: float = 1e-6
    grad_clip: float = 1.0
    grad_accum_steps: int = 2              # Effective batch = 16

    # Progressive training patch sizes
    progressive_patches: List[int] = field(default_factory=lambda: [128, 192, 256, 320])
    
    # EMA
    ema_decay: float = 0.9995

    # Mixed precision
    use_amp: bool = True

    # Validation
    val_interval: int = 10
    val_split_ratio: float = 0.97           # 97% train, 3% internal val

    # Dataset
    repeat_factor: int = 80
    num_workers: int = 6

    # Self-supervised pre-training
    use_self_supervised: bool = True
    ss_epochs: int = 50
    ss_mask_ratio: float = 0.5

    # Seed
    seed: int = 42


@dataclass
class InferenceConfig:
    tile_size: int = 512
    overlap: int = 64
    use_gaussian_weight: bool = True
    use_self_ensemble: bool = True          # 8 geometric transforms
    use_tta: bool = False                   # Additional TTA


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def get_config(**overrides) -> PipelineConfig:
    """Create config with optional overrides for ablation."""
    cfg = PipelineConfig()
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg
