import os
import argparse
import math
import cv2
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_config
from model import build_model
from losses import CompositeLoss
from dataset import NightHazeDataset, create_internal_split
from metrics import compute_psnr, compute_ssim, tensor2img
from train import predict_tiled



class EMACallback(Callback):
    """
    Lightning Callback for Exponential Moving Average of model parameters.
    Updated every batch, swapped in during validation, swapped out for training.
    """
    def __init__(self, decay=0.9995):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def on_fit_start(self, trainer, pl_module):
        """Initialize shadow weights before training or sanity check starts."""
        self.shadow = []
        self.params = []
        for param in pl_module.model.parameters():
            if param.requires_grad:
                self.shadow.append(param.data.clone().to(param.device))
                self.params.append(param)
        self.initialized = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not hasattr(self, 'initialized'): return
        # Optimized in-place update using lerp_
        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                s.lerp_(p.data, 1.0 - self.decay)

    def on_validation_start(self, trainer, pl_module):
        """Swap in the EMA weights for validation."""
        if not hasattr(self, 'initialized'): return
        self.backup = []
        for s, p in zip(self.shadow, self.params):
            self.backup.append(p.data.clone())
            p.data.copy_(s)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save snapshots every epoch if explicitly requested or if no validation is running."""
        # Check if we should save every epoch
        save_every = getattr(pl_module.cfg.train, 'save_every_epoch', False)
        
        # If no validation set is provided, we default to saving every epoch to allow manual selection
        if pl_module.trainer.val_dataloaders is None:
            save_every = True
            
        if save_every and trainer.is_global_zero:
            stage_dir = os.path.join(pl_module.cfg.train.save_dir, f'stage_{pl_module.stage}')
            os.makedirs(stage_dir, exist_ok=True)
            
            # Use current_psnr from logs if available, else 0.0
            psnr = trainer.callback_metrics.get('val_psnr', 0.0)
            psnr_val = psnr.item() if hasattr(psnr, 'item') else float(psnr)
            
            torch.save({
                'stage': pl_module.stage,
                'epoch': pl_module.current_epoch,
                'model_state_dict': pl_module.model.state_dict(),
                'psnr': psnr_val,
            }, os.path.join(stage_dir, f'epoch_{pl_module.current_epoch:03d}.pth'))
            print(f"  -> Saved Snapshot: epoch_{pl_module.current_epoch:03d}.pth (Train Epoch End)")

    def on_validation_end(self, trainer, pl_module):
        """Restore original weights and save classic 'best' checkpoint."""
        if not hasattr(self, 'backup') or not self.backup: return 
        
        # Save best model based on validation metrics
        current_psnr = trainer.callback_metrics.get('val_psnr', 0.0)
        best_psnr = getattr(self, 'best_val_psnr', -1.0)
        
        if current_psnr > best_psnr:
            self.best_val_psnr = current_psnr.item() if hasattr(current_psnr, 'item') else current_psnr
            if trainer.is_global_zero:
                stage_dir = os.path.join(pl_module.cfg.train.save_dir, f'stage_{pl_module.stage}')
                os.makedirs(stage_dir, exist_ok=True)
                torch.save({
                    'stage': pl_module.stage,
                    'epoch': pl_module.current_epoch,
                    'model_state_dict': pl_module.model.state_dict(),
                    'psnr': self.best_val_psnr,
                }, os.path.join(stage_dir, 'best_model.pth'))
                print(f"  ★ New Best Classic Model Saved! (PSNR: {self.best_val_psnr:.2f})")

        # Restore
        for b, p in zip(self.backup, self.params):
            p.data.copy_(b)
        self.backup = []


class NightDehazeDataModule(pl.LightningDataModule):
    """
    LightningDataModule to handle dataset splits and Dataloaders cleanly.
    """
    def __init__(self, cfg, csv_path, val_csv=None, stage=1, batch_size=12, patch_size=256, repeat=1):
        super().__init__()
        self.cfg = cfg
        self.csv_path = csv_path
        self.val_csv = val_csv
        self.stage = stage
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.repeat = repeat

    def setup(self, stage=None):
        # Use 100% of the training data if an explicit validation CSV is provided
        effective_split_ratio = 1.0 if (self.val_csv and os.path.exists(self.val_csv)) else self.cfg.train.val_split_ratio
        
        res = create_internal_split(self.csv_path, split_ratio=effective_split_ratio, seed=self.cfg.train.seed)
        (train_h, train_g, train_t), (val_h, val_g, val_t) = res
        
        # Override val split if an explicit val_csv is provided
        if self.val_csv and os.path.exists(self.val_csv):
            res_v = create_internal_split(self.val_csv, split_ratio=0.0, seed=self.cfg.train.seed)
            _, (val_h, val_g, val_t) = res_v

        self.train_ds = NightHazeDataset(
            train_h, train_g, train_t, split='train',
            patch_size=self.patch_size, repeat=self.repeat,
            cfg=self.cfg.augment, stage=self.stage
        )
        
        self.val_ds = NightHazeDataset(
            val_h, val_g, val_t, split='val', 
            patch_size=self.patch_size, repeat=1, stage=self.stage
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=12, pin_memory=True, drop_last=True,
            prefetch_factor=2, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=1, shuffle=False, 
            num_workers=2, pin_memory=True
        )


class NightDehazeLightning(pl.LightningModule):
    """
    The core LightningModule defining the training and validation logic.
    """
    def __init__(self, cfg, stage=1, lr=2e-4, frozen_layers=0):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])
        self.cfg = cfg
        self.stage = stage
        self.lr = lr
        
        # Instantiate Model and Loss
        self.model = build_model(cfg.model)
        self.criterion = CompositeLoss(cfg.loss)

        # Stage 2/3: Freeze early layers optionally
        if frozen_layers > 0:
            print(f"Freezing first {frozen_layers} encoder layers for adaptation...")
            if hasattr(self.model, 'encoder_layers'):
                for i in range(min(frozen_layers, len(self.model.encoder_layers))):
                    for param in self.model.encoder_layers[i].parameters():
                        param.requires_grad = False
        
        # Buffer for per-image validation metrics
        self.val_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        hazy, gt, trans = batch
        result = self.model(hazy)
        
        # Calculate loss (supporting auxiliary illumination map if enabled)
        if isinstance(result, tuple):
            pred, illum = result
            loss = self.criterion(pred, gt, illum)
        else:
            loss = self.criterion(result, gt)
            
        # Logging to TensorBoard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=hazy.shape[0])
        return loss

    def on_validation_epoch_start(self):
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        vh, vg, vt, name = batch
        # Use 512 tile size to perfectly match the inference.py script's evaluation baseline
        pred = predict_tiled(self.model, vh, tile_size=512)
        
        pred_img = tensor2img(pred)
        gt_img = tensor2img(vg)
        # Check if GT is real or just a dummy ZERO-tensor created by dataset.py
        gt_is_real = gt_img.max() > 0.01
        
        if gt_is_real:
            # Calculate individual Metrics
            psnr = compute_psnr(pred_img, gt_img)
            ssim = compute_ssim(pred_img, gt_img)
        else:
            # Missing GT: Output zeros so Logger doesn't crash 
            # (Users must upload resulting images to NTIRE site for true score)
            psnr, ssim = 0.0, 0.0
            if batch_idx == 0:
                print("\n  [!] Validation GT is missing. Outputs logged, but metrics will report 0.0.")
        
        # Log to progress bar and logger (Average handled by Lightning)
        self.log('val_psnr', psnr, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_ssim', ssim, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Store for file logging
        img_name = name[0] if isinstance(name, (list, tuple)) else name
        self.val_step_outputs.append({
            'name': img_name,
            'psnr': float(psnr),
            'ssim': float(ssim),
            'is_real': gt_is_real
        })
        
        return None

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return

        # Save to text file
        epoch = self.current_epoch
        stage_logs = os.path.join(self.cfg.train.save_dir, f"stage_{self.stage}_metrics")
        os.makedirs(stage_logs, exist_ok=True)
        
        log_file = os.path.join(stage_logs, f"val_metrics_epoch_{epoch}.txt")
        
        # Sort by name for easier comparison
        self.val_step_outputs.sort(key=lambda x: x['name'])
        
        # Only calculate average over real GTs to avoid polluting log files
        real_outputs = [x for x in self.val_step_outputs if x['is_real']]
        
        if real_outputs:
            avg_psnr = np.mean([x['psnr'] for x in real_outputs])
            avg_ssim = np.mean([x['ssim'] for x in real_outputs])
        else:
            avg_psnr, avg_ssim = 0.0, 0.0

        with open(log_file, 'w') as f:
            f.write(f"Validation Metrics - Epoch {epoch}\n")
            f.write(f"{'='*40}\n")
            if real_outputs:
                f.write(f"Average PSNR: {avg_psnr:.4f}\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            else:
                f.write("Evaluation against NTIRE Server needed (Missing GTs)\n")
            f.write(f"{'='*40}\n\n")
            
            if real_outputs:
                f.write(f"{'Image Name':<40} | {'PSNR':<10} | {'SSIM':<10}\n")
                f.write(f"{'-'*40}-|------------|-----------\n")
                for out in real_outputs:
                    f.write(f"{out['name']:<40} | {out['psnr']:<10.4f} | {out['ssim']:<10.4f}\n")
        
        if self.trainer.is_global_zero:
            print(f" [LOG] Validation results saved to: {log_file}")
            if real_outputs:
                print(f" [VAL] Epoch {epoch} Average -> PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")
            else:
                print(f" [VAL] Epoch {epoch} Complete. Upload {stage_logs} output to NTIRE for score.")
        
        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.lr, weight_decay=self.cfg.train.weight_decay, betas=self.cfg.train.betas
        )

        # Use Trainer's automated calculation of total training steps
        total_iters = self.trainer.estimated_stepping_batches
        # Decrease warmup if resuming
        warmup_iters = 0 if self.stage == 1 and self.lr < 1e-4 else 200
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_iters)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_iters - warmup_iters), eta_min=1e-6)
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def main():
    # Optimization for RTX 3090 Tensor Cores
    import torch
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser(description="PyTorch Lightning Trainer for NightDehazeNet")
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV index")
    parser.add_argument("--val_csv", type=str, default=None, help="Path to validation CSV index (optional)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Curriculum Stage")
    parser.add_argument("--resume", type=str, default=None, help="Path to classic checkpoint .pth file")
    
    # Overrides for specific stages
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=None, help="Override default dataset repeat factor")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--save_every_epoch", action="store_true", help="Force saving a checkpoint at every validation epoch")
    
    args = parser.parse_args()

    cfg = get_config()
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.save_every_epoch = args.save_every_epoch
    
    # Defaults per stage to match your curriculum
    if args.stage == 1:
        lr = 2e-4
        repeat = 1
        frozen_layers = 0
        cfg.model.use_checkpoint = True
        
        # Balanced weights for Stage 1 (Prioritize PSNR)
        cfg.loss.use_perceptual = True
        cfg.loss.ssim_weight = 0.05       # Reduced to keep PSNR stable
        cfg.loss.perceptual_weight = 0.01  # Reduced from 0.04
        cfg.train.grad_accum_steps = 1
        
        # If resuming from a mature model, use a smaller LR and skip warmup
        if args.resume:
            lr = 5e-5
            print(f"  [RESUME] Detected existing model. Setting lower LR: {lr} and skipping warmup.")
        
        print(f"  [INFO] Stage 1: Balanced Loss Suite. LR: {lr}")
    elif args.stage == 2:
        lr = 1e-4
        repeat = 20
        frozen_layers = 2
        cfg.model.use_checkpoint = False
        
        # Enforce balanced weights for stability during domain adaptation
        cfg.loss.ssim_weight = 0.05
        cfg.loss.perceptual_weight = 0.01
        
        if args.resume:
            lr = 5e-5 # Crucial: Don't shock the model when loading
            print(f"  [RESUME] Stage 2: Lowering LR to {lr} to prevent forgetting.")
            
    else:  # Stage 3
        lr = 2e-5
        repeat = 30       # Reduced from 50 to prevent over-hammering tiny dataset
        frozen_layers = 4 # Freeze deep into the encoder/bottleneck for refinement
        cfg.model.use_checkpoint = False
        
        # Absolute stability for refinement (Prioritize PSNR over guidance)
        cfg.loss.ssim_weight = 0.02 
        cfg.loss.perceptual_weight = 0.005

        if args.resume:
            lr = 5e-6 # Ultra-delicate refinement LR
            print(f"  [STABILITY] Stage 3: Ultra-low LR ({lr}) to prevent divergence.")

    # Override defaults if manually specified
    if args.repeat is not None:
        repeat = args.repeat
        print(f"  [OVERRIDE] Dataset repeat factor set to {repeat}")

    # 1. Setting up Environment constraints
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    pl.seed_everything(cfg.train.seed, workers=True)

    # Automatically resolve the correct validation CSV paths if not explicitly provided
    val_csv = args.val_csv
    if val_csv is None:
        if args.stage == 2:
            print("  [DATA] Stage 2: Using standard validation split (or val_csv if provided)")
        elif args.stage == 3:
            print("  [DATA] Stage 3: Using standard validation split (or val_csv if provided)")

    # 2. Init Data and Core Model
    datamodule = NightDehazeDataModule(
        cfg=cfg, csv_path=args.csv, val_csv=val_csv, 
        stage=args.stage, batch_size=args.batch_size, 
        patch_size=args.patch_size, repeat=repeat
    )

    model = NightDehazeLightning(cfg=cfg, stage=args.stage, lr=lr, frozen_layers=frozen_layers)

    # 3. Load from existing classic .pth checkpoint if requested
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.model.load_state_dict(state_dict)
        print(f"[SUCCESS] Loaded classic PyTorch weights from {args.resume}")

    # 4. Callbacks & Loggers
    ema_callback = EMACallback(decay=cfg.train.ema_decay)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.train.save_dir, f'stage_{args.stage}_lightning'),
        filename='best_model_{epoch}-{val_psnr:.2f}',
        monitor='val_psnr',
        mode='max',
        save_last=True
    )
    
    logger = TensorBoardLogger(save_dir=cfg.train.save_dir, name=f"stage_{args.stage}_lightning_logs")

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="gpu",
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        strategy="auto",
        accumulate_grad_batches=cfg.train.grad_accum_steps if args.stage > 1 else 1,
        gradient_clip_val=cfg.train.grad_clip,
        # Disabled checkpoint_callback to prevent massive I/O bottleneck on NFS (EMACallback handles it)
        callbacks=[ema_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50, # Log less frequent for speed
        num_sanity_val_steps=0,
        enable_checkpointing=False # Fully disable Lightning's built-in checkpointer
    )

    # 6. Fit / Train
    print(f"\n🚀 Starting Lightning Training (Stage {args.stage})")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
