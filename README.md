# NTIRE 2026 Challenge on Nighttime Image Dehazing — Team KLETech-CEVI

This repository contains the official code submission for **Team KLETech-CEVI** in the [NTIRE 2026 Nighttime Image Dehazing Challenge](https://codalab.lisn.upsaclay.fr/competitions/).

Our solution, **NightDehazeNet**, is a Multi-Scale Attention U-Net trained via a 3-stage curriculum learning strategy, achieving **PSNR: 20.785 dB** and **SSIM: 0.675** on the challenge validation set.

---

## 🚀 1. Environment Requirements

- **Operating System:** Linux (Ubuntu 20.04/22.04 recommended) or Windows
- **Hardware:** GPU with at least 24GB VRAM (e.g., NVIDIA RTX 3090)
- **Software:**
  - Python >= 3.10
  - PyTorch >= 2.0 (with CUDA support)

---

## 🛠️ 2. Installation Instructions

**Step 1:** Clone this repository

```bash
git clone https://github.com/CeviKle/KLETech-CEVI-NighttimeDehazing.git
cd KLETech-CEVI-NighttimeDehazing
```

**Step 2:** Create and activate a Conda environment

```bash
conda create -n nightdehaze python=3.10 -y
conda activate nightdehaze
```

**Step 3:** Install PyTorch (adjust CUDA version as needed; below is for CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 4:** Install required dependencies

```bash
pip install -r requirements.txt
```

**Step 5:** Verify installation (optional)

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ⚙️ 3. Pretrained Model Weights

Download the pretrained weights from [Google Drive](TODO_ADD_LINK) and place them in the `model_weights/` folder:

```
model_weights/
└── best_model.pth
```

<!-- TODO: Replace the Google Drive link above with the actual link -->

---

## 🏃 4. Running the Inference (Testing)

### Reproduce the submitted results:

```bash
python generate_submission.py \
    --input /path/to/test_images \
    --output ./output_results \
    --checkpoint model_weights/best_model.pth \
    --tile_size 512 \
    --overlap 160 \
    --self_ensemble
```

### Parameter Breakdown:

| Parameter | Value | Description |
|---|---|---|
| `--input` | Path to test images | Directory containing hazy nighttime input images |
| `--output` | `./output_results` | Directory to save dehazed output images |
| `--checkpoint` | `model_weights/best_model.pth` | Path to the trained model weights |
| `--tile_size` | `512` | Tile size for seamless tiled inference |
| `--overlap` | `160` | Overlap between tiles for Gaussian blending |
| `--self_ensemble` | Flag | Enables 8× geometric self-ensemble for improved quality |

---

## 🏋️ 5. Training (Optional)

Our model is trained using a 3-stage curriculum strategy:

```bash
# Stage 1: Synthetic Pretraining on RESIDE (ITS + SOTS)
python train_lightning.py --csv reside_paths.csv --stage 1 \
    --batch_size 16 --epochs 5 --patch_size 256

# Stage 2: Domain Adaptation on NH-Haze + GTA5 Nighttime
python train_lightning.py --csv nh_haze_paths.csv --val_csv ntire_val_real.csv --stage 2 \
    --resume experiments/checkpoints/stage_1/best_model.pth \
    --batch_size 4 --epochs 15 --patch_size 256 --repeat 20

# Stage 3: NTIRE Refinement on 25 paired images
python train_lightning.py --csv ntire_train_real.csv --val_csv ntire_val_real.csv --stage 3 \
    --resume experiments/checkpoints/stage_2/best_model.pth \
    --batch_size 2 --epochs 50 --patch_size 320 --repeat 30
```

### External Datasets Used:
- **RESIDE ITS + SOTS** — ~13,990 indoor/outdoor haze pairs
- **NH-Haze** — 55 real non-homogeneous haze pairs
- **GTA5 Nighttime** — 864 synthetic nighttime haze pairs

---

## 📊 6. Model Architecture & Fact Sheet Summary

**NightDehazeNet** is a Restormer-inspired multi-scale attention U-Net with the following specifications:

| Attribute | Value |
|---|---|
| Architecture | Multi-Scale Attention U-Net |
| Encoder-Decoder Levels | 4 |
| Transformer Block Counts | [4, 6, 6, 8] |
| Attention Heads | [1, 2, 4, 8] |
| Base Feature Dimension | 48 |
| Total Parameters | 24.03M |
| Attention Mechanisms | SE, CBAM, Transposed Multi-Head, Gated Skip Fusion |
| Context Block | Multi-Scale Dilated Conv (d=1,2,4) |
| Loss Function | Charbonnier + SSIM + VGG Perceptual + FFT + Color (Consistency + Angle) |
| Training Time | ~34 hours (3 stages, single RTX 3090) |
| Inference | Seamless Tiled (512px, 160px overlap) + 8× Self-Ensemble |

For detailed architecture and training descriptions, see the [FactSheet (PDF)](TODO_ADD_LINK).

---

## 📂 7. Repository Structure

```
NTIRE2026-KLETech-CEVI-NighttimeDehazing/
├── model.py                  # NightDehazeNet architecture definition
├── train_lightning.py        # PyTorch Lightning training script
├── generate_submission.py    # Inference script to reproduce submitted results
├── inference.py              # Seamless tiled inference & self-ensemble
├── losses.py                 # Composite loss functions
├── dataset.py                # Dataset loading & augmentation pipeline
├── config.py                 # Model & training configuration
├── metrics.py                # PSNR / SSIM evaluation metrics
├── generate_csv.py           # Dataset CSV path generator
├── run_curriculum.sh         # End-to-end 3-stage training script
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── README.md
└── model_weights/
    └── best_model.pth        # Trained model checkpoint
```

---

## 📜 8. License and Acknowledgements

This repository is released under the [MIT License](LICENSE).

### Acknowledgements
- [Restormer](https://github.com/swz30/Restormer) — Transformer architecture inspiration
- [CBAM](https://github.com/Jongchan/attention-module) — Spatial attention module
- [RESIDE Dataset](https://sites.google.com/view/raboresilientimageresearch/reside) — Synthetic haze training data
- [NH-Haze](https://competitions.codalab.org/competitions/22236) — Real non-homogeneous haze pairs
- NTIRE 2026 Challenge Organizers

