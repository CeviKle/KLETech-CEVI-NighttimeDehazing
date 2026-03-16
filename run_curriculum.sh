# NTIRE 2026 Night Dehazing - Multi-Stage Training Script
# Usage: bash run_curriculum.sh
set -e  # Exit on any error

# Check if CSVs or --index flag is present
GEN_CSV=false
REQUIRED_CSVS=("reside_paths.csv" "nh_haze_paths.csv" "ntire_train.csv" "ntire_val.csv")

for csv in "${REQUIRED_CSVS[@]}"; do
    if [ ! -f "$csv" ]; then
        GEN_CSV=true
        break
    fi
done

if [[ "$*" == *"--index"* ]]; then
    GEN_CSV=true
fi

if [ "$GEN_CSV" = true ]; then
    echo "=== STEP 0: Generating ALL CSV Indices (Parallel) ==="
    python generate_csv.py
    echo ""
else
    echo "=== STEP 0: Skipping CSV generation (Index files found) ==="
    echo "    (Use --index to force regeneration)"
fi

# Verify CSVs exist
for f in "${REQUIRED_CSVS[@]}"; do
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        echo "  OK: $f ($((count-1)) entries)"
    else
        echo "  [!] MISSING: $f"
        exit 1
    fi
done
echo ""

echo "=== STEP 1: RESIDE Pretraining (Stage 1) ==="
echo "(Skipping Stage 1 - Using existing pretrained weights)"
nice -n 10 python train_lightning.py --csv reside_paths.csv --stage 1 --batch_size 16 --epochs 5 --patch_size 256

echo "=== STEP 2: NH-Haze + GTA5 + NTIRE Adaptation (Stage 2) ==="
nice -n 10 python train_lightning.py --csv nh_haze_paths.csv --val_csv ntire_val_real.csv --stage 2 \
    --resume experiments/checkpoints/stage_1/best_model.pth --batch_size 4 --epochs 50 --patch_size 256

echo "=== STEP 3: NTIRE Refinement (Stage 3) ==="
nice -n 10 python train_lightning.py --csv ntire_train_real.csv --val_csv ntire_val_real.csv --stage 3 \
    --resume experiments/checkpoints/stage_2/best_model.pth --batch_size 2 --epochs 50 --patch_size 320

echo "=== CURRICULUM COMPLETE ==="
