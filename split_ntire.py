import os
import random
import csv

def split_ntire_csv():
    input_csv = "ntire_train.csv"
    train_out_csv = "ntire_train_real.csv"
    val_out_csv = "ntire_val_real.csv"
    val_size = 5

    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv} in the current directory.")
        return

    # Read the original CSV
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(rows) < val_size:
        print(f"Error: {input_csv} has only {len(rows)} rows, need at least {val_size + 1} to split.")
        return

    # Shuffle for random selection
    # Using a fixed seed so the split is reproducible if you run it again
    random.seed(42)
    random.shuffle(rows)

    # Split
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    # Write Validator CSV
    with open(val_out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_rows)

    # Write Training CSV
    with open(train_out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)

    print(f"Successfully split {len(rows)} images from {input_csv}:")
    print(f"  -> {val_out_csv}: {len(val_rows)} images (Used for tracking real metrics in Stage 2/3)")
    print(f"  -> {train_out_csv}: {len(train_rows)} images (Used for training in Stage 3)")

    # Also generate a FULL 25-image CSV for maximum Stage 3 training data
    full_out_csv = "ntire_train_full.csv"
    with open(full_out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)  # All 25 images (before shuffle order doesn't matter)
    print(f"  -> {full_out_csv}: {len(rows)} images (ALL images for Stage 3 full training)")

if __name__ == "__main__":
    split_ntire_csv()
