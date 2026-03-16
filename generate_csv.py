import os
import csv
import argparse
from pathlib import Path

import subprocess
import signal
import sys

def signal_handler(sig, frame):
    print("\n[!] Ctrl+C detected. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

import re
from concurrent.futures import ThreadPoolExecutor

def extract_index(filename):
    """Extract leading numeric index from a filename.
    Examples:
      '1_1_0.8.png'  -> '1'
      '01_hazy.png'  -> '01'
      '1400.png'     -> '1400'
      '0001_10.png'  -> '0001'
    """
    name = os.path.splitext(filename)[0]
    m = re.match(r'^(\d+)', name)
    return m.group(1) if m else None

def get_files_stream(directory, exts=None, limit=None):
    """Streaming 'find' for NFS to prevent hangs and allow Ctrl+C."""
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    if not os.path.exists(directory):
        print(f"    [!] Skipping: {os.path.basename(directory)} (not found)")
        return []
    
    files = []
    try:
        # '-f' tells find to not sort, which is much faster on NFS
        proc = subprocess.Popen(
            ['find', directory, '-maxdepth', '1', '-type', 'f'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        
        count = 0
        for line in proc.stdout:
            fname = os.path.basename(line.strip())
            if os.path.splitext(fname)[1].lower() in exts:
                files.append(fname)
                count += 1
                if limit and count >= limit: break
        
        proc.terminate()
        return sorted(files)
    except Exception as e:
        print(f"    Streaming scan failed: {e}")
        return []

def scan_subset(hazy_dir, gt_dir, trans_dir=None):
    """Utility to scan h/g/t triplet for a subset."""
    # Auto-detect nested folders (e.g. ITS_clear/ITS_clear/)
    gt_dir_actual = gt_dir
    basename = os.path.basename(gt_dir)
    nested = os.path.join(gt_dir, basename)
    if os.path.isdir(nested):
        print(f"    [Auto-detect] Using nested path: {basename}/{basename}/")
        gt_dir_actual = nested
    
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_h = pool.submit(get_files_stream, hazy_dir)
        f_g = pool.submit(get_files_stream, gt_dir_actual)
        f_t = pool.submit(get_files_stream, trans_dir) if trans_dir else None
        
        h_list = f_h.result()
        g_list = f_g.result()
        t_list = f_t.result() if f_t else []
    
    # Build index -> filename maps for GT and Trans
    gt_by_idx = {}
    for gf in g_list:
        idx = extract_index(gf)
        if idx: gt_by_idx[idx] = gf
    
    trans_by_idx = {}
    for tf in t_list:
        idx = extract_index(tf)
        if idx: trans_by_idx.setdefault(idx, []).append(tf)
    
    rows = []
    for hf in h_list:
        h_idx = extract_index(hf)
        if h_idx and h_idx in gt_by_idx:
            gt_file = gt_by_idx[h_idx]
            # Trans: try exact filename match first, then index match
            if hf in set(t_list):
                t_p = os.path.join(trans_dir, hf)
            elif h_idx in trans_by_idx:
                t_p = os.path.join(trans_dir, trans_by_idx[h_idx][0])
            else:
                t_p = "None"
            rows.append([os.path.join(hazy_dir, hf), os.path.join(gt_dir_actual, gt_file), t_p])
    return rows

def generate_reside_csv(output_csv, reside_base):
    print(f"\n[Parallel Scan] Indexing RESIDE at {reside_base}...")
    all_rows = []
    
    # 1. SOTS Subsets
    for subset in ["indoor", "outdoor"]:
        print(f"  -> SOTS {subset}...")
        all_rows.extend(scan_subset(
            os.path.join(reside_base, "SOTS", subset, "hazy"),
            os.path.join(reside_base, "SOTS", subset, "gt")
        ))

    # 2. ITS Subsets
    print(f"  -> ITS Train (Parallelizing H/G/T scans)...")
    all_rows.extend(scan_subset(
        os.path.join(reside_base, "ITS", "train", "ITS_haze"),
        os.path.join(reside_base, "ITS", "train", "ITS_clear"),
        os.path.join(reside_base, "ITS", "train", "ITS_trans")
    ))
    
    print(f"  -> ITS Val...")
    all_rows.extend(scan_subset(
        os.path.join(reside_base, "ITS", "val", "haze"),
        os.path.join(reside_base, "ITS", "val", "clear"),
        os.path.join(reside_base, "ITS", "val", "trans")
    ))

    if all_rows:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hazy_path', 'gt_path', 'trans_path'])
            writer.writerows(all_rows)
        print(f"DONE: {len(all_rows)} entries written to {output_csv}")
    else:
        print("Warning: No RESIDE entries found.")

def generate_nh_haze_and_gta5_csv(output_csv, nh_base, gta5_base):
    print(f"\n[Parallel Scan] Indexing NH-Haze at {nh_base}...")
    files = get_files_stream(nh_base)
    hazy_files = [f for f in files if "hazy" in f.lower()]
    gt_files = [f for f in files if "gt" in f.lower() or "GT" in f]
    
    # Build index -> GT map
    gt_by_idx = {}
    for gf in gt_files:
        idx = extract_index(gf)
        if idx: gt_by_idx[idx] = gf
    
    rows = []
    for hf in hazy_files:
        h_idx = extract_index(hf)
        if h_idx and h_idx in gt_by_idx:
            rows.append([os.path.join(nh_base, hf), os.path.join(nh_base, gt_by_idx[h_idx]), "None"])

    print(f"  -> Found {len(rows)} NH-Haze pairs.")

    print(f"\n[Parallel Scan] Indexing GTA5 Nighttime at {gta5_base}...")
    gta5_train_foggy = os.path.join(gta5_base, "train", "foggy")
    gta5_train_clean = os.path.join(gta5_base, "train", "clean")
    gta5_test_foggy = os.path.join(gta5_base, "test", "foggy")
    gta5_test_clean = os.path.join(gta5_base, "test", "clean")
    
    gta5_count = 0
    # GTA5 Train
    gta5_files = get_files_stream(gta5_train_foggy)
    for hf in gta5_files:
        rows.append([os.path.join(gta5_train_foggy, hf), os.path.join(gta5_train_clean, hf), "None"])
        gta5_count += 1
    # GTA5 Test
    if os.path.exists(gta5_test_foggy):
        gta5_test_files = get_files_stream(gta5_test_foggy)
        for hf in gta5_test_files:
            rows.append([os.path.join(gta5_test_foggy, hf), os.path.join(gta5_test_clean, hf), "None"])
            gta5_count += 1
            
    print(f"  -> Found {gta5_count} GTA5 pairs.")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hazy_path', 'gt_path', 'trans_path'])
        writer.writerows(rows)
    print(f"DONE: {len(rows)} TOTAL training entries written to {output_csv}")

def generate_ntire_csv(output_csv, base_dir, hazy_folder, gt_folder=None):
    """Generate CSV for NTIRE competition images (train_inp/train_gt/val_inp)."""
    hazy_dir = os.path.join(base_dir, hazy_folder)
    print(f"\n[Parallel Scan] Indexing NTIRE ({hazy_folder}) at {hazy_dir}...")
    
    hazy_files = get_files_stream(hazy_dir)
    
    # Build index -> GT map if GT folder exists
    gt_by_idx = {}
    gt_by_name = set()
    if gt_folder:
        gt_dir = os.path.join(base_dir, gt_folder)
        gt_list = get_files_stream(gt_dir)
        gt_by_name = set(gt_list)
        for gf in gt_list:
            idx = extract_index(gf)
            if idx: gt_by_idx[idx] = gf
    else:
        gt_dir = None
    
    rows = []
    for hf in hazy_files:
        h_path = os.path.join(hazy_dir, hf)
        h_idx = extract_index(hf)
        
        # Match by exact filename first, then by index
        if hf in gt_by_name:
            g_path = os.path.join(gt_dir, hf)
        elif h_idx and h_idx in gt_by_idx:
            g_path = os.path.join(gt_dir, gt_by_idx[h_idx])
        else:
            g_path = "None"
        rows.append([h_path, g_path, "None"])
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hazy_path', 'gt_path', 'trans_path'])
        writer.writerows(rows)
    print(f"DONE: {len(rows)} entries written to {output_csv}")

def normalize_path(path):
    return path.replace('\\', '/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reside_base", type=str, default="/NTIRE2026/C10_NightTimeDehazing/RESIDE")
    parser.add_argument("--nh_base", type=str, default="/NTIRE2026/C10_NightTimeDehazing/NH-HAZE/NH-HAZE")
    parser.add_argument("--ntire_base", type=str, default="/NTIRE2026/C10_NightTimeDehazing")
    parser.add_argument("--gta5_base", type=str, default="/NTIRE2026/C10_NightTimeDehazing/RealNightHaze/GTA5")
    args = parser.parse_args()

    res_base = normalize_path(args.reside_base)
    nh_base = normalize_path(args.nh_base)
    ntire_base = normalize_path(args.ntire_base)
    gta5_base = normalize_path(args.gta5_base)

    with ThreadPoolExecutor(max_workers=4) as main_pool:
        main_pool.submit(generate_reside_csv, "reside_paths.csv", res_base)
        main_pool.submit(generate_nh_haze_and_gta5_csv, "nh_haze_paths.csv", nh_base, gta5_base)
        main_pool.submit(generate_ntire_csv, "ntire_train.csv", ntire_base, "train_inp", "train_gt")
        main_pool.submit(generate_ntire_csv, "ntire_val.csv", ntire_base, "val_inp", None)
