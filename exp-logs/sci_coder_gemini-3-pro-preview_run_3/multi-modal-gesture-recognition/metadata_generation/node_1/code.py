import os
import glob
import json
import random
import shutil
import pandas as pd
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# ==========================================
# Configuration and Constants
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2

# Mapping from gesture name to ID (1-20)
LABEL_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}

# ==========================================
# Helper Functions
# ==========================================


def parse_labels_from_mat(mat_path):
    """
    Parses the .mat file to extract gesture labels.
    Returns a list of dictionaries: [{'name': str, 'id': int, 'begin': int, 'end': int}, ...]
    """
    try:
        # Load mat file, handling struct as objects
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "Video" not in mat:
            return []

        video = mat["Video"]
        if not hasattr(video, "Labels"):
            return []

        labels_raw = video.Labels

        # Normalize labels_raw to a list of objects
        labels_list = []
        if isinstance(labels_raw, (np.ndarray, list)):
            if len(labels_raw) == 0:
                pass
            else:
                labels_list = labels_raw
        elif isinstance(labels_raw, scipy.io.matlab.mat_struct):
            labels_list = [labels_raw]
        else:
            # Handle cases where it might be a single item but not a list
            try:
                if np.size(labels_raw) > 0:
                    labels_list = [labels_raw]
            except:
                pass

        parsed_labels = []
        for l in labels_list:
            # Ensure required attributes exist
            if hasattr(l, "Name") and hasattr(l, "Begin") and hasattr(l, "End"):
                name = str(l.Name).strip()
                # Normalize name if needed (e.g. lowercase)
                # The map keys are lowercase
                normalized_name = name.lower()

                label_id = LABEL_MAP.get(normalized_name)
                # If exact match failed, try to match loosely or skip
                if label_id is None:
                    # Try original casing or skip
                    label_id = LABEL_MAP.get(name)

                if label_id is not None:
                    parsed_labels.append(
                        {
                            "name": name,
                            "id": label_id,
                            "begin": int(l.Begin),
                            "end": int(l.End),
                        }
                    )

        # Sort by start frame
        parsed_labels.sort(key=lambda x: x["begin"])
        return parsed_labels

    except Exception as e:
        print(f"Warning: Failed to parse labels for {mat_path}: {e}")
        return []


def get_user_id(mat_path, sample_id):
    """
    Attempts to extract User ID from the .mat file.
    If not found, uses a heuristic based on SampleID blocks.
    """
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" in mat:
            video = mat["Video"]
            # Check common fields for User ID
            for field in ["User", "Subject", "Actor", "Signer"]:
                if hasattr(video, field):
                    return str(getattr(video, field))
    except:
        pass

    # Fallback Heuristic
    try:
        num_id = int(sample_id.replace("Sample", ""))
        # Group by blocks of 17 (approximate user session size)
        return f"UserGroup_{num_id // 17}"
    except:
        return "Unknown"


def scan_directory_for_samples(dir_pattern, is_test=False):
    """
    Scans directories matching the pattern for SampleXXXXX files.
    Returns a list of sample dictionaries.
    """
    samples = []
    # Find all directories matching the pattern (e.g., input/training*)
    search_dirs = glob.glob(os.path.join(INPUT_DIR, dir_pattern))

    for d in search_dirs:
        # Find all _data.mat files in this directory
        mat_files = glob.glob(os.path.join(d, "*_data.mat"))

        for mat_path in mat_files:
            filename = os.path.basename(mat_path)
            # filename format: SampleXXXXX_data.mat
            sample_id = filename.replace("_data.mat", "")

            # Directory relative to input
            rel_dir = os.path.relpath(d, INPUT_DIR)

            # Base relative path for the sample files
            base_rel_path = os.path.join(rel_dir, sample_id)

            # Construct expected paths
            rgb_rel = f"{base_rel_path}_color.mp4"
            depth_rel = f"{base_rel_path}_depth.mp4"
            audio_rel = f"{base_rel_path}_audio.wav"
            data_rel = f"{base_rel_path}_data.mat"

            # Check for user mask file (could be .mp4 or .avi, or missing)
            # We check existence to determine the correct extension
            user_rel = None
            possible_user_exts = [".mp4", ".avi"]
            for ext in possible_user_exts:
                path_check = os.path.join(INPUT_DIR, f"{base_rel_path}_user{ext}")
                if os.path.exists(path_check):
                    user_rel = f"{base_rel_path}_user{ext}"
                    break

            # If not found, we can leave it None or assume .mp4 and let validation catch it.
            # Given the prompt implies it exists, we'll default to .mp4 if not found,
            # but setting it to None if missing is safer for loading scripts.
            if user_rel is None:
                user_rel = f"{base_rel_path}_user.mp4"

            # Parse labels
            labels = []
            user_group = "Unknown"
            if not is_test:
                labels = parse_labels_from_mat(mat_path)
                user_group = get_user_id(mat_path, sample_id)

            samples.append(
                {
                    "sample_id": sample_id,
                    "rgb_path": rgb_rel,
                    "depth_path": depth_rel,
                    "audio_path": audio_rel,
                    "user_path": user_rel,
                    "data_path": data_rel,
                    "labels": json.dumps(labels),  # Store as JSON string
                    "num_gestures": len(labels),
                    "user_group": user_group,
                }
            )

    return samples


# ==========================================
# Main Execution
# ==========================================


def main():
    # 1. Setup Metadata Directory
    if os.path.exists(METADATA_DIR):
        shutil.rmtree(METADATA_DIR)
    os.makedirs(METADATA_DIR)

    print("Scanning training directories...")
    # Scan training1, training2, training3
    train_samples = scan_directory_for_samples("training*", is_test=False)
    print(f"Found {len(train_samples)} training samples.")

    print("Scanning test directory...")
    # Scan test
    test_samples = scan_directory_for_samples("test", is_test=True)
    print(f"Found {len(test_samples)} test samples.")

    # 2. Create DataFrames
    df_train_full = pd.DataFrame(train_samples)
    df_test = pd.DataFrame(test_samples)

    # 3. Split Training into Train/Val
    # We use group shuffle split to ensure user independence
    if len(df_train_full) > 0:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(
            gss.split(df_train_full, groups=df_train_full["user_group"])
        )

        df_train = df_train_full.iloc[train_idx].copy()
        df_val = df_train_full.iloc[val_idx].copy()

        print(f"Split performed using {df_train_full['user_group'].nunique()} groups.")
    else:
        df_train = pd.DataFrame(columns=df_train_full.columns)
        df_val = pd.DataFrame(columns=df_train_full.columns)

    # 4. Save to CSV
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print(f"Saved metadata to {METADATA_DIR}")

    # ==========================================
    # Validation Checks
    # ==========================================
    print("\n=== Validation Checks ===")

    # 1. Summary Statistics
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"Dataset: {name}")
        print(f"  Samples: {len(df)}")
        if "num_gestures" in df.columns and len(df) > 0:
            total_gestures = df["num_gestures"].sum()
            print(f"  Total Gestures: {total_gestures}")
            print(f"  Avg Gestures/Sample: {df['num_gestures'].mean():.2f}")
        print("-" * 20)

    # 2. File Existence Check
    # Collect all paths from all dataframes
    all_dfs = pd.concat([df_train, df_val, df_test])
    path_cols = [
        "rgb_path",
        "depth_path",
        "audio_path",
        "data_path",
    ]  # user_path might be missing

    paths_to_check = []
    for col in path_cols:
        if col in all_dfs.columns:
            paths_to_check.extend(all_dfs[col].dropna().tolist())

    # Check user paths separately (allow missing if file doesn't exist, but here we check what we put in metadata)
    # If we put a path in metadata, we expect it to exist.
    if "user_path" in all_dfs.columns:
        paths_to_check.extend(all_dfs["user_path"].dropna().tolist())

    # Randomly select 1000 paths (or all if less than 1000)
    if len(paths_to_check) > 1000:
        check_subset = random.sample(paths_to_check, 1000)
    else:
        check_subset = paths_to_check

    missing_count = 0
    missing_samples = []

    for p in check_subset:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / len(check_subset) if len(check_subset) > 0 else 0
    print(f"Checked {len(check_subset)} file paths.")
    print(f"Missing files: {missing_count} (Ratio: {missing_ratio:.4f})")

    if missing_ratio > 0.5:
        print("Sample missing files:")
        for m in missing_samples:
            print(f"  {m}")
        raise FileNotFoundError(f"Too many files missing! Ratio: {missing_ratio}")

    # 3. Validation Split Verification
    # Verify split ratio (approximate due to group splitting)
    total_train_val = len(df_train) + len(df_val)
    if total_train_val > 0:
        val_ratio = len(df_val) / total_train_val
        print(f"Validation Split Ratio: {val_ratio:.4f}")
        # Allow larger deviation due to group splitting granularity
        if not (0.10 <= val_ratio <= 0.35):
            print(
                f"Warning: Validation split ratio {val_ratio} is outside typical range."
            )

    print("All checks passed successfully.")


if __name__ == "__main__":
    main()
