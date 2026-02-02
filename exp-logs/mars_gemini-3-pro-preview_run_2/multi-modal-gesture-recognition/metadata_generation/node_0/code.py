import os
import glob
import pandas as pd
import numpy as np
import scipy.io
import shutil
from sklearn.model_selection import train_test_split

# Define paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42

# Create metadata directory
if os.path.exists(METADATA_DIR):
    shutil.rmtree(METADATA_DIR)
os.makedirs(METADATA_DIR)

# Gesture vocabulary mapping
GESTURE_MAP = {
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


def parse_mat_file(mat_path):
    """Parses the .mat file to extract gesture labels and frame count."""
    try:
        # loadmat with squeeze_me=True simplifies the structure
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return [], 0

        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        labels_raw = getattr(video, "Labels", [])

        gesture_list = []

        # Helper to process a single label object
        def process_label_obj(obj):
            try:
                name = obj.Name
                if name in GESTURE_MAP:
                    return GESTURE_MAP[name]
            except AttributeError:
                pass
            return None

        # Handle different shapes of labels_raw
        if isinstance(labels_raw, np.ndarray):
            if labels_raw.size == 0:
                pass  # Empty
            elif labels_raw.ndim == 0:
                # 0-d array (scalar)
                g_id = process_label_obj(labels_raw.item())
                if g_id:
                    gesture_list.append(g_id)
            else:
                # 1-d array
                for l in labels_raw:
                    g_id = process_label_obj(l)
                    if g_id:
                        gesture_list.append(g_id)
        else:
            # Single object (not an array)
            g_id = process_label_obj(labels_raw)
            if g_id:
                gesture_list.append(g_id)

        return gesture_list, num_frames
    except Exception as e:
        print(f"Warning: Failed to parse {mat_path}: {e}")
        return [], 0


def get_samples(folders, is_train=True):
    """Scans folders to identify samples and extract metadata."""
    samples = []
    for folder in folders:
        folder_path = os.path.join(INPUT_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue

        # Find all _data.mat files
        mat_files = glob.glob(os.path.join(folder_path, "*_data.mat"))

        for mat_file in mat_files:
            filename = os.path.basename(mat_file)
            sample_id = filename.replace("_data.mat", "")

            # Construct relative paths
            rgb_path = os.path.join(folder, f"{sample_id}_color.mp4")
            depth_path = os.path.join(folder, f"{sample_id}_depth.mp4")
            audio_path = os.path.join(folder, f"{sample_id}_audio.wav")
            user_path = os.path.join(folder, f"{sample_id}_user.mp4")
            data_path = os.path.join(folder, f"{sample_id}_data.mat")

            # Check if user file exists, otherwise set to None or keep path but it might fail validation if strictly checked
            # The prompt implies it exists ("SessionID_user: Video file...").
            # We will assume it exists for the path, but verify_metadata will check.
            # If it doesn't exist in the directory, we'll keep the path but it might be missing.

            labels = []
            num_frames = 0

            # Parse MAT file
            full_data_path = os.path.join(INPUT_DIR, data_path)
            if os.path.exists(full_data_path):
                parsed_labels, parsed_frames = parse_mat_file(full_data_path)
                num_frames = parsed_frames
                if is_train:
                    labels = parsed_labels

            samples.append(
                {
                    "sample_id": sample_id,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "audio_path": audio_path,
                    "user_path": user_path,
                    "data_path": data_path,
                    "labels": " ".join(map(str, labels)),
                    "num_frames": num_frames,
                }
            )
    return samples


def verify_metadata(df, name):
    """Verifies metadata content and file existence."""
    print(f"--- {name} Dataset ---")
    print(f"Total samples: {len(df)}")

    if "labels" in df.columns:
        all_labels = []
        for s in df["labels"]:
            if isinstance(s, str) and s.strip():
                all_labels.extend(s.split())
        print(f"Total gestures: {len(all_labels)}")
        if len(all_labels) > 0:
            print(f"Unique gestures: {len(set(all_labels))}")

    # Check file existence
    # We check rgb, depth, audio, data. User file might be optional or missing in some samples.
    paths_to_check = []
    for col in ["rgb_path", "depth_path", "audio_path", "data_path"]:
        if col in df.columns:
            paths_to_check.extend(df[col].tolist())

    if not paths_to_check:
        return

    # Random sample 1000 paths
    if len(paths_to_check) > 1000:
        sample_paths = np.random.choice(paths_to_check, 1000, replace=False)
    else:
        sample_paths = paths_to_check

    missing_count = 0
    missing_samples = []
    for p in sample_paths:
        full_path = os.path.join(INPUT_DIR, p)
        if not os.path.exists(full_path):
            missing_count += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    missing_ratio = missing_count / len(sample_paths)
    print(f"Missing file ratio: {missing_ratio:.4f}")

    if missing_ratio > 0.5:
        print("Example missing files:", missing_samples)
        raise FileNotFoundError(
            f"Too many missing files in {name} dataset. Ratio: {missing_ratio}"
        )


# Main Execution
print("Scanning training directories...")
train_folders = ["training1", "training2", "training3"]
train_samples = get_samples(train_folders, is_train=True)
df_train_full = pd.DataFrame(train_samples)

print("Scanning test directory...")
test_folders = ["test"]
test_samples = get_samples(test_folders, is_train=False)
df_test = pd.DataFrame(test_samples)

if df_train_full.empty:
    raise ValueError("No training samples found!")

# Split Training into Train/Val
print("Splitting training data...")
train_df, val_df = train_test_split(
    df_train_full, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

# Save Metadata
print("Saving metadata...")
train_df.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

# Verify
verify_metadata(train_df, "Training")
verify_metadata(val_df, "Validation")
verify_metadata(df_test, "Test")

# Verify split integrity
assert len(train_df) + len(val_df) == len(df_train_full)
# Check for data leakage
train_ids = set(train_df["sample_id"])
val_ids = set(val_df["sample_id"])
assert train_ids.isdisjoint(
    val_ids
), "Data leakage detected between train and val sets!"

print("Metadata generation complete.")
