import os
import glob
import pandas as pd
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
os.makedirs(METADATA_DIR, exist_ok=True)

# Label Mapping
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


def get_relative_path(path):
    """Returns path relative to INPUT_DIR."""
    return os.path.relpath(path, INPUT_DIR)


def parse_mat_file(mat_path):
    """
    Parses the .mat file to extract the sequence of gesture labels and frame count.
    Returns: (list of label_ids, num_frames)
    """
    try:
        # Load mat file, squeeze_me=True simplifies the structure (arrays of 1 element become scalars)
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "Video" not in mat:
            return [], 0

        video_struct = mat["Video"]
        num_frames = getattr(video_struct, "NumFrames", 0)

        # Extract Labels
        # Labels might be missing, empty, a single object, or an array of objects
        labels_indices = []
        if hasattr(video_struct, "Labels"):
            labels_data = video_struct.Labels

            # Helper to process a single label object
            def process_label_obj(obj):
                if hasattr(obj, "Name"):
                    name = obj.Name
                    # Clean string if necessary (sometimes might be array of chars)
                    if not isinstance(name, str):
                        # Try to convert if it's something else, though squeeze_me usually handles it
                        pass
                    if name in LABEL_MAP:
                        return LABEL_MAP[name]
                return None

            if isinstance(labels_data, np.ndarray):
                if labels_data.size == 0:
                    pass  # Empty
                elif labels_data.size == 1:
                    # Single element array
                    lid = process_label_obj(labels_data.item())
                    if lid:
                        labels_indices.append(lid)
                else:
                    # Array of labels
                    for l in labels_data:
                        lid = process_label_obj(l)
                        if lid:
                            labels_indices.append(lid)
            elif hasattr(labels_data, "Name"):
                # Single struct (not array)
                lid = process_label_obj(labels_data)
                if lid:
                    labels_indices.append(lid)

        return labels_indices, num_frames

    except Exception as e:
        print(f"Warning: Failed to parse {mat_path}: {e}")
        return [], 0


def scan_directories(dir_names, is_test=False):
    """
    Scans specified subdirectories in INPUT_DIR for Sample data.
    """
    samples = []

    for d in dir_names:
        full_dir_path = os.path.join(INPUT_DIR, d)
        if not os.path.exists(full_dir_path):
            continue

        # Find all .mat files matching the pattern
        mat_files = glob.glob(os.path.join(full_dir_path, "Sample*_data.mat"))

        for mat_path in mat_files:
            # Extract ID from filename (SampleXXXXX_data.mat)
            filename = os.path.basename(mat_path)
            sample_id = filename.replace("_data.mat", "")

            dir_path = os.path.dirname(mat_path)

            # Construct expected paths
            audio_path = os.path.join(dir_path, f"{sample_id}_audio.wav")
            color_path = os.path.join(dir_path, f"{sample_id}_color.mp4")
            depth_path = os.path.join(dir_path, f"{sample_id}_depth.mp4")
            user_path = os.path.join(dir_path, f"{sample_id}_user.mp4")

            # Parse metadata
            labels, num_frames = parse_mat_file(mat_path)

            # Build record
            record = {
                "sample_id": sample_id,
                "data_path": get_relative_path(mat_path),
                "color_path": (
                    get_relative_path(color_path)
                    if os.path.exists(color_path)
                    else None
                ),
                "depth_path": (
                    get_relative_path(depth_path)
                    if os.path.exists(depth_path)
                    else None
                ),
                "audio_path": (
                    get_relative_path(audio_path)
                    if os.path.exists(audio_path)
                    else None
                ),
                "user_path": (
                    get_relative_path(user_path) if os.path.exists(user_path) else None
                ),
                "num_frames": num_frames,
                "labels": ",".join(map(str, labels)),
                "dataset_type": "sequence",
            }

            # Filter out broken samples (must have at least color video)
            if record["color_path"] is None:
                continue

            samples.append(record)

    return pd.DataFrame(samples)


def scan_valid_directories(dir_prefix="valid"):
    """Scans validXX directories containing K_*.avi files and CSVs."""
    samples = []
    valid_dirs = glob.glob(os.path.join(INPUT_DIR, f"{dir_prefix}*"))

    for full_dir_path in valid_dirs:
        # Look for CSV
        csv_files = glob.glob(os.path.join(full_dir_path, "*_train.csv"))
        if not csv_files:
            continue

        csv_path = csv_files[0]
        try:
            # Read CSV (assuming no header based on snippet)
            # Format: ID_STR, LABEL
            df_csv = pd.read_csv(csv_path, header=None, names=["id_str", "label"])

            for _, row in df_csv.iterrows():
                id_str = str(row["id_str"])
                label = row["label"]

                # Extract ID number (e.g., valid01_5 -> 5)
                try:
                    # Assuming format validXX_Y
                    file_id = id_str.split("_")[-1]
                    filename = f"K_{file_id}.avi"
                    video_path = os.path.join(full_dir_path, filename)

                    if os.path.exists(video_path):
                        samples.append(
                            {
                                "sample_id": id_str,
                                "data_path": None,  # No .mat file
                                "color_path": get_relative_path(video_path),
                                "depth_path": None,
                                "audio_path": None,
                                "user_path": None,
                                "num_frames": 0,  # Unknown without opening
                                "labels": str(label),
                                "dataset_type": "segmented",
                            }
                        )
                except Exception as e:
                    print(f"Error parsing row {row} in {csv_path}: {e}")

        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    return pd.DataFrame(samples)


def stratified_split_sequences(df, test_size=0.2):
    """Performs iterative stratified split on sequence data."""
    if df.empty:
        return df, pd.DataFrame()

    # 1. Create binary label matrix (n_samples x 20)
    n_classes = 20
    y = np.zeros((len(df), n_classes))

    df = df.reset_index(drop=True)

    for idx, row in df.iterrows():
        labels_str = str(row["labels"])
        if not labels_str:
            continue
        try:
            l_list = [int(x) for x in labels_str.split(",") if x.strip()]
            for l in l_list:
                if 1 <= l <= 20:
                    y[idx, l - 1] = 1
        except:
            pass

    X = np.arange(len(df)).reshape(-1, 1)

    # 2. Split
    # iterative_train_test_split expects X(n_samples, n_features), y(n_samples, n_labels)
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X, y, test_size=test_size
    )

    train_indices = X_train.flatten()
    val_indices = X_val.flatten()

    return df.iloc[train_indices], df.iloc[val_indices]


def main():
    print("Starting metadata generation...")

    # 1. Load Sequence Training Data (SampleXXXXX)
    train_dirs = ["training1", "training2", "training3"]
    print(f"Scanning training directories: {train_dirs}")
    seq_train_df = scan_directories(train_dirs)
    print(f"Found {len(seq_train_df)} sequence training samples.")

    # 2. Load Provided Validation Data (validXX)
    print("Scanning provided validation directories (valid01-20)...")
    provided_val_df = scan_valid_directories("valid")
    print(f"Found {len(provided_val_df)} provided validation samples.")

    # 3. Load Test Data
    test_dirs = ["test"]
    print(f"Scanning test directories: {test_dirs}")
    test_df = scan_directories(test_dirs, is_test=True)
    print(f"Found {len(test_df)} test samples.")

    # 4. Perform Stratified Split on Sequence Training Data
    if len(seq_train_df) > 0:
        print("Performing iterative stratified split on sequence training data...")
        train_split, val_split = stratified_split_sequences(seq_train_df, test_size=0.2)
    else:
        train_split = pd.DataFrame(columns=seq_train_df.columns)
        val_split = pd.DataFrame(columns=seq_train_df.columns)

    # 5. Save Metadata
    # Combine stratified split validation with provided validation
    full_val_df = pd.concat([val_split, provided_val_df], ignore_index=True)

    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    val_provided_path = os.path.join(METADATA_DIR, "val_provided.csv")

    train_split.to_csv(train_csv_path, index=False)
    full_val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    provided_val_df.to_csv(val_provided_path, index=False)

    print("Metadata files saved.")
    print(f"Train: {len(train_split)}")
    print(
        f"Val (Split + Provided): {len(full_val_df)} (Split: {len(val_split)}, Provided: {len(provided_val_df)})"
    )
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    main()
