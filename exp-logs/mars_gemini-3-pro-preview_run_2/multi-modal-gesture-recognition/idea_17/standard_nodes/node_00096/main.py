import sys
import os
import glob
import shutil
import scipy.io
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein,
    smooth_predictions,
    decode_predictions,
)
from library.data import load_data, GestureDataset, collate_fn
from library.model import SG_CRCN
from library.train import Trainer


# -------------------------------------------------------------------------
# Metadata Generation Logic
# -------------------------------------------------------------------------
def parse_mat_file(mat_path):
    """Parses the .mat file to extract gesture labels and frame count."""
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "Video" not in mat:
            return [], 0
        video = mat["Video"]
        num_frames = getattr(video, "NumFrames", 0)
        labels_raw = getattr(video, "Labels", [])
        gesture_list = []

        def process_label_obj(obj):
            try:
                name = obj.Name
                if name in Config.GESTURE_MAP:
                    return Config.GESTURE_MAP[name]
            except AttributeError:
                pass
            return None

        if isinstance(labels_raw, np.ndarray):
            if labels_raw.size > 0:
                if labels_raw.ndim == 0:
                    g_id = process_label_obj(labels_raw.item())
                    if g_id:
                        gesture_list.append(g_id)
                else:
                    for l in labels_raw:
                        g_id = process_label_obj(l)
                        if g_id:
                            gesture_list.append(g_id)
        else:
            g_id = process_label_obj(labels_raw)
            if g_id:
                gesture_list.append(g_id)

        return gesture_list, num_frames
    except Exception as e:
        # print(f"Warning: Failed to parse {mat_path}: {e}")
        return [], 0


def get_samples(folders, is_train=True):
    samples = []
    for folder in folders:
        folder_path = os.path.join(Config.INPUT_DIR, folder)
        if not os.path.exists(folder_path):
            continue
        mat_files = glob.glob(os.path.join(folder_path, "*_data.mat"))
        for mat_file in mat_files:
            filename = os.path.basename(mat_file)
            sample_id = filename.replace("_data.mat", "")

            # Construct paths
            rgb_path = os.path.join(folder, f"{sample_id}_color.mp4")
            depth_path = os.path.join(folder, f"{sample_id}_depth.mp4")
            audio_path = os.path.join(folder, f"{sample_id}_audio.wav")
            user_path = os.path.join(folder, f"{sample_id}_user.mp4")
            data_path = os.path.join(folder, f"{sample_id}_data.mat")

            labels = []
            num_frames = 0

            # Parse MAT
            full_data_path = os.path.join(Config.INPUT_DIR, data_path)
            if os.path.exists(full_data_path):
                parsed_labels, parsed_frames = parse_mat_file(full_data_path)
                num_frames = parsed_frames
                if is_train:
                    labels = parsed_labels

            # Only add if we have valid frame count
            if num_frames > 0:
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


def generate_metadata():
    print("Regenerating metadata...")
    if os.path.exists(Config.METADATA_DIR):
        shutil.rmtree(Config.METADATA_DIR)
    os.makedirs(Config.METADATA_DIR)

    # Train/Val
    train_folders = ["training1", "training2", "training3"]
    train_samples = get_samples(train_folders, is_train=True)
    df_train_full = pd.DataFrame(train_samples)

    if not df_train_full.empty:
        train_df, val_df = train_test_split(
            df_train_full, test_size=0.2, random_state=Config.SEED, shuffle=True
        )
        train_df.to_csv(os.path.join(Config.METADATA_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(Config.METADATA_DIR, "val.csv"), index=False)
        print(f"Generated train.csv ({len(train_df)}) and val.csv ({len(val_df)})")
    else:
        print("Warning: No training samples found!")

    # Test
    test_folders = ["test"]
    test_samples = get_samples(test_folders, is_train=False)
    df_test = pd.DataFrame(test_samples)
    df_test.to_csv(os.path.join(Config.METADATA_DIR, "test.csv"), index=False)
    print(f"Generated test.csv ({len(df_test)})")


def evaluate_and_analyze(model, val_loader, device):
    """
    Evaluates the model on the validation set, computes the final metric,
    and performs failure analysis (correlation of error with sequence length).
    """
    model.eval()

    all_errors = []
    all_lengths = []

    total_dist = 0
    total_len = 0

    print("Evaluating on validation set...")

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]  # Keep on CPU for decoding
            lengths = batch["lengths"]  # Keep on CPU

            # Forward pass
            outputs = model(features, mask)

            # Use Stage 3 Class Probabilities for final prediction
            s3_probs = outputs["stage3_cls"]  # (B, T, C)
            frame_preds = torch.argmax(s3_probs, dim=2).cpu().numpy()
            frame_targets = labels.numpy()

            for i in range(len(features)):
                length = lengths[i].item()

                # Extract valid frames
                raw_pred = frame_preds[i, :length]
                raw_target = frame_targets[i, :length]

                # Post-processing
                smoothed_pred = smooth_predictions(
                    raw_pred, window_size=Config.MEDIAN_WINDOW
                )
                decoded_pred = decode_predictions(smoothed_pred, background_class=0)
                decoded_target = decode_predictions(raw_target, background_class=0)

                # Compute Levenshtein Distance
                dist = compute_levenshtein(decoded_pred, decoded_target)
                tgt_len = len(decoded_target)

                total_dist += dist
                total_len += tgt_len

                # Store for analysis
                all_errors.append(dist)
                all_lengths.append(length)

    # Compute Final Metric (Normalized Levenshtein)
    final_metric = total_dist / total_len if total_len > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Sequence Length
    if len(all_errors) > 1:
        # Check for constant input to avoid warnings
        if np.std(all_errors) > 0 and np.std(all_lengths) > 0:
            corr, _ = pearsonr(all_errors, all_lengths)
            print(
                f"Correlation between Error (Levenshtein) and Sequence Length (Frames): {corr:.4f}"
            )
        else:
            print("Correlation undefined (constant error or length).")

    return final_metric


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")

    # Load Test Data
    test_data = load_data(mode="test", load_cached_data=False)
    test_dataset = GestureDataset(test_data, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["ids"]
            lengths = batch["lengths"]

            outputs = model(features, mask)
            s3_probs = outputs["stage3_cls"]
            frame_preds = torch.argmax(s3_probs, dim=2).cpu().numpy()

            for i in range(len(features)):
                length = lengths[i].item()
                sample_id = ids[i]

                raw_pred = frame_preds[i, :length]
                smoothed_pred = smooth_predictions(
                    raw_pred, window_size=Config.MEDIAN_WINDOW
                )
                decoded_pred = decode_predictions(smoothed_pred, background_class=0)

                # Format: SessionID,label1,label2,...
                # If empty, just SessionID,
                pred_str = ",".join(map(str, decoded_pred))
                results.append(f"{sample_id},{pred_str}")

    # Save to file
    submission_path = Config.SUBMISSION_FILE
    # Ensure directory exists (Config does this, but double check)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    with open(submission_path, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Submission saved to {submission_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Regenerate Metadata (Cite debug_lesson_14)
    generate_metadata()

    # 2. Training
    # Initialize Trainer
    trainer = Trainer(device=device)

    # Load Datasets
    print("Loading datasets...")
    trainer.load_datasets()

    # Train Model
    # We rely on the Trainer's internal loop and early stopping
    print("Starting training...")
    trainer.fit()

    # 3. Evaluation & Failure Analysis
    # Load Best Model
    print(f"Loading best model from {Config.MODEL_CHECKPOINT}...")
    best_model = SG_CRCN().to(device)
    best_model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    # Run Evaluation
    final_metric = evaluate_and_analyze(best_model, trainer.val_loader, device)

    # 4. Submission
    # Threshold check
    THRESHOLD = 0.08548168249660787

    if final_metric < THRESHOLD:
        generate_submission(best_model, device)
    else:
        print(
            f"Validation metric {final_metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
