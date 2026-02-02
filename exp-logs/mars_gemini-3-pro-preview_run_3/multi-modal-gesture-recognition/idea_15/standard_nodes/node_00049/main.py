import sys
import os
import torch
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from collections import defaultdict

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    rle_encode,
    levenshtein_distance,
)
from library.data_loader import get_dataloaders, GestureDataset
from library.model import RSKARN
from library.trainer import Trainer
from library.inference import generate_predictions


def aggregate_predictions(model, loader, device):
    """
    Runs inference on a loader and aggregates sliding window probabilities
    to reconstruct full sequence predictions.
    """
    model.eval()
    dataset = loader.dataset
    raw_predictions = defaultdict(list)

    with torch.no_grad():
        for data, _, indices in loader:
            data = data.to(device)

            # Forward pass - use Stage 3 probabilities
            outputs = model(data)
            probs = outputs["probs_3"].cpu().numpy()

            for i, idx in enumerate(indices):
                idx_int = idx.item()
                meta = dataset.window_metadata[idx_int]
                sample_id = meta["sample_id"]
                start_frame = meta["start_frame"]
                raw_predictions[sample_id].append((start_frame, probs[i]))

    final_preds = {}

    for sample_id, windows in raw_predictions.items():
        # Determine max length
        max_len = 0
        for start, p in windows:
            end = start + p.shape[0]
            if end > max_len:
                max_len = end

        # Accumulate
        full_probs = np.zeros((max_len, Config.NUM_CLASSES), dtype=np.float32)
        counts = np.zeros((max_len, 1), dtype=np.float32)

        for start, p in windows:
            length = p.shape[0]
            full_probs[start : start + length] += p
            counts[start : start + length] += 1.0

        counts[counts == 0] = 1.0
        avg_probs = full_probs / counts

        # Decode
        frame_preds = np.argmax(avg_probs, axis=1)
        gesture_list = rle_encode(frame_preds)
        final_preds[sample_id] = gesture_list

    return final_preds


def load_ground_truth(csv_path):
    """
    Parses the metadata CSV to get ground truth gesture sequences.
    Returns: Dict[sample_id, List[int]]
    """
    df = pd.read_csv(csv_path)
    ground_truth = {}

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        labels_json = row["labels"]
        labels_list = json.loads(labels_json)

        # Sort by begin frame just in case
        labels_list.sort(key=lambda x: x["begin"])

        # Extract IDs
        gt_ids = [l["id"] for l in labels_list]
        ground_truth[sample_id] = gt_ids

    return ground_truth, df


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Optimize for fast baseline execution
    Config.NUM_EPOCHS = 50
    Config.PATIENCE = 7

    set_seed()
    Config.setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # ==========================================
    # 2. Data Loading & Training
    # ==========================================
    print("\n--- Phase 1: Training ---")
    train_loader, val_loader = get_dataloaders(batch_size=Config.BATCH_SIZE)

    model = RSKARN()
    trainer = Trainer(model, train_loader, val_loader)

    # Train the model
    trainer.fit(epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\n--- Phase 2: Validation & Metric ---")

    # Load best model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Generate predictions on validation set (reconstructing sequences)
    print("Aggregating validation predictions...")
    val_preds = aggregate_predictions(model, val_loader, device)

    # Load Ground Truth
    val_gt, val_df = load_ground_truth(Config.VAL_CSV)

    # Align predictions and ground truth
    pred_sequences = []
    target_sequences = []
    sample_ids = []

    # Ensure we only compare samples present in both (should be all val samples)
    for sid in val_gt.keys():
        if sid in val_preds:
            pred_sequences.append(val_preds[sid])
            target_sequences.append(val_gt[sid])
            sample_ids.append(sid)
        else:
            # If a sample produced no windows (e.g. too short), predict empty
            pred_sequences.append([])
            target_sequences.append(val_gt[sid])
            sample_ids.append(sid)

    # Compute Metric
    metric = compute_normalized_levenshtein(pred_sequences, target_sequences)
    print(f"Final Validation Metric: {metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Phase 3: Failure Analysis ---")

    errors = []
    num_gestures = []
    # Approximate duration from metadata if available, else skip
    # We can get duration from the validation dataframe 'num_gestures' or file paths
    # Let's use 'num_gestures' from the dataframe as a proxy for complexity

    val_df.set_index("sample_id", inplace=True)

    for i, sid in enumerate(sample_ids):
        # Calculate individual Levenshtein distance
        dist = levenshtein_distance(pred_sequences[i], target_sequences[i])
        errors.append(dist)

        # Get metadata features
        if sid in val_df.index:
            n_gest = val_df.loc[sid, "num_gestures"]
            num_gestures.append(n_gest)
        else:
            num_gestures.append(0)

    # Correlation Analysis
    if len(errors) > 1:
        corr_gest, _ = pearsonr(errors, num_gestures)
        print(f"Correlation (Error vs. Num Gestures): {corr_gest:.4f}")

        avg_error = np.mean(errors)
        print(f"Average Levenshtein Distance per Sequence: {avg_error:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n--- Phase 4: Submission ---")

    # Threshold check
    if metric < 0.2251:
        print(f"Metric {metric} meets threshold (< 0.2251). Generating submission...")
        generate_predictions(checkpoint_path=best_model_path)
    else:
        print(f"Metric {metric} did not meet threshold (< 0.2251). Submission skipped.")


if __name__ == "__main__":
    main()
