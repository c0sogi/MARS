import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided libraries
from library import config
from library import utils
from library import data_loader
from library import model
from library import train
from library import inference


def load_ground_truth(metadata_path):
    """Loads ground truth label sequences for validation samples."""
    df = pd.read_csv(metadata_path)
    gt_dict = {}
    for _, row in df.iterrows():
        sid = row["sample_id"]
        labels_json = row["labels"]
        if isinstance(labels_json, str):
            try:
                gestures = json.loads(labels_json)
                gestures.sort(key=lambda x: x["begin"])
                gt_ids = [g["id"] for g in gestures]
                gt_dict[sid] = gt_ids
            except:
                gt_dict[sid] = []
        else:
            gt_dict[sid] = []
    return gt_dict


def perform_failure_analysis(model_path):
    print("Performing Failure Analysis on Validation Set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Validation Data
    # We use the dataset directly to get metadata and window indices
    val_ds = data_loader.GestureDataset(
        config.VAL_METADATA_PATH, "val", load_cached_data=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Load Model
    net = model.PG_HCKN().to(device)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Model not found at {model_path}")
        return

    net.eval()

    # 3. Prepare Reconstruction Buffers
    # Map sample_id -> {num_frames, start_idx}
    sample_info_map = {item["sample_id"]: item for item in val_ds.sample_map}

    seq_probs = {
        sid: np.zeros((item["num_frames"], config.NUM_CLASSES), dtype=np.float32)
        for sid, item in sample_info_map.items()
    }
    seq_counts = {
        sid: np.zeros((item["num_frames"], 1), dtype=np.float32)
        for sid, item in sample_info_map.items()
    }

    window_indices = val_ds.window_indices
    global_idx = 0

    # 4. Inference Loop
    with torch.no_grad():
        for features, _, _ in val_loader:
            features = features.to(device)
            outputs = net(features)
            # Use Stage 3 predictions
            probs = torch.nn.functional.softmax(outputs["stage3"], dim=2).cpu().numpy()

            batch_size = features.size(0)

            for i in range(batch_size):
                if global_idx >= len(window_indices):
                    break

                start_global, end_global, sid, needs_padding = window_indices[
                    global_idx
                ]
                window_prob = probs[i]

                # Get sample start index for relative offset calculation
                sample_start_global = sample_info_map[sid]["start_idx"]

                if needs_padding:
                    actual_len = sample_info_map[sid]["num_frames"]
                    valid_len = min(config.WINDOW_SIZE, actual_len)
                    window_prob = window_prob[:valid_len]

                    # Target slice in buffer (starts at 0 for padded small samples)
                    target_slice = slice(0, valid_len)
                else:
                    rel_start = start_global - sample_start_global
                    rel_end = end_global - sample_start_global
                    target_slice = slice(rel_start, rel_end)

                if sid in seq_probs:
                    # Safety check on shapes
                    dest_shape = seq_probs[sid][target_slice].shape
                    if dest_shape[0] == window_prob.shape[0]:
                        seq_probs[sid][target_slice] += window_prob
                        seq_counts[sid][target_slice] += 1.0

                global_idx += 1

    # 5. Compute Metrics & Correlations
    val_gt = load_ground_truth(config.VAL_METADATA_PATH)

    errors = []
    lengths = []
    num_gestures_list = []

    for sid, prob_sum in seq_probs.items():
        count = seq_counts[sid]
        count[count == 0] = 1.0
        avg_probs = prob_sum / count

        frame_preds = np.argmax(avg_probs, axis=1)
        pred_gestures = utils.run_length_encoding(
            frame_preds, min_duration=config.MIN_GESTURE_DURATION
        )

        target_gestures = val_gt.get(sid, [])

        dist = utils.levenshtein_distance(pred_gestures, target_gestures)

        errors.append(dist)
        lengths.append(sample_info_map[sid]["num_frames"])
        num_gestures_list.append(len(target_gestures))

    # Correlation Analysis
    if len(errors) > 1:
        corr_len, _ = pearsonr(errors, lengths)
        corr_ng, _ = pearsonr(errors, num_gestures_list)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_ng:.4f}")
    else:
        print("Not enough samples for correlation analysis.")


def main():
    # Set fixed seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print("=== Starting PG-HCKN Pipeline ===")

    # 1. Train
    # We use the default epochs (50) which is sufficient and fast for this dataset size.
    best_score = train.train_model(limit=None, epochs=config.NUM_EPOCHS)

    # Print Metric exactly as requested
    print(f"Final Validation Metric: {best_score}")

    # 2. Failure Analysis
    model_path = os.path.join(config.WORKING_DIR, "idea_34", "best_model.pth")
    perform_failure_analysis(model_path)

    # 3. Submission
    if best_score < 0.2251:
        print("Validation score is satisfactory. Generating submission...")
        inference.generate_submission(model_path=model_path)
    else:
        print(
            f"Validation score {best_score} is above threshold 0.2251. Submission skipped."
        )


if __name__ == "__main__":
    main()
