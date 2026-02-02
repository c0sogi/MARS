import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import DataLoader

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.inference import InferenceManager
from library.model import MultiStreamGRU
from library.data_loader import GestureDataset, collate_fn
from library.utils import set_seed, get_device
from library.metrics import levenshtein_distance, decode_sequence


def calculate_validation_metric(device):
    """
    Runs inference on validation set, computes Levenshtein error rate,
    and returns the metric along with a DataFrame for failure analysis.
    """
    # Load Best Model
    model = MultiStreamGRU().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")
    model.eval()

    # Load Validation Data
    val_dataset = GestureDataset(split="val", augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Load Metadata for Ground Truth
    val_df = pd.read_csv(Config.VAL_CSV)
    # Map sample_id to labels and num_frames for quick lookup
    # Note: val_dataset.sample_ids is ordered, we can index directly or map.
    # Mapping is safer.
    meta_map = {}
    for _, row in val_df.iterrows():
        # Cite debug_lesson_5: Sanitize Metadata Inputs
        raw_val = row["labels"]
        if (
            pd.isna(raw_val)
            or str(raw_val).strip() == ""
            or str(raw_val).lower() == "nan"
        ):
            gt_list = []
        else:
            gt_list = [int(x) for x in str(raw_val).split(",")]
        meta_map[row["sample_id"]] = {"gt": gt_list, "num_frames": row["num_frames"]}

    total_distance = 0
    total_gestures = 0
    analysis_data = []

    global_idx = 0

    with torch.no_grad():
        for batch_idx, (skeleton, audio, labels, lengths, indices) in enumerate(
            val_loader
        ):
            skeleton = skeleton.to(device)
            audio = audio.to(device)

            # Inference
            logits = model(skeleton, audio)
            predictions = torch.argmax(logits, dim=2).cpu().numpy()

            batch_size = skeleton.shape[0]
            for i in range(batch_size):
                sample_id = val_dataset.sample_ids[global_idx]
                global_idx += 1

                # Get Ground Truth
                meta = meta_map.get(sample_id, {"gt": [], "num_frames": 0})
                gt_list = meta["gt"]

                # Process Prediction
                valid_len = lengths[i]
                raw_pred = predictions[i, :valid_len]

                # Decode (using shared metrics logic with median filter)
                pred_list = decode_sequence(raw_pred, apply_median_filter=True)

                # Compute Metric
                dist = levenshtein_distance(pred_list, gt_list)

                total_distance += dist
                total_gestures += len(gt_list)

                # Collect data for failure analysis
                analysis_data.append(
                    {
                        "sample_id": sample_id,
                        "error": dist,
                        "num_frames": meta["num_frames"],
                        "num_gt": len(gt_list),
                    }
                )

    final_metric = total_distance / total_gestures if total_gestures > 0 else 0.0
    return final_metric, pd.DataFrame(analysis_data)


def main():
    set_seed(Config.SEED)
    device = get_device()

    # 1. Train
    print("Initializing Trainer...")
    trainer = Trainer(limit=None)
    print("Starting Training...")
    trainer.fit(epochs=Config.NUM_EPOCHS)

    # 2. Validate & Compute Metric
    print("Calculating validation metric...")
    metric, error_df = calculate_validation_metric(device)
    print(f"Final Validation Metric: {metric}")

    # 3. Failure Analysis
    print("Performing failure analysis...")
    if not error_df.empty:
        # Correlation with sequence length
        corr_len = error_df["error"].corr(error_df["num_frames"])
        # Correlation with complexity (number of gestures)
        corr_count = error_df["error"].corr(error_df["num_gt"])

        print(f"Correlation (Error vs NumFrames): {corr_len:.10f}")
        print(f"Correlation (Error vs NumGestures): {corr_count:.10f}")

    # 4. Submission
    threshold = 0.1369
    if metric < threshold:
        print(f"Metric {metric} < {threshold}. Generating submission...")
        inference_manager = InferenceManager()
        inference_manager.predict_all()
    else:
        print(f"Metric {metric} >= {threshold}. Submission generation skipped.")


if __name__ == "__main__":
    main()
