import sys
import os
import json
import pandas as pd
import numpy as np
import torch
import nltk

# Import provided library modules
from library.config import Config
from library.utils import set_seeds, compute_levenshtein_ratio, rle_encode
from library.data_loader import get_dataloaders
from library.model import RLK_RN
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    # 1. Setup and Configuration
    set_seeds(Config.SEED)

    # Adjust configuration for a fast baseline execution
    # The dataset is small (232 samples), so we can use the full set,
    # but we limit epochs to ensure it finishes well within the time limit.
    Config.NUM_EPOCHS = 20
    Config.DEBUG = False  # Use full dataset for best performance

    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    print("Initializing RLK-RN Model...")
    model = RLK_RN()

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # 5. Task-Specific Validation (Levenshtein Metric)
    print("Performing Validation on Hold-out Set...")

    # Initialize Inference Engine with the best model saved during training
    inference_engine = InferenceEngine(model_path=Config.BEST_MODEL_PATH)

    # Load validation metadata to iterate over samples
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    predictions = []
    ground_truths = []

    # Storage for failure analysis
    sample_metrics = []

    # Iterate over validation samples
    # We use the inference engine's helper to extract features and predict
    for idx, row in val_df.iterrows():
        sample_id = row["sample_id"]

        # Extract features using the logic from InferenceEngine
        # Note: We access the protected method _extract_features_from_row as it contains the exact logic needed
        features = inference_engine._extract_features_from_row(row)

        if features is None:
            # Fallback for empty/corrupt samples
            pred_seq = []
            duration_frames = 0
        else:
            # Predict probabilities
            probs = inference_engine.predict_sequence(features)
            duration_frames = features.shape[0]

            # Decode: Argmax -> RLE -> Filter Background
            frame_preds = np.argmax(probs, axis=1)
            pred_seq = rle_encode(frame_preds, background_label=Config.BACKGROUND_LABEL)

        predictions.append(pred_seq)

        # Parse Ground Truth
        labels_json = json.loads(row["labels"])
        # Ensure sorted by time
        labels_json.sort(key=lambda x: x["begin"])
        gt_seq = [l["id"] for l in labels_json]
        ground_truths.append(gt_seq)

        # Calculate individual error for analysis
        dist = nltk.edit_distance(pred_seq, gt_seq)

        sample_metrics.append(
            {
                "sample_id": sample_id,
                "error": dist,
                "gt_length": len(gt_seq),
                "duration_frames": duration_frames,
                "pred_length": len(pred_seq),
            }
        )

    # Compute Final Metric: Sum(Distances) / Sum(GT Lengths)
    total_distance = sum(m["error"] for m in sample_metrics)
    total_gt_gestures = sum(m["gt_length"] for m in sample_metrics)

    if total_gt_gestures > 0:
        final_metric = total_distance / total_gt_gestures
    else:
        final_metric = float("inf")

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    analysis_df = pd.DataFrame(sample_metrics)

    if not analysis_df.empty:
        # Correlation between Error and Duration
        corr_dur = analysis_df["duration_frames"].corr(analysis_df["error"])
        # Correlation between Error and Sequence Complexity (Number of Gestures)
        corr_cplx = analysis_df["gt_length"].corr(analysis_df["error"])

        print(f"Correlation (Error vs. Duration): {corr_dur:.4f}")
        print(f"Correlation (Error vs. Num Gestures): {corr_cplx:.4f}")

        print("\nTop 3 Worst Performing Samples:")
        worst_samples = analysis_df.sort_values(by="error", ascending=False).head(3)
        print(
            worst_samples[["sample_id", "error", "gt_length", "pred_length"]].to_string(
                index=False
            )
        )
    else:
        print("No validation samples to analyze.")

    # 7. Submission
    # Threshold defined in instructions
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric:.4f}) is below threshold ({THRESHOLD}). Generating Submission..."
        )
        inference_engine.generate_submission(output_path=Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation Metric ({final_metric:.4f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
