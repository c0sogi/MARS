import os
import torch
import numpy as np
import pandas as pd
import nltk
from scipy.stats import pearsonr

# Import provided library modules
from library import (
    config,
    data_utils,
    features,
    dataset,
    model,
    loss,
    trainer,
    inference,
)


def calculate_levenshtein_error(predicted_seq, target_seq):
    """
    Calculates Levenshtein distance and the length of the target sequence.
    """
    # Handle edge cases where sequences might be empty
    if len(predicted_seq) == 0 and len(target_seq) == 0:
        return 0.0, 0

    dist = nltk.edit_distance(predicted_seq, target_seq)
    length = len(target_seq)
    return dist, length


def main():
    # ==========================================
    # 1. Configuration & Overrides
    # ==========================================
    # Optimized training duration
    config.NUM_EPOCHS = 25
    print(f"Running optimized training with {config.NUM_EPOCHS} epochs.")

    # Ensure reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets and statistics...")
    # Load raw data structures for analysis and manual validation loop
    train_data, val_data, test_data, stats = features.load_data_and_stats(
        load_cached_data=True
    )

    # Get DataLoaders for the training loop
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=True
    )

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Starting model training...")
    trained_model = trainer.train_model(train_loader, val_loader)

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    print("Performing full validation evaluation...")
    device = torch.device(config.DEVICE)
    trained_model.eval()

    total_distance = 0
    total_length = 0

    # Container for failure analysis
    analysis_records = []

    val_ids = val_data["sample_ids"]
    val_skeletons = val_data["skeleton"]
    val_audios = val_data["audio"]
    val_labels = val_data["labels"]

    # Iterate over validation set
    for i, sample_id in enumerate(val_ids):
        skel = val_skeletons[i]
        aud = val_audios[i]
        labels_raw = val_labels[i]

        # Extract Ground Truth Sequence
        gt_seq = [l["id"] for l in labels_raw]

        # Predict
        probs = inference.predict_sequence(trained_model, skel, aud, stats, device)
        pred_seq = inference.decode_predictions(probs)

        # Compute Metric
        dist, length = calculate_levenshtein_error(pred_seq, gt_seq)

        total_distance += dist
        total_length += length

        # Collect features for failure analysis
        seq_len_frames = skel.shape[0]
        num_gestures = len(gt_seq)
        # Simple audio feature: mean of MFCCs (approximate energy/activity level)
        audio_mean = np.mean(aud) if aud.size > 0 else 0.0

        # Calculate sample-wise error rate (safe division)
        sample_error = dist / length if length > 0 else (1.0 if dist > 0 else 0.0)

        analysis_records.append(
            {
                "sample_id": sample_id,
                "error_rate": sample_error,
                "seq_len": seq_len_frames,
                "num_gestures": num_gestures,
                "audio_mean": audio_mean,
            }
        )

    # Compute Final Metric
    # Metric = Total Levenshtein Distance / Total Number of Gestures
    final_metric = total_distance / total_length if total_length > 0 else 0.0

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(analysis_records)

    if len(df_analysis) > 1:
        # Correlation: Error Rate vs Sequence Length
        corr_len, _ = pearsonr(df_analysis["error_rate"], df_analysis["seq_len"])
        print(f"Correlation (Error vs Seq Len): {corr_len:.4f}")

        # Correlation: Error Rate vs Number of Gestures
        corr_num, _ = pearsonr(df_analysis["error_rate"], df_analysis["num_gestures"])
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")

        # Correlation: Error Rate vs Audio Mean
        corr_aud, _ = pearsonr(df_analysis["error_rate"], df_analysis["audio_mean"])
        print(f"Correlation (Error vs Audio Mean): {corr_aud:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = 0.2251
    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        inference.generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric} is not below threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
