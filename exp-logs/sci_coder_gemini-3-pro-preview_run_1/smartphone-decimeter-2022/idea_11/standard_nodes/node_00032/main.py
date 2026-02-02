import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.train import train_model
from library.data import load_data
from library.inference import generate_submission
from library.utils import set_seed

# Constants
METADATA_DIR = "./metadata"
VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
MODEL_PATH = "./working/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_SCORE = 3.802240262877392


def calculate_competition_metric(df_results):
    """
    Calculates the mean of the 50th and 95th percentile distance errors.
    """
    # Group by phone (trip)
    trips = df_results.groupby("trip_id")
    trip_scores = []

    for _, group in trips:
        errors = group["error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        trip_scores.append((p50 + p95) / 2)

    if not trip_scores:
        return 0.0

    return np.mean(trip_scores)


def run_validation_and_analysis(trainer):
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load Validation Data
    val_dataset = load_data(
        VAL_META,
        split="train",  # Validation set has GT, so we treat it like train for loading
        load_cached_data=True,
    )

    if len(val_dataset) == 0:
        print("Validation dataset is empty.")
        return float("inf")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = trainer.model
    model.eval()

    all_errors = []
    all_trip_ids = []

    # For Failure Analysis
    feature_names = val_dataset.feature_cols
    feature_correlations = {name: [] for name in feature_names}
    error_magnitudes = []
    feature_values_accum = []

    print(f"Validating on {len(val_dataset)} trips...")

    with torch.no_grad():
        for i, (x, y, meta) in enumerate(val_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Forward pass
            output = model(x)

            # Calculate Euclidean distance error (meters)
            # Output and Y are (Batch, SeqLen, 2) -> (East, North)
            diff = output - y
            dist_error = torch.sqrt(torch.sum(diff**2, dim=2)).cpu().numpy().flatten()

            # Store for metric
            trip_id = f"{meta['drive_id'][0]}-{meta['phone_name'][0]}"
            all_errors.extend(dist_error)
            all_trip_ids.extend([trip_id] * len(dist_error))

            # Store for failure analysis (subsampling to avoid OOM on large sets)
            # We take every 10th sample for correlation analysis to be fast
            if len(dist_error) > 0:
                mask = np.arange(0, len(dist_error), 10)
                error_magnitudes.extend(dist_error[mask])

                # x is (Batch, SeqLen, Features)
                x_np = x.cpu().numpy()[0]  # Take first batch
                x_sub = x_np[mask, :]
                feature_values_accum.append(x_sub)

    # 1. Calculate Metric
    results_df = pd.DataFrame({"trip_id": all_trip_ids, "error": all_errors})

    final_metric = calculate_competition_metric(results_df)
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    if feature_values_accum:
        print("\n--- Failure Analysis ---")
        features_np = np.concatenate(feature_values_accum, axis=0)
        errors_np = np.array(error_magnitudes)

        correlations = []
        for idx, feat_name in enumerate(feature_names):
            # Handle potential constant features or NaNs
            feat_vals = features_np[:, idx]
            if np.std(feat_vals) == 0:
                corr = 0
            else:
                corr = np.corrcoef(feat_vals, errors_np)[0, 1]
                if np.isnan(corr):
                    corr = 0
            correlations.append((feat_name, corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 5 Features correlated with Error Magnitude:")
        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")

    return final_metric


def main():
    # 1. Setup
    set_seed(42)

    # 2. Train Model (Fast Baseline)
    # Limiting to 5 epochs and 10 drives to ensure execution within time limits
    print("--- Starting Model Training ---")
    trainer = train_model(
        epochs=5,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=3,
        max_drives=10,
        load_cached_data=True,
    )

    if trainer is None:
        print("Training failed or no data found.")
        return

    # 3. Validation & Analysis
    metric = run_validation_and_analysis(trainer)

    # 4. Submission
    if metric < THRESHOLD_SCORE:
        print(
            f"\nValidation score {metric:.4f} meets threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        try:
            generate_submission(load_cached_data=True)
        except Exception as e:
            print(f"Error generating submission: {e}")
    else:
        print(
            f"\nValidation score {metric:.4f} did not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
