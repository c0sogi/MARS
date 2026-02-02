import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, WGS84Utils
from library.dataset import load_data, GNSSDataset, gnss_collate_fn
from library.model import ResUNet1D
from library.train import run_training
from library.inference import generate_submission


def evaluate_validation():
    """
    Performs inference on the validation set, calculates the competition metric,
    and runs failure analysis.
    """
    print("\n--- Starting Validation Evaluation ---")

    # Load validation data
    # We use load_cached_data=True to reuse the parquet file generated during training
    val_df = load_data(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE, load_cached_data=True
    )

    if val_df.empty:
        raise ValueError("Validation data is empty.")

    # Prepare Dataset and DataLoader
    val_dataset = GNSSDataset(val_df)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True,
    )

    # Load Model
    device = torch.device(Config.DEVICE)
    model = ResUNet1D().to(device)

    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"
        )

    checkpoint = torch.load(Config.MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_errors = []
    all_trip_ids = []
    feature_values = {feat: [] for feat in Config.INPUT_FEATURES}

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)  # (B, L, C)
            targets = batch["targets"].to(device)  # (B, L, 2) [East, North]
            mask = batch["mask"].to(device)  # (B, L)
            trip_ids = batch["trip_ids"]

            # Permute for model input
            features_in = features.permute(0, 2, 1)  # (B, C, L)

            # Forward pass
            outputs = model(features_in)  # (B, 2, L)
            outputs = outputs.permute(0, 2, 1)  # (B, L, 2)

            # Calculate errors per sequence
            for i in range(len(trip_ids)):
                valid_len = mask[i].sum().item()

                # Slice valid data
                pred = outputs[i, :valid_len, :].cpu().numpy()  # (L, 2)
                target = targets[i, :valid_len, :].cpu().numpy()  # (L, 2)
                feat = features[i, :valid_len, :].cpu().numpy()  # (L, C)

                # Calculate Euclidean distance error in meters
                # pred/target columns: 0=East, 1=North
                diff = pred - target
                errors = np.sqrt(np.sum(diff**2, axis=1))

                all_errors.extend(errors)
                all_trip_ids.extend([trip_ids[i]] * valid_len)

                # Store features for failure analysis
                for f_idx, f_name in enumerate(Config.INPUT_FEATURES):
                    feature_values[f_name].extend(feat[:, f_idx])

    # Create Analysis DataFrame
    eval_df = pd.DataFrame({"trip_id": all_trip_ids, "error": all_errors})

    # Add features to df
    for f_name, values in feature_values.items():
        eval_df[f_name] = values

    # --- Calculate Metric ---
    # Metric: Mean of (50th + 95th percentile errors) averaged over phones (trips)
    trip_scores = []
    for trip_id, group in eval_df.groupby("trip_id"):
        errs = group["error"].values
        p50 = np.percentile(errs, 50)
        p95 = np.percentile(errs, 95)
        score = (p50 + p95) / 2
        trip_scores.append(score)

    final_metric = np.mean(trip_scores)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis (Correlation with Error) ---")
    correlations = {}
    for feat in Config.INPUT_FEATURES:
        corr = eval_df[feat].corr(eval_df["error"])
        correlations[feat] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"{feat}: {corr:.4f}")

    return final_metric


if __name__ == "__main__":
    # Purge stale artifacts to ensure clean execution with new logic
    # Cite debug_lesson_3: Purge Stale Artifacts to Prevent Cross-Experiment Contamination
    # Cite debug_lesson_8: Verify Code Activation via Log Observability
    for cache_file in [Config.TRAIN_CACHE, Config.VAL_CACHE, Config.TEST_CACHE]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Deleted stale cache: {cache_file}")

    if os.path.exists(Config.MODEL_CHECKPOINT):
        os.remove(Config.MODEL_CHECKPOINT)
        print(f"Deleted stale checkpoint: {Config.MODEL_CHECKPOINT}")

    # 1. Configure for Fast Baseline
    Config.EPOCHS = 5  # Reduce epochs for speed

    # 2. Run Training
    print("=== Starting Training Pipeline ===")
    run_training(load_cached_data=True)

    # 3. Validate and Analyze
    print("=== Starting Validation Pipeline ===")
    val_metric = evaluate_validation()

    # 4. Generate Submission if Metric Condition Met
    THRESHOLD = 3.802240262877392
    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({val_metric}) >= Threshold ({THRESHOLD}). Skipping submission generation."
        )
