import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import library modules
from library.config import Config
from library.train import train_model
from library.model import InkDetector
from library.utils import predict_tiled, rle_encode
from library.data import load_fragment, get_global_stats


def main():
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Training
    # Increased samples to 12000 (Cite solution_lesson_node_00020)
    print("\n--- Starting Training ---")
    best_score = train_model(
        load_cached_data=True, num_train_samples=Config.NUM_TRAIN_SAMPLES
    )

    # 3. Validation Metric Reporting
    # Strict requirement: Print the final validation metric in full precision.
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    print("\n--- Starting Failure Analysis ---")

    # Load best model
    model = InkDetector().to(device)
    checkpoint_path = Config.CHECKPOINT_DIR / "best_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded best model for analysis.")
    else:
        print("Error: Best model checkpoint not found.")
        return

    model.eval()

    # Load validation data
    val_metadata_path = Config.VAL_METADATA
    if not val_metadata_path.exists():
        print("Validation metadata not found.")
        return

    df_val = pd.read_csv(val_metadata_path)

    # Get global stats for normalization
    # We load the training metadata just to compute/retrieve the stats used during training
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA)
    mean, std = get_global_stats(
        df_train_meta, Config.WORKING_DIR, load_cached_data=True
    )

    # We will analyze the first validation fragment found
    if not df_val.empty:
        row = df_val.iloc[0]
        print(f"Analyzing validation fragment: {row['fragment_id']}")

        # Load data
        vol, mask, label = load_fragment(row, Config.WORKING_DIR, load_cached_data=True)

        if label is not None:
            # Inference
            pred_map = predict_tiled(
                model,
                vol,
                patch_size=Config.PATCH_SIZE,
                stride=Config.STRIDE,
                device=device,
                mean=mean,
                std=std,
            )

            # Calculate Error Magnitude
            # pred_map is probability (0-1), label is binary (0 or 1)
            error_map = np.abs(pred_map - label)

            # Calculate Input Features (collapsed to 2D)
            # 1. Mean Intensity across Z
            input_mean = np.mean(vol, axis=0)
            # 2. Std Dev across Z (Texture/Variation)
            input_std = np.std(vol, axis=0)

            # Mask out invalid pixels for correlation calculation
            valid_mask = mask > 0

            flat_error = error_map[valid_mask]
            flat_mean = input_mean[valid_mask]
            flat_std = input_std[valid_mask]

            # Calculate Correlations
            if len(flat_error) > 0:
                corr_mean = np.corrcoef(flat_error, flat_mean)[0, 1]
                corr_std = np.corrcoef(flat_error, flat_std)[0, 1]

                print(f"Correlation (Error vs Mean Intensity): {corr_mean:.6f}")
                print(f"Correlation (Error vs Z-Variance): {corr_std:.6f}")

                if abs(corr_mean) > 0.1:
                    print(
                        "-> Significant correlation with intensity detected. Model may be relying too much on brightness."
                    )
                if abs(corr_std) > 0.1:
                    print("-> Significant correlation with texture variance detected.")
            else:
                print("No valid pixels found for correlation analysis.")

    # 5. Submission Generation
    TARGET_THRESHOLD = 0.39266693592071533

    if best_score > TARGET_THRESHOLD:
        print("\n--- Generating Submission ---")

        # Load optimized threshold
        threshold_file = Config.WORKING_DIR / "threshold.txt"
        if threshold_file.exists():
            with open(threshold_file, "r") as f:
                opt_threshold = float(f.read().strip())
            print(f"Using optimized threshold: {opt_threshold}")
        else:
            opt_threshold = 0.5
            print("Warning: Threshold file not found, using default 0.5")

        # Load Test Metadata
        test_metadata_path = Config.TEST_METADATA
        if not test_metadata_path.exists():
            print("Test metadata not found.")
            return

        df_test = pd.read_csv(test_metadata_path)
        submission_data = []

        for _, row in df_test.iterrows():
            frag_id = row["fragment_id"]
            print(f"Processing test fragment: {frag_id}")

            # Load volume (mask is loaded if available, else generated as ones)
            vol, mask, _ = load_fragment(row, Config.WORKING_DIR, load_cached_data=True)

            # Inference
            pred_map = predict_tiled(
                model,
                vol,
                patch_size=Config.PATCH_SIZE,
                stride=Config.STRIDE,
                device=device,
                mean=mean,
                std=std,
            )

            # Apply Mask
            pred_map = pred_map * mask

            # Binarize
            binary_pred = (pred_map > opt_threshold).astype(np.uint8)

            # Encode
            rle_str = rle_encode(binary_pred)
            submission_data.append({"Id": frag_id, "Predicted": rle_str})

        # Save Submission
        submission_df = pd.DataFrame(submission_data)

        # Ensure directory exists
        sub_dir = Path("./submission")
        sub_dir.mkdir(exist_ok=True)

        output_path = sub_dir / "submission.csv"
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

        # Also save to root as per general convention/config backup
        submission_df.to_csv("submission.csv", index=False)

    else:
        print(
            f"\nValidation score ({best_score}) did not meet the threshold ({TARGET_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
