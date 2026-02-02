import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
from scipy.stats import pearsonr

# Add library path to sys.path just in case, though structure suggests it's in root
sys.path.append(".")

# Import provided library modules
from library.config import Config
from library.train import train
from library.model import WSDN_ABS
from library.inference import predict_fragment, rle_encode
from library.dataset import load_fragment_data, seed_everything
from library.utils import f05_score


def main():
    # 1. Configuration Override for Fast Baseline
    # Limit runtime to ensure completion within 50 mins
    Config.NUM_EPOCHS = 5
    Config.SAMPLES_PER_EPOCH = 3000
    Config.BATCH_SIZE = 8

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("--- Starting Pipeline ---")
    Config.print_config()

    # 2. Training
    print("\n--- Training Model ---")
    train()

    # 3. Validation & Metric Calculation
    print("\n--- Performing Validation ---")
    device = torch.device(Config.DEVICE)
    model = WSDN_ABS(
        in_channels=Config.Z_DIM,
        model_channels=Config.MODEL_CHANNELS,
        dilation_rates=Config.DILATION_RATES,
    )

    model_path = Config.WORKING_DIR / "best_model.pth"
    if not model_path.exists():
        print("Error: Best model not found after training.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_meta_path = Config.METADATA_DIR / "val.csv"
    if not val_meta_path.exists():
        print("Validation metadata not found.")
        return

    df_val = pd.read_csv(val_meta_path)

    all_preds = []
    all_labels = []
    all_intensities = []  # For failure analysis

    # Process validation fragments
    for _, row in df_val.iterrows():
        fid = str(row["fragment_id"])
        print(f"Predicting validation fragment {fid}...")

        # Get Prediction (Probability Map)
        prob_map = predict_fragment(model, fid, "val", device)
        if prob_map is None:
            continue

        # Get Ground Truth and Volume for Analysis
        vol, mask, label = load_fragment_data(
            fid,
            "val",
            row["surface_volume_path"],
            row["mask_path"],
            row["inklabels_path"],
            load_cached_data=True,
        )

        # Apply mask to valid area
        if mask is not None:
            # Resize mask if necessary (though usually they match)
            if mask.shape != prob_map.shape:
                mask = cv2.resize(
                    mask,
                    (prob_map.shape[1], prob_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Zero out invalid areas in prediction
            prob_map = prob_map * mask

            # Flatten valid pixels for metric calculation
            # We use the mask to select valid pixels to reduce memory and focus on valid region
            valid_indices = mask > 0

            flat_preds = prob_map[valid_indices]
            flat_labels = label[valid_indices]

            # For failure analysis: compute mean intensity
            # Volume is (Z, H, W). Mean across Z.
            mean_intensity = np.mean(vol, axis=0).astype(np.float32)
            flat_intensity = mean_intensity[valid_indices]

            all_preds.append(flat_preds)
            all_labels.append(flat_labels)
            all_intensities.append(flat_intensity)

    if not all_preds:
        print("No validation predictions generated.")
        return

    # Concatenate all fragments
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_intensities = np.concatenate(all_intensities)

    # Threshold Optimization
    print("Optimizing threshold...")
    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
    )
    best_score = -1.0
    best_th = 0.5

    # Subsample for threshold search to speed up if array is huge
    if len(all_preds) > 5_000_000:
        indices = np.random.choice(len(all_preds), 5_000_000, replace=False)
        search_preds = all_preds[indices]
        search_labels = all_labels[indices]
    else:
        search_preds = all_preds
        search_labels = all_labels

    for th in thresholds:
        score = f05_score(search_preds, search_labels, threshold=th)
        if score > best_score:
            best_score = score
            best_th = th

    # Calculate exact score on full set with best threshold
    final_metric = f05_score(all_preds, all_labels, threshold=best_th)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error
    # Binarize predictions for error analysis or use continuous error?
    # Prompt asks for "error magnitude". Continuous error |prob - label| is usually more informative.
    errors = np.abs(all_preds - all_labels)

    # Calculate Correlation
    # Subsample for correlation to be safe on memory/time
    if len(errors) > 1_000_000:
        indices = np.random.choice(len(errors), 1_000_000, replace=False)
        sample_errors = errors[indices]
        sample_intensities = all_intensities[indices]
    else:
        sample_errors = errors
        sample_intensities = all_intensities

    # Handle constant input cases
    if np.std(sample_errors) == 0 or np.std(sample_intensities) == 0:
        corr = 0.0
    else:
        corr, _ = pearsonr(sample_errors, sample_intensities)

    print(f"Correlation between Error Magnitude and Input Intensity: {corr:.4f}")

    # 5. Submission
    target_metric = 0.4064630960392697
    if final_metric > target_metric:
        print(
            f"\nMetric ({final_metric:.4f}) > Target ({target_metric:.4f}). Generating submission..."
        )

        test_meta_path = Config.METADATA_DIR / "test.csv"
        if not test_meta_path.exists():
            print("Test metadata not found.")
            return

        df_test = pd.read_csv(test_meta_path)
        submission_data = []

        for _, row in df_test.iterrows():
            fid = str(row["fragment_id"])
            print(f"Processing test fragment {fid}...")

            # Predict
            prob_map = predict_fragment(model, fid, "test", device)

            if prob_map is None:
                submission_data.append({"Id": fid, "Predicted": ""})
                continue

            # Load valid mask to clean up prediction
            _, mask, _ = load_fragment_data(
                fid,
                "test",
                row["surface_volume_path"],
                row["mask_path"],
                None,
                load_cached_data=True,
            )

            # Binarize using best threshold
            binary_map = (prob_map > best_th).astype(np.uint8)

            # Mask out invalid areas
            if mask is not None:
                if mask.shape != binary_map.shape:
                    mask = cv2.resize(
                        mask,
                        (binary_map.shape[1], binary_map.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                binary_map = binary_map * mask

            # RLE Encode
            rle = rle_encode(binary_map)
            submission_data.append({"Id": fid, "Predicted": rle})

        # Save Submission
        submission_df = pd.DataFrame(submission_data)
        submission_path = "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nMetric ({final_metric:.4f}) <= Target ({target_metric:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
