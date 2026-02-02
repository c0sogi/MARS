import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.train import run_training
from library.model import TSPCResUNet
from library.dataset import DenoisingDataset
from library.utils import set_seed, calculate_rmse, apply_tta, tiled_inference
import library.train as train_module  # To access InferenceWrapper if needed

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline
    Config.EPOCHS = 10
    Config.PATCHES_PER_IMAGE = 50  # Reduce sampling density for speed
    Config.EARLY_STOPPING_PATIENCE = 3

    # Ensure reproducible runs
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("Starting Training Phase...")
    # run_training handles data loading, model init, training loop, and saving best model
    run_training()

    # -------------------------------------------------------------------------
    # 3. Load Best Model
    # -------------------------------------------------------------------------
    print("\nLoading best model for analysis...")
    model = TSPCResUNet().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Model checkpoint not found at {Config.BEST_MODEL_PATH}")
        return

    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Wrap model for inference (to get only the final stage output)
    infer_model = train_module.InferenceWrapper(model)
    infer_model.eval()

    # -------------------------------------------------------------------------
    # 4. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing Validation and Failure Analysis...")

    val_dataset = DenoisingDataset(
        metadata_path=Config.VAL_METADATA,
        root_dir=Config.INPUT_DIR,
        augment=False,
        train_mode=False,
        load_cached_data=True,
    )

    val_errors = []
    feat_areas = []
    feat_intensities = []

    # Iterate over validation set
    for i in range(len(val_dataset)):
        noisy, clean, img_id = val_dataset[i]
        noisy = noisy.to(device)

        # Inference
        with torch.no_grad():
            # Using tiled_inference directly for validation speed (no TTA)
            pred_noise = tiled_inference(
                infer_model,
                noisy,
                patch_size=Config.PATCH_SIZE,
                overlap_ratio=Config.OVERLAP_RATIO,
                batch_size=Config.BATCH_SIZE,
                device=device,
            )

        # Reconstruct Clean Image
        noisy_cpu = noisy.cpu().squeeze(0)
        pred_clean = noisy_cpu - pred_noise
        pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

        clean_cpu = clean.squeeze(0)

        # Calculate Metric
        rmse = calculate_rmse(pred_clean, clean_cpu)
        val_errors.append(rmse)

        # Features for Failure Analysis
        h, w = noisy_cpu.shape
        feat_areas.append(h * w)
        feat_intensities.append(noisy_cpu.mean().item())

    final_metric = np.mean(val_errors)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    if len(val_errors) > 1:
        corr_area, _ = pearsonr(val_errors, feat_areas)
        corr_intensity, _ = pearsonr(val_errors, feat_intensities)
        print(f"Correlation (Error vs Image Area): {corr_area:.4f}")
        print(f"Correlation (Error vs Mean Intensity): {corr_intensity:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.0076658159

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = DenoisingDataset(
            metadata_path=Config.TEST_METADATA,
            root_dir=Config.INPUT_DIR,
            augment=False,
            train_mode=False,
            load_cached_data=True,
        )

        submission_rows = []

        for i in range(len(test_dataset)):
            noisy, _, img_id = test_dataset[i]
            noisy = noisy.to(device)

            # Use TTA for best possible test predictions
            with torch.no_grad():
                pred_noise = apply_tta(
                    infer_model,
                    noisy,
                    patch_size=Config.PATCH_SIZE,
                    overlap_ratio=Config.OVERLAP_RATIO,
                    device=device,
                )

            # Reconstruct
            noisy_cpu = noisy.cpu().squeeze(0)
            pred_clean = noisy_cpu - pred_noise
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            # Convert to numpy
            pred_clean_np = pred_clean.numpy()  # (H, W)
            h, w = pred_clean_np.shape

            # Vectorized ID generation
            # Rows are 1..H, Cols are 1..W
            rows, cols = np.indices((h, w))
            rows = rows + 1
            cols = cols + 1

            # Flatten arrays
            flat_vals = pred_clean_np.flatten()
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()

            # Create IDs: "{img_id}_{row}_{col}"
            # Using list comprehension for string formatting is usually fast enough for this scale
            # but let's try to be efficient with pandas

            ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

            df_chunk = pd.DataFrame({"id": ids, "value": flat_vals})
            submission_rows.append(df_chunk)

        # Concatenate all chunks
        full_submission = pd.concat(submission_rows, ignore_index=True)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        full_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
