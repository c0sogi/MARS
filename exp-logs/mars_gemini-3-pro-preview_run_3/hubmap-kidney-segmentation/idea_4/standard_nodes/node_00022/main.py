import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import HuBMAPDataset
from library.model import StainNet
from library.loss import DeepSupervisionLoss
from library.train import train_model
from library.inference import predict_test_set


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config defaults to ensure execution completes within time limits
    Config.EPOCHS = 3
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Process a subset of tiles for speed

    # Initialize directories and seeds
    Config.setup()

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print("--- Starting Training Pipeline ---")
    # train_model handles dataset creation, training loop, and saving best_model.pth
    train_model(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = StainNet()
    model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Error: Best model file not found. Training may have failed.")
        return

    model.eval()

    # Initialize Validation Dataset
    val_dataset = HuBMAPDataset(mode="validation", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load metadata for failure analysis mapping
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Metrics tracking
    running_dice = 0.0
    dataset_size = 0
    analysis_data = []

    # Global index to track tiles in the non-shuffled loader
    global_idx = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)

            # Handle Deep Supervision output (list of tensors)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Compute predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > Config.THRESHOLD).float()

            # Compute Dice for each sample in the batch
            # Flatten: (B, H*W)
            preds_flat = preds.view(batch_size, -1)
            masks_flat = masks.view(batch_size, -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)
            smooth = 1e-7

            # Dice per sample
            dices = (2.0 * intersection + smooth) / (union + smooth)

            # Update aggregate metric
            running_dice += dices.sum().item()
            dataset_size += batch_size

            # Collect data for failure analysis
            batch_dices_np = dices.cpu().numpy()

            for i in range(batch_size):
                if global_idx + i >= len(val_dataset):
                    break

                # Get tile metadata to link back to patient/image info
                tile_meta = val_dataset.get_tile_metadata(global_idx + i)
                image_id = tile_meta["id"]

                # Retrieve image-level metadata
                # (Assuming id matches; val_meta_df has one row per image)
                meta_row = val_meta_df[val_meta_df["id"] == image_id]

                if not meta_row.empty:
                    row = meta_row.iloc[0]
                    error = 1.0 - batch_dices_np[i]

                    analysis_data.append(
                        {
                            "error": error,
                            "age": row.get("age", np.nan),
                            "weight": row.get("weight_kilograms", np.nan),
                            "bmi": row.get("bmi_kg/m^2", np.nan),
                            "percent_cortex": row.get("percent_cortex", np.nan),
                            "percent_medulla": row.get("percent_medulla", np.nan),
                        }
                    )

            global_idx += batch_size

    # Calculate Final Metric
    final_metric = running_dice / dataset_size if dataset_size > 0 else 0.0
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Perform Failure Analysis
    print("\nFailure Analysis (Correlation of Error with Features):")
    if analysis_data:
        df_analysis = pd.DataFrame(analysis_data)
        # Drop rows with missing values for correlation
        df_analysis = df_analysis.dropna()

        if not df_analysis.empty:
            # Compute correlation of 'error' with other columns
            correlations = df_analysis.corr()["error"].drop("error")
            print(correlations.sort_values(ascending=False))
        else:
            print("Insufficient data for correlation analysis after dropping NaNs.")
    else:
        print("No analysis data collected.")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9132

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.4f}) exceeds threshold ({THRESHOLD})."
        )
        print("Generating submission for test set...")
        predict_test_set(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_metric:.4f}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
