import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.data_preprocessing import get_data
from library.dataset import GNSSSequenceDataset, collate_padded_sequences
from library.model import TransUNet1D
from library.trainer import Trainer
from library.inference import generate_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(model, val_loader, feature_names, device):
    print("\n--- Failure Analysis ---")
    model.eval()

    all_errors = []
    all_features = []

    with torch.no_grad():
        for features, targets, mask, _ in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            # Forward pass
            outputs = model(features, mask)
            outputs = outputs.permute(0, 2, 1)  # (B, L, 2)

            # Move to CPU
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            features_np = features.permute(0, 2, 1).cpu().numpy()  # (B, L, C)
            mask_np = mask.cpu().numpy().astype(bool)

            batch_size = features.shape[0]

            for i in range(batch_size):
                valid_len = np.sum(mask_np[i])
                if valid_len == 0:
                    continue

                # Extract valid data
                valid_pred = outputs_np[i, :valid_len, :]
                valid_target = targets_np[i, :valid_len, :]
                valid_feats = features_np[i, :valid_len, :]

                # Calculate Euclidean error (meters)
                # targets are dLat_meters, dLon_meters
                diff = valid_pred - valid_target
                errors = np.sqrt(np.sum(diff**2, axis=1))

                all_errors.append(errors)
                all_features.append(valid_feats)

    if not all_errors:
        print("No validation data found for analysis.")
        return

    # Concatenate all time steps
    flat_errors = np.concatenate(all_errors)
    flat_features = np.concatenate(all_features)

    print(f"Analyzing {len(flat_errors)} validation samples.")

    # Calculate correlations
    correlations = {}
    for idx, feat_name in enumerate(feature_names):
        feat_values = flat_features[:, idx]
        # Handle constant features (std=0) to avoid NaN correlation
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(feat_values, flat_errors)
        correlations[feat_name] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Input Features and Error Magnitude:")
    for name, corr in sorted_corrs:
        print(f"  {name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for fast baseline
    Config.NUM_EPOCHS = 5  # Reduced from 50 for speed

    print(f"Running on device: {Config.DEVICE}")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        print("Metadata not found. Please ensure metadata generation script has run.")
        return

    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # 3. Load and Process Data
    # We use load_cached_data=True to leverage existing parquet files if available
    print("Loading Training Data...")
    df_train = get_data(train_meta, load_cached_data=True)

    print("Loading Validation Data...")
    df_val = get_data(val_meta, load_cached_data=True)

    # 4. Create Datasets
    # Scaler is fitted on training data and applied to validation data
    train_dataset = GNSSSequenceDataset(
        df_train,
        feature_cols=Config.INPUT_FEATURES,
        target_cols=Config.TARGET_COLS,
        mode="train",
    )

    val_dataset = GNSSSequenceDataset(
        df_val,
        feature_cols=Config.INPUT_FEATURES,
        target_cols=Config.TARGET_COLS,
        mode="val",
        scaler=train_dataset.scaler,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_padded_sequences,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_padded_sequences,
        num_workers=2,
        pin_memory=True,
    )

    # 6. Initialize Model
    model = TransUNet1D()

    # 7. Train
    trainer = Trainer(model, train_loader, val_loader, device=Config.DEVICE)
    best_model = trainer.fit()

    # 8. Final Validation Metric
    print("Computing final validation metric...")
    _, final_metric = trainer.validate()
    print(f"Final Validation Metric: {final_metric}")

    # 9. Failure Analysis
    perform_failure_analysis(
        best_model, val_loader, Config.INPUT_FEATURES, Config.DEVICE
    )

    # 10. Submission
    THRESHOLD = 3.802240262877392
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(
            model_path=trainer.checkpoint_path, load_cached_data=True, batch_size=1
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
