import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import library modules
from library import config, utils, data_loader, model, train


def main():
    # 1. Setup
    utils.seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    # process_data returns a dictionary with keys corresponding to metadata splits
    data = data_loader.process_data(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    inc_train = data["inc_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    inc_val = data["inc_val"]

    X_test = data["X_test"]
    inc_test = data["inc_test"]

    scaling_stats = (data["ch_mins"], data["ch_maxs"])

    print(f"Train set shape: {X_train.shape}")
    print(f"Val set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 3. Create DataLoaders
    # Train Loader (with Augmentation)
    train_ds = data_loader.IcebergDataset(
        X_train,
        inc_train,
        y_train,
        transform=data_loader.get_transforms(augment=True),
        scaling_stats=scaling_stats,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Val Loader (No Augmentation)
    val_ds = data_loader.IcebergDataset(
        X_val,
        inc_val,
        y_val,
        transform=data_loader.get_transforms(augment=False),
        scaling_stats=scaling_stats,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Test Loader
    test_ds = data_loader.IcebergDataset(
        X_test,
        inc_test,
        labels=None,
        transform=data_loader.get_transforms(augment=False),
        scaling_stats=scaling_stats,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Train Model
    # We use fold_idx=0 for naming, but we are training on the fixed train split
    print("Starting training...")
    train.run_fold(0, train_loader, val_loader, device)

    # 5. Evaluation
    print("Loading best model for evaluation...")
    net = model.WBDIN().to(device)
    model_path = os.path.join(config.WORKING_DIR, "model_fold_0.pth")
    net.load_state_dict(torch.load(model_path))

    # Inference on Validation Set
    val_probs = train.predict(net, val_loader, device)

    # Calculate Metric
    # Ensure y_val is correct shape/type
    final_metric = log_loss(y_val, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate errors
    # val_probs is (N, 1) or (N,), y_val is (N,)
    preds_flat = val_probs.flatten()
    errors = np.abs(y_val - preds_flat)

    # Calculate simple image stats for correlation
    # X_val is (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    # We use the raw values (before scaling in dataset) for analysis as they have physical meaning (dB)
    # Note: X_val from process_data is unscaled.
    b1_mean = X_val[:, 0, :, :].mean(axis=(1, 2))
    b2_mean = X_val[:, 1, :, :].mean(axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_val,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission
    threshold = 0.16676861786296204
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is below threshold ({threshold}). Generating submission..."
        )

        test_probs = train.predict(net, test_loader, device)
        test_probs = test_probs.flatten()

        # Load test metadata to get IDs
        df_test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

        # Create submission DataFrame
        submission = pd.DataFrame({"id": df_test_meta["id"], "is_iceberg": test_probs})

        # Save
        os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)
        submission.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE}")
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
