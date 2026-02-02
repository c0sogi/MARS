import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import provided library modules
from library import config, utils, data_loader, model, train


def run_pipeline():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # 2. Training Phase
    # Run the 5-fold CV training. This saves models to config.WORK_DIR
    # We use debug_size=None to train on the full dataset for maximum performance.
    print("Starting Training Phase...")
    train.run_training(debug_size=None)

    # 3. Validation Phase
    print("\nStarting Validation Phase...")

    # Load validation metadata to identify hold-out samples
    val_meta_path = config.VAL_META
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    df_val_meta = pd.read_csv(val_meta_path)
    val_ids_set = set(df_val_meta["id"].values)

    # Load cached preprocessed data
    # Training has just completed, so cache exists.
    train_data, test_data, stats = data_loader.process_and_cache_data(
        load_cached_data=True
    )

    # Filter training data to get only the validation set
    all_ids = train_data["ids"]
    mask = np.isin(all_ids, list(val_ids_set))

    val_images = train_data["images"][mask]
    val_angles = train_data["angles"][mask]
    val_labels = train_data["labels"][mask]
    val_ids = train_data["ids"][mask]

    print(f"Validation set size: {len(val_ids)}")

    # Create Validation Dataset and Loader
    val_dataset = data_loader.IcebergDataset(
        images=val_images,
        angles=val_angles,
        stats=stats,
        labels=val_labels,
        ids=val_ids,
        transform=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference on Validation Set
    fold_preds = []

    for fold in range(config.NUM_FOLDS):
        model_path = os.path.join(config.WORK_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        # Initialize and load model
        net = model.InputAnchoredWideBodyNet().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint)
        net.eval()

        preds = []
        with torch.no_grad():
            for images, angles, _, _ in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = net(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds.extend(probs)

        fold_preds.append(np.array(preds))

    if not fold_preds:
        raise RuntimeError("No models available for validation.")

    # Average predictions (Ensemble)
    avg_preds = np.mean(fold_preds, axis=0)

    # Calculate Metric (Log Loss)
    # Clip to prevent log(0)
    avg_preds_clipped = np.clip(avg_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_labels, avg_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nFailure Analysis...")
    errors = np.abs(val_labels - avg_preds)

    # Compute simple image stats for correlation analysis
    # val_images is (N, 75, 75, 3). Band 0: HH, Band 1: HV
    b1_mean = np.mean(val_images[:, :, :, 0], axis=(1, 2))
    b2_mean = np.mean(val_images[:, :, :, 1], axis=(1, 2))

    fa_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": val_angles,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
        }
    )

    correlations = fa_df.corr()["error"].drop("error")
    print("Correlations with Error Magnitude:")
    print(correlations)

    # 5. Submission
    THRESHOLD = 0.14772333549413377

    if final_metric < THRESHOLD:
        print("\nMetric meets threshold. Generating submission...")

        # Prepare Test Loader
        test_dataset = data_loader.IcebergDataset(
            images=test_data["images"],
            angles=test_data["angles"],
            stats=stats,
            labels=None,
            ids=test_data["ids"],
            transform=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        test_fold_preds = []

        # Inference on Test Set
        for fold in range(config.NUM_FOLDS):
            model_path = os.path.join(config.WORK_DIR, f"model_fold_{fold}.pth")
            if not os.path.exists(model_path):
                continue

            net = model.InputAnchoredWideBodyNet().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            net.load_state_dict(checkpoint)
            net.eval()

            preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = net(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    preds.extend(probs)

            test_fold_preds.append(np.array(preds))

        # Ensemble Average
        avg_test_preds = np.mean(test_fold_preds, axis=0)

        # Save Submission
        sub_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": avg_test_preds})

        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
