import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything
from library.data_loader import process_and_cache_data, get_dataloaders, IcebergDataset
from library.model import MS_IDPH_CNN
from library.trainer import Trainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Data Loading
    print("Loading data...")
    data = process_and_cache_data(load_cached_data=True)

    # 3. Cross-Validation Loop
    oof_preds = []
    oof_targets = []

    # Metadata for failure analysis
    meta_angles = []
    meta_b1_mean = []
    meta_b2_mean = []

    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold_idx in range(Config.NUM_FOLDS):
        # Get DataLoaders
        train_loader, val_loader = get_dataloaders(data, fold_idx)

        # Initialize Model
        model = MS_IDPH_CNN().to(Config.DEVICE)

        # Initialize Trainer
        trainer = Trainer(model, Config.DEVICE, train_loader, val_loader, fold_idx)

        # Train
        trainer.fit()

        # --- Inference for OOF and Analysis ---
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Load best model
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for imgs, angs, labels in val_loader:
                # Move to device
                imgs = imgs.to(Config.DEVICE)
                angs = angs.to(Config.DEVICE)

                # Predict
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy()

                # Store predictions and targets
                fold_preds.extend(probs.flatten())
                fold_targets.extend(labels.numpy().flatten())

                # Extract metadata for failure analysis
                # imgs shape: (B, 4, 75, 75). Channel 0: HH, Channel 1: HV
                imgs_np = imgs.cpu().numpy()
                b1_batch = imgs_np[:, 0, :, :]
                b2_batch = imgs_np[:, 1, :, :]

                # Compute means per image
                meta_b1_mean.extend(np.mean(b1_batch, axis=(1, 2)))
                meta_b2_mean.extend(np.mean(b2_batch, axis=(1, 2)))
                meta_angles.extend(angs.cpu().numpy().flatten())

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

    # 4. Global Validation Metric
    y_true = np.array(oof_targets)
    y_pred = np.array(oof_preds)

    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-sample error (Log Loss contribution)
    # Loss = - (y * log(p) + (1-y) * log(1-p))
    sample_losses = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    analysis_df = pd.DataFrame(
        {
            "loss": sample_losses,
            "inc_angle": meta_angles,
            "b1_mean": meta_b1_mean,
            "b2_mean": meta_b2_mean,
        }
    )

    print("Correlation between Error (Log Loss) and Features:")
    features = ["inc_angle", "b1_mean", "b2_mean"]
    for feat in features:
        # Check for NaNs (though imputation should have handled angles)
        valid_mask = ~np.isnan(analysis_df[feat])
        if valid_mask.sum() > 1:
            corr, _ = pearsonr(
                analysis_df.loc[valid_mask, feat], analysis_df.loc[valid_mask, "loss"]
            )
            print(f"  {feat}: {corr:.6f}")
        else:
            print(f"  {feat}: Not enough valid data for correlation.")

    # 6. Submission
    THRESHOLD = 0.17174082291273365
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(data)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


def generate_submission(data):
    """
    Generates submission file by ensembling predictions from all 5 folds.
    """
    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # Create Test Dataset and Loader
    # Note: Pass y=None
    test_dataset = IcebergDataset(X_test, None, angle_test, transform=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    ensemble_probs = np.zeros((len(X_test), 1))

    print("Predicting on Test Set...")
    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint for fold {fold} not found.")
            continue

        # Load Model
        model = MS_IDPH_CNN().to(Config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs = imgs.to(Config.DEVICE)
                angs = angs.to(Config.DEVICE)

                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)

        # Stack batches
        fold_probs = np.vstack(fold_probs)
        ensemble_probs += fold_probs

    # Average
    avg_probs = ensemble_probs / Config.NUM_FOLDS

    # Save
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_probs.flatten()})
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
