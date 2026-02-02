import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import provided library components
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data_loader import get_loaders, process_data
from library.model_arch import DMWBN
from library.engine import Trainer


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Initializing pipeline on device: {device}")

    # Containers for Out-Of-Fold (OOF) data
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_img_means = []
    oof_img_stds = []

    # ==========================================
    # 2. Cross-Validation Training Loop
    # ==========================================
    for fold in range(Config.N_FOLDS):
        print(f"\n" + "=" * 30)
        print(f"Fold {fold}/{Config.N_FOLDS - 1}")
        print("=" * 30)

        # Load Data
        train_loader, val_loader, _ = get_loaders(fold_idx=fold, load_cached_data=True)

        # Initialize Model
        model = DMWBN().to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=False,
        )

        # Trainer
        trainer = Trainer(model, device, optimizer, scheduler)

        # Train
        save_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
        trainer.fit(
            train_loader,
            val_loader,
            epochs=Config.EPOCHS,
            patience=Config.EARLY_STOPPING_PATIENCE,
            save_path=save_path,
        )

        # ==========================================
        # 3. Fold Validation & Feature Extraction
        # ==========================================
        # Load best weights for inference
        model = load_checkpoint(model, save_path, device)
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Inference
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                # Store predictions and targets
                fold_preds.extend(probs)
                fold_targets.extend(labels.numpy())

                # Collect features for failure analysis
                oof_angles.extend(angles.cpu().numpy())

                # Calculate image statistics (Mean and Std) for the batch
                # images shape: (B, C, H, W)
                imgs_np = images.cpu().numpy()
                batch_means = np.mean(imgs_np, axis=(1, 2, 3))
                batch_stds = np.std(imgs_np, axis=(1, 2, 3))

                oof_img_means.extend(batch_means)
                oof_img_stds.extend(batch_stds)

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

    # ==========================================
    # 4. Global Evaluation
    # ==========================================
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Clip predictions to avoid log(0)
    eps = 1e-15
    oof_preds_clipped = np.clip(oof_preds, eps, 1 - eps)

    final_metric = log_loss(oof_targets, oof_preds_clipped, labels=[0, 1])
    print(f"\nFinal Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    errors = np.abs(oof_targets - oof_preds)
    oof_angles = np.array(oof_angles)
    oof_img_means = np.array(oof_img_means)
    oof_img_stds = np.array(oof_img_stds)

    # Correlation: Error vs Incidence Angle
    # Filter out NaNs if any (though loader imputes them)
    valid_ang_mask = ~np.isnan(oof_angles)
    if np.sum(valid_ang_mask) > 1:
        corr_ang = np.corrcoef(errors[valid_ang_mask], oof_angles[valid_ang_mask])[0, 1]
        print(f"Correlation (Error vs Incidence Angle): {corr_ang:.4f}")
    else:
        print("Correlation (Error vs Incidence Angle): N/A")

    # Correlation: Error vs Image Brightness
    corr_mean = np.corrcoef(errors, oof_img_means)[0, 1]
    print(f"Correlation (Error vs Image Mean Intensity): {corr_mean:.4f}")

    # Correlation: Error vs Image Contrast
    corr_std = np.corrcoef(errors, oof_img_stds)[0, 1]
    print(f"Correlation (Error vs Image Contrast/Std): {corr_std:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = 0.15744295919935183

    if final_metric < threshold:
        print(
            f"\nMetric meets threshold ({final_metric} < {threshold}). Generating submission..."
        )

        # Retrieve Test IDs
        data_cache = process_data(load_cached_data=True)
        test_ids = data_cache["ids_test"]

        # Get Test Loader (Fold 0 is sufficient as test set is constant)
        _, _, test_loader = get_loaders(fold_idx=0, load_cached_data=True)

        ensemble_preds = np.zeros(len(test_ids))

        # Ensemble Inference
        for fold in range(Config.N_FOLDS):
            print(f"Inference Fold {fold}...")
            model = DMWBN().to(device)
            save_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold}.pth")
            model = load_checkpoint(model, save_path, device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            ensemble_preds += np.array(fold_preds)

        # Average predictions
        avg_preds = ensemble_preds / Config.N_FOLDS

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        # Save
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} failed to meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
