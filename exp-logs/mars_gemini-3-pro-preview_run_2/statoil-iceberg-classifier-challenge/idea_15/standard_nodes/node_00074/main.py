import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings

# Import library modules
from library import config, utils, model, data_loader, train

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load cached data or process from scratch
    X_train, y_train, inc_train, X_test, inc_test, ids_test = data_loader.process_data(
        load_cached_data=True
    )

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    oof_preds = np.zeros(len(y_train))
    trained_models = []

    print(f"\nStarting {config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n{'='*10} Fold {fold} {'='*10}")

        # Prepare Datasets
        # Note: We manually create datasets/loaders here to have direct access to indices
        # instead of using data_loader.get_fold_loaders which hides them.
        train_ds = data_loader.IcebergDataset(
            X_train[train_idx], inc_train[train_idx], y_train[train_idx], transform=True
        )
        val_ds = data_loader.IcebergDataset(
            X_train[val_idx], inc_train[val_idx], y_train[val_idx], transform=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model and Training Components
        net = model.DPCNet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

        # Train
        trainer = train.Trainer(
            model=net,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            patience=config.PATIENCE,
        )

        best_model = trainer.fit(train_loader, val_loader, max_epochs=config.MAX_EPOCHS)

        # Save Model
        save_path = os.path.join(config.WORKING_DIR, f"gdpnet_fold_{fold}.pth")
        utils.save_checkpoint(best_model, save_path)
        trained_models.append(best_model)

        # Generate OOF Predictions for this fold
        best_model.eval()
        fold_probs = []

        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = best_model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)

        # Store predictions
        oof_preds[val_idx] = np.concatenate(fold_probs).flatten()

    # 4. Validation Assessment
    # Clip predictions to avoid log(0)
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_train, oof_preds_clipped)

    print(f"\nFinal Validation Metric: {final_metric:.10f}")

    # 5. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(y_train - oof_preds)

    # Calculate mean intensity per image for correlation
    # X_train is (N, 3, 75, 75)
    mean_intensity = np.mean(X_train, axis=(1, 2, 3))

    # Correlation with Incidence Angle
    corr_inc = np.corrcoef(errors, inc_train)[0, 1]
    # Correlation with Mean Intensity
    corr_int = np.corrcoef(errors, mean_intensity)[0, 1]

    print(f"Correlation (Error vs Inc Angle): {corr_inc:.6f}")
    print(f"Correlation (Error vs Mean Intensity): {corr_int:.6f}")

    # 6. Submission Generation
    threshold = 0.16676861786296204

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric:.6f} < Threshold {threshold:.6f}. Generating submission..."
        )

        # Prepare Test Loader
        test_ds = data_loader.IcebergDataset(X_test, inc_test, transform=False)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Inference
        avg_preds = np.zeros(len(X_test))

        for i, net in enumerate(trained_models):
            print(f"Predicting with model fold {i}...")
            net.eval()
            fold_test_preds = []

            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)
                    outputs = net(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    fold_test_preds.append(probs)

            avg_preds += np.concatenate(fold_test_preds).flatten()

        # Average predictions
        avg_preds /= config.NUM_FOLDS

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric:.6f} >= Threshold {threshold:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
