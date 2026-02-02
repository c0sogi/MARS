import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torchvision import transforms

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import process_data, IcebergDataset
from library.model import MADResNet
from library.engine import train_one_epoch, evaluate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    # Utilizing cached data as requested
    data = process_data(load_cached_data=True)

    # Merge Train and Val sets for Stratified K-Fold CV
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    angle_full = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # 3. Initialize Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store results
    oof_preds = np.zeros(len(y_full))
    test_preds_accum = np.zeros(len(X_test))

    # Define Transforms (Augmentation for training only)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split data
        X_tr, X_va = X_full[train_idx], X_full[val_idx]
        ang_tr, ang_va = angle_full[train_idx], angle_full[val_idx]
        y_tr, y_va = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(X_tr, ang_tr, labels=y_tr, transform=train_transform)
        val_ds = IcebergDataset(X_va, ang_va, labels=y_va, transform=None)

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Loss
        model = MADResNet().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training
        best_val_loss = float("inf")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_loss": best_val_loss,
                    },
                    is_best=True,
                    fold=fold,
                    output_dir=Config.IDEA_DIR,
                )

        # Load best model for inference
        best_model_path = os.path.join(Config.IDEA_DIR, f"model_best_fold_{fold}.pth")
        load_checkpoint(best_model_path, model, device=device)

        # Generate OOF Predictions
        # Re-create val loader without shuffle just to be safe (though it was False above)
        val_loader_inf = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        fold_oof_preds = predict(model, val_loader_inf, device)
        oof_preds[val_idx] = fold_oof_preds

        # Generate Test Predictions
        test_ds = IcebergDataset(X_test, angle_test, ids=ids_test, transform=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds

        # Cleanup
        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

    # 5. Validation Assessment
    final_metric = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error
    errors = np.abs(y_full - oof_preds)

    # Extract features for correlation
    # inc_angle
    angles = angle_full
    # Mean intensity of Band 1 (HH) and Band 2 (HV)
    # X_full shape is (N, 3, 75, 75). Channel 0 is HH, Channel 1 is HV.
    mean_b1 = X_full[:, 0, :, :].mean(axis=(1, 2))
    mean_b2 = X_full[:, 1, :, :].mean(axis=(1, 2))

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {"error": errors, "inc_angle": angles, "mean_b1": mean_b1, "mean_b2": mean_b2}
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.18120490171618245

    if final_metric < THRESHOLD:
        avg_test_preds = test_preds_accum / Config.NUM_FOLDS

        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        # Ensure output directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
