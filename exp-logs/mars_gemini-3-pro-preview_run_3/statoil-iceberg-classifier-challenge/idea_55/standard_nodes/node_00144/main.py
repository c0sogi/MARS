import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided libraries
from library.utils import set_seed, get_logger, save_checkpoint, load_checkpoint
from library.dataset import load_and_process_data, IcebergDataset, get_transforms
from library.model import SPCNN
from library.train import train_one_epoch, validate_one_epoch


def analyze_failures(X, angles, y_true, y_pred):
    """
    Analyzes the correlation between prediction error and input features.
    """
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Extract features
    # X shape: (N, 3, 75, 75) -> Band 1 is index 0, Band 2 is index 1
    # We compute statistics over the spatial dimensions (axis 2 and 3)
    b1_mean = np.mean(X[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X[:, 1, :, :], axis=(1, 2))

    features = {
        "inc_angle": angles,
        "b1_mean": b1_mean,
        "b1_std": b1_std,
        "b2_mean": b2_mean,
        "b2_std": b2_std,
    }

    print("\n--- Failure Analysis (Correlation with Error) ---")
    for name, feat_values in features.items():
        # Handle NaNs in angles if any (though they should be imputed)
        mask = ~np.isnan(feat_values)
        if np.sum(mask) > 0:
            corr = np.corrcoef(errors[mask], feat_values[mask])[0, 1]
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: NaN (insufficient data)")


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    work_dir = "./working/idea_55_run"
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    submission_dir = os.path.join("./submission")  # Target location
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    logger = get_logger(os.path.join(work_dir, "run.log"))
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    # Load cached data using the library function
    train_data_raw, val_data_raw, test_data_raw = load_and_process_data(
        load_cached_data=True
    )

    # Combine train and val for 5-Fold CV to use all labeled data
    X_full = np.concatenate([train_data_raw["X"], val_data_raw["X"]], axis=0)
    ang_full = np.concatenate(
        [train_data_raw["angles"], val_data_raw["angles"]], axis=0
    )
    ids_full = np.concatenate([train_data_raw["ids"], val_data_raw["ids"]], axis=0)
    y_full = np.concatenate([train_data_raw["y"], val_data_raw["y"]], axis=0)

    logger.info(f"Total training samples: {len(y_full)}")

    # 3. Cross-Validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store OOF predictions
    oof_preds = np.zeros(len(y_full))

    # Training parameters
    epochs = 50  # Fast baseline
    patience = 10
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"--- Starting Fold {fold} ---")

        # Split data
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        ang_train, ang_val = ang_full[train_idx], ang_full[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]
        ids_train, ids_val = ids_full[train_idx], ids_full[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train, ang_train, ids_train, y_train, transform=get_transforms("train")
        )
        val_dataset = IcebergDataset(
            X_val, ang_val, ids_val, y_val, transform=get_transforms("val")
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Model & Optimizer
        model = SPCNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

            # Checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {"state_dict": model.state_dict(), "val_loss": val_loss},
                    is_best=True,
                    checkpoint_dir=checkpoint_dir,
                    fold=fold,
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    f"Fold {fold} Early stopping at epoch {epoch}. Best Loss: {best_loss:.6f}"
                )
                break

        # Load best model for OOF prediction
        best_path = os.path.join(checkpoint_dir, f"model_best_fold_{fold}.pth")
        load_checkpoint(best_path, model, device=device)
        model.eval()

        # Generate OOF preds
        fold_preds = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                angs = batch["angle"].to(device)
                logits = model(imgs, angs)
                probs = torch.sigmoid(logits)
                fold_preds.extend(probs.cpu().numpy())

        oof_preds[val_idx] = np.array(fold_preds)

        # Clean up
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    final_metric = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # 5. Failure Analysis
    analyze_failures(X_full, ang_full, y_full, oof_preds)

    # 6. Submission
    # Threshold from requirements
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        logger.info("Metric met threshold. Generating submission...")

        # Prepare Test Loader
        test_dataset = IcebergDataset(
            test_data_raw["X"],
            test_data_raw["angles"],
            test_data_raw["ids"],
            y=None,
            transform=get_transforms("test"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Ensemble Inference
        test_preds_accum = np.zeros(len(test_data_raw["ids"]))

        for fold in range(n_folds):
            model = SPCNN().to(device)
            best_path = os.path.join(checkpoint_dir, f"model_best_fold_{fold}.pth")
            try:
                load_checkpoint(best_path, model, device=device)
                model.eval()

                fold_test_preds = []
                with torch.no_grad():
                    for batch in test_loader:
                        imgs = batch["image"].to(device)
                        angs = batch["angle"].to(device)
                        logits = model(imgs, angs)
                        probs = torch.sigmoid(logits)
                        fold_test_preds.extend(probs.cpu().numpy())

                test_preds_accum += np.array(fold_test_preds)
            except FileNotFoundError:
                logger.error(f"Checkpoint for fold {fold} not found. Skipping.")

        # Average
        avg_preds = test_preds_accum / n_folds

        # Save
        sub_df = pd.DataFrame({"id": test_data_raw["ids"], "is_iceberg": avg_preds})
        sub_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")
    else:
        logger.info(
            f"Metric {final_metric:.6f} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
