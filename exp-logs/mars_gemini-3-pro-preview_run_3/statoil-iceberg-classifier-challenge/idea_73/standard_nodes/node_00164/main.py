import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.model import ADSICNN
from library.data_loader import load_data, get_dataloaders, get_test_loader
from library.trainer import train_one_epoch, validate
from library.utils import set_seed


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Fast baseline settings
    EPOCHS = 30

    print(f"Initializing training on device: {device}")

    # 2. Load Data
    # load_data handles caching and returns numpy arrays
    X, y, angles, ids_train, X_test, angle_test, ids_test = load_data(
        load_cached_data=True
    )

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store OOF predictions and Test predictions
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # Get DataLoaders (handles leak-free imputation and augmentations)
        train_loader, val_loader = get_dataloaders(X, y, angles, train_idx, val_idx)

        # Initialize Model, Optimizer, Criterion
        model = ADSICNN().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_metric = validate(model, val_loader, criterion, device)

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                break

        # Load best model for inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()

        # Generate OOF Predictions
        # Iterate val_loader to ensure correct order and preprocessing
        fold_oof_preds = []
        with torch.no_grad():
            for imgs, angs, _ in val_loader:
                imgs = imgs.to(device)
                angs = angs.to(device)
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs)
                fold_oof_preds.extend(probs.cpu().numpy())

        # Store OOF predictions
        oof_preds[val_idx] = np.array(fold_oof_preds).flatten()

        # Generate Test Predictions
        # Use full training angles for imputation reference
        test_loader = get_test_loader(X_test, angle_test, angles)
        fold_test_preds = []
        with torch.no_grad():
            for imgs, angs in test_loader:
                imgs = imgs.to(device)
                angs = angs.to(device)
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs)
                fold_test_preds.extend(probs.cpu().numpy())

        # Accumulate test predictions
        test_preds_accum += np.array(fold_test_preds).flatten()

    # 4. Final Validation Metric
    final_metric = log_loss(y, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate error magnitude
    errors = np.abs(y - oof_preds)

    # Prepare features for correlation
    # Impute angles globally for analysis purposes (simple median)
    angles_imputed = np.where(np.isnan(angles), np.nanmedian(angles), angles)

    # Calculate image statistics (Mean and Std for Band 1 and Band 2)
    # X shape is (N, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
    b1_mean = np.mean(X[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X[:, 1, :, :], axis=(1, 2))
    b1_std = np.std(X[:, 0, :, :], axis=(1, 2))
    b2_std = np.std(X[:, 1, :, :], axis=(1, 2))

    analysis_features = {
        "Incidence Angle": angles_imputed,
        "Band 1 Mean": b1_mean,
        "Band 2 Mean": b2_mean,
        "Band 1 Std": b1_std,
        "Band 2 Std": b2_std,
    }

    print("Correlation between Error Magnitude and Features:")
    for name, feat_values in analysis_features.items():
        # Compute correlation [0,1] is the correlation coefficient
        corr = np.corrcoef(errors, feat_values)[0, 1]
        print(f"  {name}: {corr:.10f}")

    # 6. Submission Generation
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        avg_test_preds = test_preds_accum / Config.N_FOLDS

        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission generated and saved to {sub_path}")
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
