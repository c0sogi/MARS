import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

# Import provided library components
from library.model import SWDINet
from library.data_loader import get_fold_loaders
from library.train_eval import train_one_epoch, validate, predict, EarlyStopping
from library.utils import seed_everything


def main():
    # ==========================================
    # 1. Configuration
    # ==========================================
    SEED = 42
    # Cite {solution_lesson_node_00023}: "Low and Slow" optimization strategy
    EPOCHS = 60
    BATCH_SIZE = 32
    PATIENCE = 15
    THRESHOLD = 0.16676861786296204
    OUTPUT_DIR = "./submission"
    N_SPLITS = 5

    # Ensure reproducibility
    seed_everything(SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Storage for Global Metrics and Ensemble
    oof_preds = []
    oof_targets = []
    oof_angles = []

    test_preds_accum = None
    test_ids = None

    # ==========================================
    # 2. Cross-Validation Loop
    # ==========================================
    for fold in range(N_SPLITS):
        print(f"\n=== FOLD {fold} ===")

        # Get Fold DataLoaders (Handles independent scaling internally)
        train_loader, val_loader, test_loader, ids_test_fold = get_fold_loaders(
            fold_idx=fold, n_splits=N_SPLITS, batch_size=BATCH_SIZE, seed=SEED
        )

        # Initialize Test Accumulator once
        if test_ids is None:
            test_ids = ids_test_fold
            test_preds_accum = np.zeros((len(ids_test_fold), 1))

        # Initialize Model, Optimizer, Scheduler
        model = SWDINet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False)

        # --- Training Loop ---
        for epoch in range(EPOCHS):
            # Train
            t_loss, t_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # Validate (for scheduler/early stopping)
            v_loss, v_acc, _ = validate(model, val_loader, criterion, device)

            scheduler.step(v_loss)
            early_stopping(v_loss, model)

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Tr Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}"
                )

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load Best Weights
        early_stopping.load_best_weights(model)

        # --- Validation Inference (for OOF & Failure Analysis) ---
        model.eval()
        fold_val_preds = []
        fold_val_targets = []
        fold_val_angles = []

        with torch.no_grad():
            for imgs, angles, labels in val_loader:
                imgs = imgs.to(device)
                angles = angles.to(device)

                # Forward pass
                out = model(imgs, angles)
                probs = torch.sigmoid(out).cpu().numpy()

                # Collect data
                fold_val_preds.extend(probs.flatten())
                fold_val_targets.extend(labels.numpy().flatten())
                fold_val_angles.extend(angles.cpu().numpy().flatten())

        oof_preds.extend(fold_val_preds)
        oof_targets.extend(fold_val_targets)
        oof_angles.extend(fold_val_angles)

        # --- Test Inference (Ensemble) ---
        fold_test_preds = predict(model, test_loader, device)
        test_preds_accum += fold_test_preds.reshape(-1, 1)

    # ==========================================
    # 3. Global Metrics & Failure Analysis
    # ==========================================
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)
    oof_angles = np.array(oof_angles)

    # Clip predictions for Log Loss stability
    oof_preds_clipped = np.clip(oof_preds, 0.001, 0.999)

    # Compute Final Metric
    final_log_loss = log_loss(oof_targets, oof_preds_clipped)
    print(f"Final Validation Metric: {final_log_loss}")

    # Failure Analysis: Correlation between Error and Incidence Angle
    errors = np.abs(oof_targets - oof_preds)

    # Filter out any potential NaNs in angles (though loader handles this, good practice)
    valid_mask = ~np.isnan(oof_angles)
    if np.sum(valid_mask) > 0:
        corr = np.corrcoef(errors[valid_mask], oof_angles[valid_mask])[0, 1]
        print(f"Correlation between Error and Incidence Angle: {corr}")
    else:
        print("Could not calculate correlation due to invalid incidence angles.")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    if final_log_loss < THRESHOLD:
        print(
            f"Validation metric meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Average predictions across folds
        avg_test_preds = test_preds_accum / N_SPLITS

        # Clip for safety
        avg_test_preds = np.clip(avg_test_preds, 0.001, 0.999)

        # Create DataFrame
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds.flatten()})

        # Save
        sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Validation metric {final_log_loss} is NOT lower than {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
