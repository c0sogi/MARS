import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.modeling import get_model, verify_initialization
from library.training import train_one_epoch, validate, predict

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# Optimization Configuration
Config.EPOCHS = 12
Config.T_0 = (
    12  # Synchronize scheduler cycle with epochs Cite {solution_lesson_node_00015}
)
Config.N_FOLDS = 5  # Robust validation Cite {solution_lesson_node_00009}
Config.PHASE2_SEEDS = [42, 101, 2022, 999, 12345]


def main():
    print("Starting Fast Baseline Run...")
    seed_everything(Config.SEED)

    # Ensure model directory exists
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # =========================================================================
    # Phase 1: Proxy Calibration (Reduced Folds)
    # =========================================================================
    print("\n" + "=" * 30)
    print("PHASE 1: Proxy Calibration")
    print("=" * 30)

    oof_preds_no_tta = []
    oof_preds_tta = []
    oof_targets = []
    oof_features = []  # Store brightness/contrast for failure analysis

    val_auc_history = []

    for fold in range(Config.N_FOLDS):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")
        seed_everything(Config.SEED + fold)

        # 1. Data Loading
        train_loader, val_loader, _, class_weights = get_dataloaders(
            fold_idx=fold, phase="phase1"
        )
        class_weights = class_weights.to(Config.DEVICE)

        # 2. Model Setup
        model = get_model(device=Config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

        # 3. Initialization Verification
        # Using unweighted loss for verification as initialization targets natural priors
        verify_initialization(model, train_loader, nn.CrossEntropyLoss(), Config.DEVICE)

        # 4. Training Loop
        best_auc = -1.0
        best_model_path = os.path.join(models_dir, f"phase1_fold_{fold}.pth")

        fold_aucs = []

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, Config.DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)
            scheduler.step()

            fold_aucs.append(val_auc)
            print(
                f"  Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val AUC {val_auc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        val_auc_history.append(fold_aucs)

        # 5. OOF Prediction & Feature Extraction
        print(f"  Generating OOF predictions for Fold {fold+1}...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        fold_preds_no = []
        fold_preds_tta = []
        fold_targs = []
        fold_feats = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.DEVICE)

                # Extract Features for Failure Analysis: Brightness (mean), Contrast (std)
                # Images are normalized, but relative stats still hold
                brightness = images.mean(dim=(1, 2, 3)).cpu().numpy()
                contrast = images.std(dim=(1, 2, 3)).cpu().numpy()
                batch_feats = np.stack([brightness, contrast], axis=1)
                fold_feats.append(batch_feats)

                # Targets
                fold_targs.append(labels.numpy())

                # Predictions: No TTA
                out = model(images)
                prob = torch.softmax(out, dim=1)
                fold_preds_no.append(prob.cpu().numpy())

                # Predictions: TTA (Horizontal + Vertical Flip)
                # H-Flip
                out_h = model(torch.flip(images, [3]))
                prob_h = torch.softmax(out_h, dim=1)
                # V-Flip
                out_v = model(torch.flip(images, [2]))
                prob_v = torch.softmax(out_v, dim=1)

                avg_prob = (prob + prob_h + prob_v) / 3.0
                fold_preds_tta.append(avg_prob.cpu().numpy())

        oof_preds_no_tta.append(np.concatenate(fold_preds_no))
        oof_preds_tta.append(np.concatenate(fold_preds_tta))
        oof_targets.append(np.concatenate(fold_targs))
        oof_features.append(np.concatenate(fold_feats))

    # =========================================================================
    # Analysis & Metrics
    # =========================================================================
    print("\n" + "=" * 30)
    print("ANALYSIS")
    print("=" * 30)

    # 1. Aggregate Results
    all_targets = np.concatenate(oof_targets)
    all_preds_no = np.concatenate(oof_preds_no_tta)
    all_preds_tta = np.concatenate(oof_preds_tta)
    all_features = np.concatenate(oof_features)

    # 2. Calculate Metrics
    auc_no = calculate_metric(all_targets, all_preds_no)
    auc_tta = calculate_metric(all_targets, all_preds_tta)

    final_metric = max(auc_no, auc_tta)
    use_tta = auc_tta > auc_no

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")
    print(
        f"TTA Strategy: {'Enabled' if use_tta else 'Disabled'} (No TTA: {auc_no:.6f}, TTA: {auc_tta:.6f})"
    )

    # 3. Determine Optimal Epoch
    avg_auc_curve = np.mean(np.array(val_auc_history), axis=0)
    optimal_epoch = int(np.argmax(avg_auc_curve) + 1)
    print(f"Optimal Epoch: {optimal_epoch}")

    # =========================================================================
    # Failure Analysis
    # =========================================================================
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    # Calculate Error Magnitude: 1.0 - Probability assigned to the true class
    true_indices = np.argmax(all_targets, axis=1)
    best_preds = all_preds_tta if use_tta else all_preds_no
    prob_true = best_preds[np.arange(len(best_preds)), true_indices]
    error_magnitude = 1.0 - prob_true

    # Features
    brightness_vals = all_features[:, 0]
    contrast_vals = all_features[:, 1]

    # Correlations
    corr_bright, _ = pearsonr(error_magnitude, brightness_vals)
    corr_contrast, _ = pearsonr(error_magnitude, contrast_vals)

    print(f"Correlation (Error vs Brightness): {corr_bright:.6f}")
    print(f"Correlation (Error vs Contrast): {corr_contrast:.6f}")

    # =========================================================================
    # Phase 2: Submission
    # =========================================================================
    THRESHOLD = 0.9871488489626378

    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric} > {THRESHOLD}. Proceeding to Phase 2 (Production Training)..."
        )

        ensemble_preds = []
        test_ids = None

        # Train on Full Data
        for seed in Config.PHASE2_SEEDS:
            print(f"\nTraining Seed Model (Seed: {seed})...")
            seed_everything(seed)

            # Load Full Data
            train_loader, _, test_loader, class_weights = get_dataloaders(
                phase="phase2"
            )
            class_weights = class_weights.to(Config.DEVICE)

            model = get_model(device=Config.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
            )

            # Verify Init
            verify_initialization(
                model, train_loader, nn.CrossEntropyLoss(), Config.DEVICE
            )

            # Train for optimal epoch
            for epoch in range(optimal_epoch):
                loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, Config.DEVICE
                )
                scheduler.step()
                print(f"  Epoch {epoch+1}/{optimal_epoch}: Train Loss {loss:.4f}")

            # Predict
            print("  Generating predictions...")
            ids, preds = predict(model, test_loader, Config.DEVICE, use_tta=use_tta)
            ensemble_preds.append(preds)
            if test_ids is None:
                test_ids = ids

        # Ensemble Averaging
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Create Submission DataFrame
        df_sub = pd.DataFrame(
            {
                "image_id": test_ids,
                "healthy": avg_preds[:, 0],
                "multiple_diseases": avg_preds[:, 1],
                "rust": avg_preds[:, 2],
                "scab": avg_preds[:, 3],
            }
        )

        # Ensure column order
        df_sub = df_sub[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"\nSubmission saved to {Config.SUBMISSION_PATH}")
        print(df_sub.head())

    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
