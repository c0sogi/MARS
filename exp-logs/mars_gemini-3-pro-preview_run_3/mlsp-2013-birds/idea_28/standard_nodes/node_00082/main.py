import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, RobustMetric, SnapshotManager
from library.data_factory import (
    load_histogram_features,
    create_folds,
    get_loaders,
    get_test_loader,
)
from library.model_factory import get_cnn_model, SymbolicMLP
from library.engine import train_one_epoch, validate_one_epoch


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Preparation
    # Load cached features and fold definitions
    hist_features = load_histogram_features(load_cached_data=True)
    folds_df = create_folds(load_cached_data=True)

    # Prepare checkpoint directory
    ckpt_root = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    # 3. Training Loop (Iterative Stratified K-Fold)
    print("Starting Training Loop...")

    for fold_idx in range(Config.N_FOLDS):
        # Get DataLoaders for the current fold
        train_loader, val_loader = get_loaders(fold_idx, folds_df, hist_features)

        # --- Stream A: Symbolic MLP Training ---
        mlp_model = SymbolicMLP().to(device)
        mlp_optimizer = AdamW(
            mlp_model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_mlp_auc = 0.0
        mlp_ckpt_path = os.path.join(ckpt_root, f"mlp_fold_{fold_idx}_best.pth")

        # Train MLP
        for epoch in range(Config.EPOCHS):
            _ = train_one_epoch(
                mlp_model,
                train_loader,
                mlp_optimizer,
                criterion,
                device,
                input_key="features",
                mixup_alpha=0.4,  # Manifold mixup
            )
            _, val_auc = validate_one_epoch(
                mlp_model, val_loader, criterion, device, input_key="features"
            )

            if val_auc > best_mlp_auc:
                best_mlp_auc = val_auc
                torch.save(mlp_model.state_dict(), mlp_ckpt_path)

        # --- Stream B: Deep CNN Ensemble Training ---
        for model_name in Config.CNN_MODELS:
            cnn_model = get_cnn_model(model_name).to(device)
            cnn_optimizer = AdamW(
                cnn_model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            cnn_scheduler = CosineAnnealingLR(cnn_optimizer, T_max=Config.EPOCHS)

            # Snapshot Manager for this specific model architecture and fold
            model_ckpt_dir = os.path.join(ckpt_root, model_name)
            snapshot_manager = SnapshotManager(
                model_ckpt_dir, k=Config.SNAPSHOTS_K, maximize=True
            )

            # Train CNN
            for epoch in range(Config.EPOCHS):
                _ = train_one_epoch(
                    cnn_model,
                    train_loader,
                    cnn_optimizer,
                    criterion,
                    device,
                    input_key="image",
                    mixup_alpha=Config.MIXUP_ALPHA,
                )
                _, val_auc = validate_one_epoch(
                    cnn_model, val_loader, criterion, device, input_key="image"
                )

                cnn_scheduler.step()

                # Save snapshot if it's among the top-K
                snapshot_manager.save(cnn_model, val_auc, epoch, fold_idx, model_name)

    # 4. Validation Inference (Ensemble)
    print("Starting Validation Inference...")

    final_val_preds = []
    final_val_targets = []
    final_val_ids = []

    # Iterate folds to generate out-of-fold predictions
    for fold_idx in range(Config.N_FOLDS):
        _, val_loader = get_loaders(fold_idx, folds_df, hist_features)

        # --- MLP Inference ---
        mlp_model = SymbolicMLP().to(device)
        mlp_ckpt_path = os.path.join(ckpt_root, f"mlp_fold_{fold_idx}_best.pth")
        if os.path.exists(mlp_ckpt_path):
            mlp_model.load_state_dict(torch.load(mlp_ckpt_path, map_location=device))
        mlp_model.eval()

        mlp_fold_probs = []
        targets_list = []
        ids_list = []

        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                targs = batch["target"].to(device)
                rids = batch["rec_id"]

                out = mlp_model(feats)
                probs = torch.sigmoid(out).cpu().numpy()

                mlp_fold_probs.append(probs)
                targets_list.append(targs.cpu().numpy())
                ids_list.append(rids.numpy())

        fold_preds_mlp = np.vstack(mlp_fold_probs)
        fold_targets = np.vstack(targets_list)
        fold_ids = np.concatenate(ids_list)

        # --- CNN Inference (Snapshot Ensemble) ---
        cnn_architecture_preds = []

        for model_name in Config.CNN_MODELS:
            model_ckpt_dir = os.path.join(ckpt_root, model_name)
            # Find snapshots for this fold
            pattern = os.path.join(model_ckpt_dir, f"{model_name}_fold{fold_idx}_*.pth")
            snapshots = glob.glob(pattern)

            if not snapshots:
                continue

            model_snapshot_preds = []

            for snap_path in snapshots:
                model = get_cnn_model(model_name).to(device)
                model.load_state_dict(torch.load(snap_path, map_location=device))
                model.eval()

                snap_probs = []
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch["image"].to(device)
                        out = model(imgs)
                        snap_probs.append(torch.sigmoid(out).cpu().numpy())

                model_snapshot_preds.append(np.vstack(snap_probs))

            # Average across snapshots for this architecture
            if model_snapshot_preds:
                avg_arch_pred = np.mean(model_snapshot_preds, axis=0)
                cnn_architecture_preds.append(avg_arch_pred)

        # Average across CNN architectures
        if cnn_architecture_preds:
            fold_preds_cnn = np.mean(cnn_architecture_preds, axis=0)
        else:
            fold_preds_cnn = np.zeros_like(fold_preds_mlp)

        # --- Fusion ---
        # Weighted Average (50% MLP, 50% CNN Ensemble)
        fold_final_preds = (fold_preds_mlp + fold_preds_cnn) / 2.0

        final_val_preds.append(fold_final_preds)
        final_val_targets.append(fold_targets)
        final_val_ids.append(fold_ids)

    # Concatenate all folds to form full validation set
    y_pred_val = np.vstack(final_val_preds)
    y_true_val = np.vstack(final_val_targets)
    val_ids_flat = np.concatenate(final_val_ids)

    # 5. Metric Calculation
    # Calculate Macro-Averaged AUC
    auc_scores = []
    for i in range(Config.NUM_CLASSES):
        # Only calculate if class is present in validation set
        if len(np.unique(y_true_val[:, i])) > 1:
            auc_scores.append(roc_auc_score(y_true_val[:, i], y_pred_val[:, i]))

    final_metric = np.mean(auc_scores) if auc_scores else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate per-sample Mean Absolute Error
    per_sample_error = np.mean(np.abs(y_true_val - y_pred_val), axis=1)

    # Feature for correlation: Number of Labels (Complexity)
    label_counts = []
    for rid in val_ids_flat:
        row = folds_df[folds_df["rec_id"] == rid]
        if not row.empty:
            lbl_str = str(row.iloc[0]["labels"])
            if lbl_str == "?" or lbl_str == "nan":
                count = 0
            else:
                count = len(lbl_str.split())
            label_counts.append(count)
        else:
            label_counts.append(0)

    label_counts = np.array(label_counts)

    if len(per_sample_error) > 1:
        corr, _ = pearsonr(per_sample_error, label_counts)
        print(f"Correlation between Error and Label Count: {corr}")

    # 7. Test Inference & Submission
    threshold = 0.9479806884980326

    if final_metric > threshold:
        print("Metric threshold met. Generating submission...")
        test_loader = get_test_loader(hist_features)

        # --- MLP Test Preds (Average 5 fold models) ---
        mlp_test_preds_folds = []
        for fold_idx in range(Config.N_FOLDS):
            mlp_model = SymbolicMLP().to(device)
            mlp_ckpt_path = os.path.join(ckpt_root, f"mlp_fold_{fold_idx}_best.pth")
            if os.path.exists(mlp_ckpt_path):
                mlp_model.load_state_dict(
                    torch.load(mlp_ckpt_path, map_location=device)
                )
                mlp_model.eval()

                fold_probs = []
                with torch.no_grad():
                    for batch in test_loader:
                        feats = batch["features"].to(device)
                        out = mlp_model(feats)
                        fold_probs.append(torch.sigmoid(out).cpu().numpy())
                mlp_test_preds_folds.append(np.vstack(fold_probs))

        avg_mlp_test = np.mean(mlp_test_preds_folds, axis=0)

        # --- CNN Test Preds (Average all snapshots from all models) ---
        cnn_test_preds_accum = []

        for model_name in Config.CNN_MODELS:
            model_ckpt_dir = os.path.join(ckpt_root, model_name)
            snapshots = glob.glob(os.path.join(model_ckpt_dir, "*.pth"))

            for snap_path in snapshots:
                model = get_cnn_model(model_name).to(device)
                model.load_state_dict(torch.load(snap_path, map_location=device))
                model.eval()

                snap_probs = []
                with torch.no_grad():
                    for batch in test_loader:
                        imgs = batch["image"].to(device)
                        out = model(imgs)
                        snap_probs.append(torch.sigmoid(out).cpu().numpy())

                cnn_test_preds_accum.append(np.vstack(snap_probs))

        avg_cnn_test = np.mean(cnn_test_preds_accum, axis=0)

        # --- Final Fusion ---
        final_test_probs = (avg_mlp_test + avg_cnn_test) / 2.0

        # --- Submission File Generation ---
        test_df = pd.read_csv(Config.TEST_CSV)
        test_rec_ids = test_df["rec_id"].values

        submission_rows = []
        for i, rec_id in enumerate(test_rec_ids):
            probs = final_test_probs[i]
            for species_id in range(Config.NUM_CLASSES):
                # Format: rec_id * 100 + species_id
                row_id = rec_id * 100 + species_id
                prob = probs[species_id]
                submission_rows.append({"Id": row_id, "Probability": prob})

        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Metric {final_metric} did not exceed threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run()
