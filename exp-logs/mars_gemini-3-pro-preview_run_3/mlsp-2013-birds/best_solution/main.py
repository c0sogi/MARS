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
    get_holdout_loader,
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
    # Cite {solution_lesson_node_00056}: Force regeneration of folds to exclude hold-out set
    folds_df = create_folds(load_cached_data=False)

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
    # Cite {solution_lesson_node_00056}: Evaluate the entire ensemble on the fixed hold-out set.
    print("Starting Validation Inference on Hold-Out Set...")

    val_loader = get_holdout_loader(hist_features)

    # Collect targets once
    val_targets = []
    val_ids_list = []
    for batch in val_loader:
        val_targets.append(batch["target"].numpy())
        val_ids_list.append(batch["rec_id"].numpy())
    y_true_val = np.vstack(val_targets)
    val_ids_flat = np.concatenate(val_ids_list)

    # --- MLP Ensemble Inference ---
    mlp_preds_accum = []
    for fold_idx in range(Config.N_FOLDS):
        mlp_ckpt_path = os.path.join(ckpt_root, f"mlp_fold_{fold_idx}_best.pth")
        if not os.path.exists(mlp_ckpt_path):
            continue

        mlp_model = SymbolicMLP().to(device)
        mlp_model.load_state_dict(torch.load(mlp_ckpt_path, map_location=device))
        mlp_model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                out = mlp_model(feats)
                fold_probs.append(torch.sigmoid(out).cpu().numpy())
        mlp_preds_accum.append(np.vstack(fold_probs))

    avg_mlp_val = (
        np.mean(mlp_preds_accum, axis=0)
        if mlp_preds_accum
        else np.zeros_like(y_true_val)
    )

    # --- CNN Ensemble Inference (All Snapshots) ---
    cnn_preds_accum = []

    for model_name in Config.CNN_MODELS:
        model_ckpt_dir = os.path.join(ckpt_root, model_name)
        # Cite {debug_lesson_13}: Enforce strict file search pattern to avoid artifact pollution.
        snapshots = glob.glob(os.path.join(model_ckpt_dir, f"{model_name}_*.pth"))

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
            cnn_preds_accum.append(np.vstack(snap_probs))

    avg_cnn_val = (
        np.mean(cnn_preds_accum, axis=0)
        if cnn_preds_accum
        else np.zeros_like(y_true_val)
    )

    # --- Fusion ---
    y_pred_val = (avg_mlp_val + avg_cnn_val) / 2.0

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
    # Cite {solution_lesson_node_00040}: Failure analysis on hold-out set.
    # Use ground truth matrix directly to count labels
    label_counts = np.sum(y_true_val, axis=1)

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
            # Cite {debug_lesson_13}: Enforce strict file search pattern for test inference.
            snapshots = glob.glob(os.path.join(model_ckpt_dir, f"{model_name}_*.pth"))

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
