"""
Orchestration script for Apple Disease Detection (5-Fold CV Ensemble).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device, print_metric
from library.data import get_loaders, get_test_loader, get_class_weights, _process_data
from library.model import AppleDiseaseModel
from library.engine import fit
from library.inference import predict_ensemble, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # Initialize Config with debug=False to ensure we use the full dataset
    # to achieve the high metric requirement.
    cfg = Config(debug=False)

    # Set random seeds for reproducibility
    seed_everything(cfg.seed)

    # Detect device
    device = get_device()
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. 5-Fold Cross-Validation Training
    # -------------------------------------------------------------------------
    trained_models = []

    # Containers for Out-Of-Fold (OOF) predictions and targets
    # We will collect these to calculate the global CV metric
    oof_preds_list = []
    oof_targets_list = []
    oof_ids_list = []

    print(f"Starting {cfg.n_folds}-Fold Cross-Validation...")

    for fold in range(cfg.n_folds):
        print(f"\n" + "=" * 30)
        print(f"Training Fold {fold}")
        print("=" * 30)

        # --- Data Loading ---
        # get_loaders handles the splitting and transformation
        train_loader, val_loader = get_loaders(fold, cfg, load_cached_data=True)

        # Calculate class weights for imbalance handling
        pos_weights = get_class_weights(fold, cfg, load_cached_data=True).to(device)

        # --- Model Initialization ---
        model = AppleDiseaseModel(cfg.model_name, cfg.num_classes, pretrained=True)
        model.to(device)

        # --- Optimization ---
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # Cite solution_lesson_node_00014: Cosine Annealing for optimization stability
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_lr,
        )

        # Loss function with positive class weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        # --- Training ---
        # fit() handles the training loop, validation, and early stopping
        # It returns the model with the best weights loaded
        model, best_fold_auc = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            criterion,
            cfg,
            fold,
        )

        # Move model to CPU to save GPU memory while training next folds
        # We will move it back to GPU for inference later
        model.cpu()
        trained_models.append(model)

        # --- Generate OOF Predictions for this Fold ---
        # We need to run inference on the validation set again to store predictions
        model.to(device)
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(outputs)

                fold_preds.append(preds.cpu().numpy())
                fold_targets.append(targets.cpu().numpy())

        # Move model back to CPU
        model.cpu()

        # Store OOF results
        oof_preds_list.append(np.concatenate(fold_preds, axis=0))
        oof_targets_list.append(np.concatenate(fold_targets, axis=0))

        # --- Retrieve Image IDs for Failure Analysis ---
        # We reconstruct the validation split logic to get the corresponding image_ids
        full_df = _process_data(cfg, "train", load_cached_data=True)
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        _, val_idx = list(skf.split(full_df, full_df["stratify_label"]))[fold]
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        oof_ids_list.extend(val_df["image_id"].tolist())

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30)
    print("Validation Assessment")
    print("=" * 30)

    # Concatenate all OOF results
    all_oof_preds = np.concatenate(oof_preds_list, axis=0)
    all_oof_targets = np.concatenate(oof_targets_list, axis=0)

    # Calculate Mean Column-wise ROC AUC
    final_auc = roc_auc_score(all_oof_targets, all_oof_preds, average="macro")

    # Print metric in required format
    print(f"Final Validation Metric: {final_auc}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30)
    print("Failure Analysis")
    print("=" * 30)

    # Calculate error magnitude (Mean Absolute Error)
    # Shape: (N_samples, 2) -> Mean over classes -> (N_samples,)
    errors = np.mean(np.abs(all_oof_targets - all_oof_preds), axis=1)

    # Load metadata to access file paths
    full_df = _process_data(cfg, "train", load_cached_data=True)
    id_to_path = dict(zip(full_df["image_id"], full_df["file_path"]))

    # Extract features for correlation analysis
    widths = []
    heights = []
    file_sizes = []
    aspect_ratios = []

    print("Extracting image features for analysis...")
    for img_id in oof_ids_list:
        rel_path = id_to_path.get(img_id)
        full_path = os.path.join(cfg.input_dir, rel_path)

        try:
            # Get file size
            f_size = os.path.getsize(full_path)

            # Get dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                ar = w / h if h > 0 else 0
            else:
                h, w, ar = 0, 0, 0

            widths.append(w)
            heights.append(h)
            file_sizes.append(f_size)
            aspect_ratios.append(ar)

        except Exception as e:
            # Fallback for errors
            widths.append(0)
            heights.append(0)
            file_sizes.append(0)
            aspect_ratios.append(0)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "file_size": file_sizes,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.9954104122251848

    if final_auc > threshold:
        print(f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}).")
        print("Generating submission...")

        # Load Test Data
        test_loader = get_test_loader(cfg, load_cached_data=True)

        # Generate Ensemble Predictions
        # predict_ensemble handles moving models to device and TTA
        ids, preds = predict_ensemble(trained_models, test_loader, device)

        # Save Submission
        generate_submission(ids, preds, cfg.submission_path)

    else:
        print(
            f"\nValidation metric ({final_auc}) does NOT meet threshold ({threshold})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
