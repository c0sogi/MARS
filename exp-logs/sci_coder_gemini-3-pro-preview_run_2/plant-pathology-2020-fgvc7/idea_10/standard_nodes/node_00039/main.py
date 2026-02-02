import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import CFG
from library.utils import seed_everything, calculate_metric
from library.dataset import process_data, AppleDataset, get_transforms
from library.models import AppleNet
from library.trainer import run_fold
from library.stacking import (
    train_meta_learner,
    inference_meta_learner,
    create_submission,
    reconstruct_probs,
)

# --- Configuration Override for Fast Baseline ---
# Reducing epochs to ensure the script completes within the time limit
# while still providing a meaningful baseline.
CFG.epochs = 5


def predict_dataset(model, dataset, batch_size, device):
    """
    Runs inference on a dataset and returns probabilities.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    model.eval()
    probs_list = []

    with torch.no_grad():
        for batch in loader:
            # Unpack based on dataset mode (test returns 2 items, train/val returns 3 items)
            if len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch

            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def get_image_metadata(df):
    """
    Extracts metadata for failure analysis.
    """
    meta_stats = []
    for idx, row in df.iterrows():
        # Construct full path using input_dir since file_path is relative
        full_path = os.path.join(CFG.input_dir, row["file_path"])

        try:
            size = os.path.getsize(full_path)
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                intensity = img.mean()
            else:
                h, w, intensity = 0, 0, 0
        except:
            size, h, w, intensity = 0, 0, 0, 0

        meta_stats.append(
            {"file_size": size, "width": w, "height": h, "mean_intensity": intensity}
        )
    return pd.DataFrame(meta_stats)


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    print("Processing Data...")
    # Load and process data
    # We use the provided metadata paths and cache locations
    train_df = process_data(CFG.train_metadata_path, CFG.train_cache_path)
    val_df = process_data(CFG.val_metadata_path, CFG.val_cache_path)
    test_df = process_data(CFG.test_metadata_path, CFG.test_cache_path)

    # Prepare containers for Stacking
    # OOF Predictions: (N_train, 2) per model
    oof_preds_dict = {}
    # Hold-out Val Predictions: (N_val, 2) per model (averaged over folds)
    val_preds_dict = {}
    # Test Predictions: (N_test, 2) per model (averaged over folds)
    test_preds_dict = {}

    # Initialize dictionaries with zeros
    for model_name in CFG.backbones:
        oof_preds_dict[model_name] = np.zeros((len(train_df), 2))
        val_preds_dict[model_name] = np.zeros((len(val_df), 2))
        test_preds_dict[model_name] = np.zeros((len(test_df), 2))

    # 2. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    # We stratify based on the 'stratify_label' column created in metadata
    folds = list(skf.split(train_df, train_df["stratify_label"]))

    for model_name in CFG.backbones:
        img_size = CFG.img_sizes[model_name]
        print(f"\n=== Processing Model: {model_name} ===")

        # Temporary accumulators for averaging over folds
        val_fold_accum = np.zeros((len(val_df), 2))
        test_fold_accum = np.zeros((len(test_df), 2))

        for fold, (train_idx, valid_idx) in enumerate(folds):
            print(f"--- Fold {fold} ---")

            # Split data
            train_sub = train_df.iloc[train_idx].reset_index(drop=True)
            valid_sub = train_df.iloc[valid_idx].reset_index(drop=True)

            # Train
            # run_fold saves the best model to disk and returns the best score
            run_fold(fold, train_sub, valid_sub, model_name, img_size)

            # Load Best Model for Inference
            model = AppleNet(model_name=model_name, pretrained=False)
            safe_model_name = model_name.replace(".", "_")
            ckpt_path = os.path.join(
                CFG.output_dir, f"best_model_{safe_model_name}_fold_{fold}.pth"
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)
            model.eval()

            # 1. Predict OOF (Validation part of CV)
            # We use 'valid' transforms for inference
            valid_dataset = AppleDataset(
                valid_sub, transform=get_transforms("valid", img_size), mode="val"
            )
            oof_preds = predict_dataset(model, valid_dataset, CFG.batch_size, device)

            # Store OOF predictions in the global array using original indices
            oof_preds_dict[model_name][valid_idx] = oof_preds

            # 2. Predict Hold-out Validation
            holdout_dataset = AppleDataset(
                val_df, transform=get_transforms("valid", img_size), mode="val"
            )
            val_preds = predict_dataset(model, holdout_dataset, CFG.batch_size, device)
            val_fold_accum += val_preds

            # 3. Predict Test
            test_dataset = AppleDataset(
                test_df, transform=get_transforms("valid", img_size), mode="test"
            )
            test_preds = predict_dataset(model, test_dataset, CFG.batch_size, device)
            test_fold_accum += test_preds

            # Cleanup to save VRAM/RAM
            del model, valid_dataset, holdout_dataset, test_dataset
            torch.cuda.empty_cache()
            gc.collect()

        # Average predictions over folds
        val_preds_dict[model_name] = val_fold_accum / CFG.n_folds
        test_preds_dict[model_name] = test_fold_accum / CFG.n_folds

    # 3. Stacking
    print("\n=== Training Meta-Learner ===")
    # Prepare targets for meta-learner (from train_df)
    train_targets = train_df[["target_rust", "target_scab"]].values

    # Train Logistic Regression Meta-Learners
    meta_models = train_meta_learner(oof_preds_dict, train_targets)

    # 4. Final Validation & Metric
    print("\n=== Validating Ensemble ===")
    # Inference on Hold-out Val
    val_rust_probs, val_scab_probs = inference_meta_learner(meta_models, val_preds_dict)

    # Reconstruct 4-class probs for metric calculation
    val_probs_df = reconstruct_probs(val_rust_probs, val_scab_probs)
    val_preds_arr = val_probs_df[CFG.final_cols].values

    # Get Ground Truth for Hold-out Val
    # val_df has one-hot columns: healthy, multiple_diseases, rust, scab
    val_true_arr = val_df[CFG.final_cols].values

    # Compute Metric
    final_score = calculate_metric(val_true_arr, val_preds_arr)
    # Print strictly as requested
    print(f"Final Validation Metric: {final_score:.16f}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error per sample (Mean Absolute Error on the binary tasks)
    val_rust_true = val_df["target_rust"].values
    val_scab_true = val_df["target_scab"].values

    rust_err = np.abs(val_rust_true - val_rust_probs)
    scab_err = np.abs(val_scab_true - val_scab_probs)
    mean_err = (rust_err + scab_err) / 2.0

    # Extract metadata
    meta_df = get_image_metadata(val_df)
    meta_df["error"] = mean_err

    # Correlations
    features = ["file_size", "width", "height", "mean_intensity"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if feat in meta_df.columns and meta_df[feat].std() > 0:
            corr, _ = pearsonr(meta_df["error"], meta_df[feat])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: NaN (Constant or Missing)")

    # 6. Submission
    threshold = 0.9954104122251848
    if final_score > threshold:
        print("\n=== Generating Submission ===")
        test_rust_probs, test_scab_probs = inference_meta_learner(
            meta_models, test_preds_dict
        )
        create_submission(test_df, test_rust_probs, test_scab_probs)
    else:
        print(
            f"\nMetric ({final_score:.6f}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
