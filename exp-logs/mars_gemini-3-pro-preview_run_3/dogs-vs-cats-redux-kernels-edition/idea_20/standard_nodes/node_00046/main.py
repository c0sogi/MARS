import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import (
    load_train_metadata,
    load_test_metadata,
    DogCatDataset,
    get_transforms,
)
from library.models import get_model
from library.engine import train_model, predict, evaluate
from library.ensemble import find_optimal_weights, weighted_average


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # --- Fast Baseline Configuration ---
    # We override some Config defaults to ensure execution within the 1-hour limit.
    # 5000 samples * 5 folds * 3 models * 2 epochs fits comfortably in the timeframe.
    NUM_EPOCHS = 2
    SUBSET_SIZE = 5000
    BATCH_SIZE = 32

    print("Initializing Fast Baseline Run...")

    # 2. Data Loading
    # Load combined metadata (Train + Val) for Cross-Validation
    full_df = load_train_metadata(load_cached_data=True)

    # Subsample dataset for speed
    if len(full_df) > SUBSET_SIZE:
        print(f"Subsampling dataset from {len(full_df)} to {SUBSET_SIZE} samples.")
        # Stratified subsampling to maintain class balance
        _, subset_df = train_test_split(
            full_df,
            test_size=SUBSET_SIZE,
            stratify=full_df["label"],
            random_state=Config.SEED,
        )
        full_df = subset_df.reset_index(drop=True)

    # Load Test Metadata
    test_df = load_test_metadata(load_cached_data=True)

    # Initialize storage for predictions
    # OOF: Store predictions for every sample in the training set
    model_names = list(Config.MODEL_CONFIGS.keys())
    oof_preds_dict = {name: np.zeros(len(full_df)) for name in model_names}

    # Test: Accumulate predictions across folds to average later
    test_preds_accumulator = {name: np.zeros(len(test_df)) for name in model_names}

    # 3. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    print(
        f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation on {len(full_df)} samples..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        print(f"\n=== Fold {fold + 1}/{Config.NUM_FOLDS} ===")

        # Create Fold DataFrames
        train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Iterate over each architecture in the Heterogeneous Ensemble
        for model_key, model_cfg in Config.MODEL_CONFIGS.items():
            print(f"  Training {model_key}...")

            img_size = model_cfg["img_size"]

            # Datasets & Loaders
            train_dataset = DogCatDataset(
                train_fold_df, transform=get_transforms(img_size, "train"), mode="train"
            )
            val_dataset = DogCatDataset(
                val_fold_df, transform=get_transforms(img_size, "val"), mode="train"
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model Setup
            model = get_model(model_key, pretrained=True, num_classes=1)
            model = model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS
            )

            # Training
            model, _ = train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                NUM_EPOCHS,
                patience=1,
            )

            # OOF Inference (Validation)
            _, val_preds, _ = evaluate(model, val_loader, device)

            # Store OOF predictions (map back to original indices)
            oof_preds_dict[model_key][val_idx] = val_preds.flatten()

            # Test Inference (TTA) for this fold
            test_dataset = DogCatDataset(
                test_df, transform=get_transforms(img_size, "val"), mode="test"
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            _, fold_test_preds = predict(model, test_loader, device)

            # Accumulate (Average later)
            test_preds_accumulator[model_key] += fold_test_preds

            # Cleanup to save memory
            del model, optimizer, scheduler, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

    # Average Test Predictions across folds
    for name in model_names:
        test_preds_accumulator[name] /= Config.NUM_FOLDS

    # 4. Ensemble Optimization (OOF-Optimized Weighting)
    print("\nOptimizing Ensemble Weights...")
    true_labels = full_df["label"].values

    # Quality Gating: Filter out models that failed to converge
    valid_model_keys = []
    prediction_list = []

    for key in model_names:
        loss = log_loss(true_labels, oof_preds_dict[key])
        print(f"  Model {key} OOF Log Loss: {loss:.5f}")

        if loss < Config.OOF_THRESHOLD:
            valid_model_keys.append(key)
            prediction_list.append(oof_preds_dict[key])
        else:
            print(
                f"  [WARNING] Dropping {key} due to high loss (Poison Pill detection)."
            )

    if not valid_model_keys:
        raise RuntimeError("All models failed quality gating. Training failed.")

    # Find Optimal Weights using SLSQP
    optimal_weights = find_optimal_weights(prediction_list, true_labels)
    weight_dict = dict(zip(valid_model_keys, optimal_weights))
    print(f"  Optimal Weights: {weight_dict}")

    # 5. Validation Reporting
    # We must report the metric on the hold-out validation set (val.csv).
    # Since we used 5-Fold CV on the combined data, we extract the OOF predictions
    # corresponding to the original validation images.

    # Load original validation metadata to identify hold-out samples
    val_metadata_orig = pd.read_csv(Config.VAL_METADATA_PATH)
    val_filepaths = set(val_metadata_orig["filepath"].values)

    # Create mask for hold-out set
    is_holdout = full_df["filepath"].isin(val_filepaths)

    # Compute Weighted OOF Predictions
    final_oof_preds = weighted_average(prediction_list, optimal_weights)

    # Extract hold-out subset
    holdout_preds = final_oof_preds[is_holdout]
    holdout_labels = full_df.loc[is_holdout, "label"].values

    if len(holdout_labels) == 0:
        # Fallback if subsampling accidentally excluded all val images (unlikely)
        final_metric = log_loss(true_labels, final_oof_preds)
        print(
            "Warning: No original validation images in subsample. Using full OOF metric."
        )
    else:
        final_metric = log_loss(holdout_labels, holdout_preds)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(holdout_labels - holdout_preds)

    # Analyze correlation with File Size
    # We need to retrieve file sizes for the holdout set
    holdout_df = full_df[is_holdout].copy()
    holdout_df["error"] = errors

    file_sizes = []
    for fp in holdout_df["filepath"]:
        full_path = os.path.join(Config.INPUT_DIR, fp)
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    holdout_df["file_size"] = file_sizes

    if len(holdout_df) > 1:
        corr, pval = pearsonr(holdout_df["error"], holdout_df["file_size"])
        print(f"Correlation between Error and File Size: {corr:.10f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 7. Submission
    # Threshold check as per instructions
    THRESHOLD = 0.009074434935821756

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is lower than threshold ({THRESHOLD}). Generating submission..."
        )

        # Gather test predictions for valid models
        test_pred_list = [test_preds_accumulator[key] for key in valid_model_keys]

        # Apply optimal weights
        final_test_preds = weighted_average(test_pred_list, optimal_weights)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_df["id"], "label": final_test_preds})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
