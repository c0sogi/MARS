import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import (
    get_data,
    DogDataset,
    get_train_transforms,
    get_valid_transforms,
)
from library.model import get_model
from library.engine import train_loop
from library.weight_averaging import average_checkpoints
from library.inference import run_inference, predict_with_tta

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline Execution
    # We reduce epochs to ensure the script completes quickly while still testing the pipeline.
    # A100 GPU is available, so 10 epochs on ~7k images will be very fast (~15 mins total).
    Config.EPOCHS = 10
    Config.SWA_EPOCHS = 3
    Config.PHASE1_EPOCHS = 1

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Load Training Metadata
    # We use the training set for 5-fold CV
    df_train_full, class_to_idx, idx_to_class = get_data(mode="train")

    # 3. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    print(f"Starting 5-Fold CV with {len(df_train_full)} samples...")

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["breed"])
    ):
        print(f"\n{'='*20} Fold {fold_idx} {'='*20}")

        # Prepare Data Splits
        df_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_full.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_dataset = DogDataset(
            df_train,
            class_to_idx=class_to_idx,
            transforms=get_train_transforms(Config.IMG_SIZE),
        )
        val_dataset = DogDataset(
            df_val,
            class_to_idx=class_to_idx,
            transforms=get_valid_transforms(Config.IMG_SIZE),
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = get_model(pretrained=True)
        model.to(device)

        # Optimizer (Initial setup for Phase 1)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.PHASE1_LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler (Cosine Annealing for Phase 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Execute Training Loop (Phase 1 + Phase 2 + SWA Checkpointing)
        train_loop(
            model, train_loader, val_loader, optimizer, scheduler, device, fold_idx
        )

        # Perform Manual Weight Averaging (Model Soup)
        print(f"Fold {fold_idx}: Averaging SWA checkpoints...")
        ckpt_paths = []
        start_swa = Config.EPOCHS - Config.SWA_EPOCHS
        for e in range(start_swa, Config.EPOCHS):
            p = os.path.join(Config.WORKING_DIR, f"swa_fold_{fold_idx}_epoch_{e}.pth")
            if os.path.exists(p):
                ckpt_paths.append(p)

        avg_output_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")

        if ckpt_paths:
            average_checkpoints(ckpt_paths, avg_output_path)
        else:
            # Fallback to best model if SWA checkpoints are missing
            print("Warning: No SWA checkpoints found. Fallback to best model.")
            best_p = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold_idx}.pth")
            if os.path.exists(best_p):
                state = torch.load(best_p)
                torch.save(state, avg_output_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Validation Assessment on Hold-out Set
    print("\n" + "=" * 40)
    print("VALIDATION ASSESSMENT")
    print("=" * 40)

    # Load the official hold-out validation set
    df_holdout, _, _ = get_data(mode="val")

    holdout_dataset = DogDataset(
        df_holdout,
        class_to_idx=class_to_idx,
        transforms=get_valid_transforms(Config.IMG_SIZE),
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate Ensemble Predictions
    ensemble_probs = None
    models_used = 0

    # Ground Truth
    y_true = np.array([class_to_idx[b] for b in df_holdout["breed"]])

    for fold_idx in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            continue

        # Load Model
        model = get_model(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Predict with Test-Time Augmentation
        probs, _ = predict_with_tta(model, holdout_loader, device)

        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs

        models_used += 1
        del model
        torch.cuda.empty_cache()

    if models_used == 0:
        print("Error: No models available for validation.")
        return

    # Average probabilities
    ensemble_probs /= models_used

    # Calculate Metric
    final_metric = log_loss(
        y_true, ensemble_probs.numpy(), labels=list(range(Config.NUM_CLASSES))
    )
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate per-sample loss (Cross Entropy)
    probs_np = ensemble_probs.numpy()
    epsilon = 1e-15
    probs_np = np.clip(probs_np, epsilon, 1 - epsilon)

    sample_losses = []
    for i, true_idx in enumerate(y_true):
        p = probs_np[i, true_idx]
        sample_losses.append(-np.log(p))

    sample_losses = np.array(sample_losses)

    # Extract Metadata Features for Correlation
    widths, heights, aspect_ratios, file_sizes = [], [], [], []

    for path_rel in df_holdout["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, path_rel)
        try:
            size = os.path.getsize(full_path)
            with Image.open(full_path) as img:
                w, h = img.size

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            file_sizes.append(size)
        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            file_sizes.append(0)

    # Compute Correlations
    df_analysis = pd.DataFrame(
        {
            "loss": sample_losses,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "file_size": file_sizes,
        }
    )

    print("Correlation between Error Magnitude (Loss) and Input Features:")
    for col in ["width", "height", "aspect_ratio", "file_size"]:
        if df_analysis[col].std() > 0:
            corr, _ = pearsonr(df_analysis["loss"], df_analysis[col])
            print(f"  {col}: {corr:.4f}")
        else:
            print(f"  {col}: NaN (No variance)")

    # 6. Submission Generation
    THRESHOLD = 0.14004325100369866

    if final_metric < THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        run_inference()
    else:
        print(
            f"\nValidation Metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Submission skipped.")


if __name__ == "__main__":
    main()
