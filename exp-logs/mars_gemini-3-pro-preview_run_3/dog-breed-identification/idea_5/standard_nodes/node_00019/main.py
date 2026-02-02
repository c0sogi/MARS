import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data_and_classes, get_transforms, DogDataset
from library.model import DogModel, ModelEMA
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print(f"Using device: {device}")

    # 2. Data Loading
    # We load the full training data (train+val from metadata) to perform 5-Fold CV
    full_train_df, test_df, class_to_idx, classes = get_data_and_classes(
        load_cached_data=True
    )

    # 3. Stratified K-Fold Training
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Placeholder to store OOF scores or just track fold paths
    model_paths = []

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["breed"])
    ):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Split Data
        train_df_fold = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = full_train_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = DogDataset(
            train_df_fold,
            class_to_idx=class_to_idx,
            transform=get_transforms("train", Config.IMG_SIZE),
            mode="train",
        )
        val_dataset = DogDataset(
            val_df_fold,
            class_to_idx=class_to_idx,
            transform=get_transforms("val", Config.IMG_SIZE),
            mode="val",
        )

        # Create DataLoaders
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
        model = DogModel(num_classes=len(classes), pretrained=True)
        model.to(device)

        # Initialize EMA
        model_ema = ModelEMA(model, device=device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Cosine Annealing Scheduler
        # T_max is set to total epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        best_loss = float("inf")
        save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        model_paths.append(save_path)

        # Training Loop
        for epoch in range(1, Config.EPOCHS + 1):
            # Phase 1: Freeze Backbone
            if epoch <= Config.FREEZE_EPOCHS:
                model.freeze_backbone()
                print(f"Epoch {epoch}: Backbone Frozen")
            else:
                model.unfreeze_all()

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, epoch, model_ema
            )

            # Step Scheduler
            scheduler.step()

            # Validate (using EMA model for stability)
            val_metrics = valid_one_epoch(model_ema.module, val_loader, device, epoch)

            # Save Best Model
            if val_metrics["Loss"] < best_loss:
                best_loss = val_metrics["Loss"]
                torch.save(model_ema.module.state_dict(), save_path)
                print(f"Saved Best Model for Fold {fold+1} with Loss: {best_loss:.4f}")

        # Cleanup to free memory
        del (
            model,
            model_ema,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Final Validation on Hold-Out Set
    print(f"\n{'='*20} Final Evaluation on Hold-Out Set {'='*20}")

    # Load specific hold-out set
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    holdout_df = pd.read_csv(val_csv_path)

    holdout_dataset = DogDataset(
        holdout_df,
        class_to_idx=class_to_idx,
        transform=get_transforms("val", Config.IMG_SIZE),
        mode="train",  # 'train' mode returns label, which we need for metric calculation, but we won't use it in inference_fn
    )

    # For inference_fn, we need a loader that returns (image, id).
    # DogDataset in 'test' mode returns (image, id).
    # But we need to align predictions with ground truth.
    # Let's create a dataset in 'test' mode to get IDs, and we will map IDs to labels using the dataframe.

    holdout_eval_dataset = DogDataset(
        holdout_df,
        class_to_idx=class_to_idx,
        transform=get_transforms("val", Config.IMG_SIZE),
        mode="test",
    )

    holdout_loader = DataLoader(
        holdout_eval_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Prediction
    ensemble_preds = np.zeros((len(holdout_df), len(classes)))

    for fold_idx, path in enumerate(model_paths):
        print(f"Predicting with model fold {fold_idx+1}...")
        model = DogModel(num_classes=len(classes), pretrained=False)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        preds, ids = inference_fn(model, holdout_loader, device)

        # Align predictions with dataframe order just in case, though loader is sequential
        # The loader returns ids in order. We can double check or just assume sequential.
        # Since shuffle=False, it matches holdout_df order.
        ensemble_preds += preds

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Average predictions
    ensemble_preds /= Config.N_FOLDS

    # Get Ground Truth
    # Map breed strings to indices
    y_true = holdout_df["breed"].map(class_to_idx).values

    # Calculate Metric
    final_metric = log_loss(y_true, ensemble_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")

    # Calculate loss per sample
    # Gather probability assigned to the true class
    # Clip to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(ensemble_preds, epsilon, 1 - epsilon)
    prob_true = preds_clipped[np.arange(len(y_true)), y_true]
    sample_losses = -np.log(prob_true)

    # Get File Sizes
    file_sizes = []
    for rel_path in holdout_df["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except:
            file_sizes.append(0)

    # Calculate Correlation
    if len(file_sizes) == len(sample_losses):
        corr, p_val = pearsonr(sample_losses, file_sizes)
        print(f"Correlation between Error (Log Loss) and File Size: {corr:.4f}")
    else:
        print("Could not calculate correlation due to length mismatch.")

    # 6. Submission
    THRESHOLD = 0.14004325100369866

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_dataset = DogDataset(
            test_df,
            class_to_idx=None,  # Not needed for test
            transform=get_transforms("val", Config.IMG_SIZE),
            mode="test",
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ensemble_preds = np.zeros((len(test_df), len(classes)))
        test_ids_list = []

        for fold_idx, path in enumerate(model_paths):
            print(f"Inference on Test Set with model fold {fold_idx+1}...")
            model = DogModel(num_classes=len(classes), pretrained=False)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            preds, ids = inference_fn(model, test_loader, device)
            test_ensemble_preds += preds

            if fold_idx == 0:
                test_ids_list = ids

            del model
            torch.cuda.empty_cache()
            gc.collect()

        test_ensemble_preds /= Config.N_FOLDS

        # Create Submission DataFrame
        sub_df = pd.DataFrame(test_ensemble_preds, columns=classes)
        sub_df.insert(0, "id", test_ids_list)

        # Save
        submission_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric {final_metric} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
