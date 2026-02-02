import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from library
from library.config import Config
from library.dataset import PathologyDataset, get_transforms
from library.model import TumorClassifier
from library.engine import train_loop, validate
from library.utils import seed_everything, calculate_auc

# --- Configuration Overrides for Fast Baseline ---
# We override some config parameters to ensure the code runs within the time limit
# while maintaining high performance.
Config.epochs = (
    10  # Reduced to ensure < 2 hours runtime on A100 while allowing convergence
)
Config.batch_size = 256
Config.num_workers = 12


def override_config():
    """Applies runtime configurations."""
    print(f"Configuration:")
    print(f"  Device: {Config.device}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch Size: {Config.batch_size}")
    print(f"  Image Size: {Config.image_size} -> Crop: {Config.crop_size}")


def get_optimizer_and_scheduler(model, samples_per_epoch):
    """Creates optimizer and scheduler."""
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    return optimizer, scheduler


def run_fold(fold, train_df, val_df):
    """
    Executes the training pipeline for a single fold.
    """
    print(f"\n=== Starting Fold {fold} ===")

    # Datasets
    train_dataset = PathologyDataset(train_df, transforms=get_transforms(data="train"))
    val_dataset = PathologyDataset(val_df, transforms=get_transforms(data="valid"))

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model
    model = TumorClassifier(pretrained=True)
    model.to(Config.device)

    # Optimization
    optimizer, scheduler = get_optimizer_and_scheduler(model, len(train_loader))

    # Train
    best_auc = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        fold=fold,
        epochs=Config.epochs,
    )

    # Clear memory
    del (
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    )
    torch.cuda.empty_cache()

    return best_auc


def apply_tta(images, model):
    """
    Applies 8-view TTA (4 rotations x 2 flips) and averages predictions.
    Args:
        images: Tensor (B, C, H, W)
        model: PyTorch model
    Returns:
        probs: Tensor (B, 1)
    """
    probs_list = []

    # Standard: 0, 90, 180, 270
    # Flips: Horizontal
    # Combinations cover the D4 group (8 symmetries of a square)

    for k in [0, 1, 2, 3]:
        # Rotate
        if k == 0:
            rot_imgs = images
        else:
            rot_imgs = torch.rot90(images, k, [2, 3])

        # 1. Forward Pass (Original Rotation)
        logits = model(rot_imgs)
        probs_list.append(torch.sigmoid(logits))

        # 2. Forward Pass (Horizontal Flip of Rotation)
        flip_imgs = torch.flip(rot_imgs, [3])
        logits_flip = model(flip_imgs)
        probs_list.append(torch.sigmoid(logits_flip))

    # Stack and Mean
    probs_stack = torch.stack(probs_list, dim=0)  # (8, B, 1)
    avg_probs = torch.mean(probs_stack, dim=0)  # (B, 1)

    return avg_probs


def inference_ensemble(models, df, device):
    """
    Performs inference using an ensemble of models with TTA.
    """
    dataset = PathologyDataset(df, transforms=get_transforms(data="test"))
    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    all_preds = []

    # Set models to eval
    for model in models:
        model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Ensemble averaging
            batch_preds = []
            for model in models:
                # Get TTA predictions for this model
                probs = apply_tta(images, model)
                batch_preds.append(probs)

            # Average across models
            batch_preds_stack = torch.stack(batch_preds, dim=0)  # (N_models, B, 1)
            ensemble_probs = torch.mean(batch_preds_stack, dim=0)  # (B, 1)

            all_preds.extend(ensemble_probs.cpu().numpy().flatten().tolist())

    return np.array(all_preds)


def perform_failure_analysis(df_val, preds):
    """
    Analyzes correlation between error magnitude and image statistics.
    """
    print("\n=== Failure Analysis ===")

    targets = df_val["label"].values
    errors = np.abs(targets - preds)

    brightness = []
    contrast = []

    print("Calculating image statistics for failure analysis...")

    # Pre-construct paths
    paths = [os.path.join(Config.input_dir, p) for p in df_val["file_path"].values]

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            # Fallback for safety, though metadata is verified
            brightness.append(0.5)
            contrast.append(0.0)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Simple stats
        b = img.mean()
        c = img.std()

        brightness.append(b)
        contrast.append(c)

    brightness = np.array(brightness)
    contrast = np.array(contrast)

    # Correlations
    corr_b, _ = pearsonr(errors, brightness)
    corr_c, _ = pearsonr(errors, contrast)

    print(f"Correlation between Error and Brightness: {corr_b:.16f}")
    print(f"Correlation between Error and Contrast:   {corr_c:.16f}")

    # Interpretation
    if abs(corr_b) > 0.1:
        print(f"  -> Significant relationship with brightness.")
    if abs(corr_c) > 0.1:
        print(f"  -> Significant relationship with contrast.")


def main():
    seed_everything(Config.seed)
    override_config()
    Config.setup()

    # 1. Load Metadata
    print("Loading metadata...")
    df_train_full = pd.read_csv(Config.train_metadata_path)
    df_holdout_val = pd.read_csv(Config.val_metadata_path)

    # 2. Train Ensemble (5 Folds)
    # We split the available 'train' set into 5 folds for training the ensemble members.

    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    trained_models = []

    # Create fold column
    df_train_full["fold"] = -1
    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        df_train_full.loc[val_idx, "fold"] = fold_id

    for fold in range(Config.n_folds):
        # Prepare dataframes
        train_df = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
        val_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

        # Run training
        run_fold(fold, train_df, val_df)

        # Load best model for this fold
        model = TumorClassifier(pretrained=False)  # Architecture only
        ckpt_path = os.path.join(Config.checkpoint_dir, f"best_model_fold_{fold}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.device))
        model.to(Config.device)
        trained_models.append(model)

    # 3. Final Validation on Hold-out Set
    print("\n=== Final Validation on Hold-out Set ===")
    val_preds = inference_ensemble(trained_models, df_holdout_val, Config.device)
    val_targets = df_holdout_val["label"].values

    final_auc = calculate_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc:.16f}")

    # 4. Failure Analysis
    perform_failure_analysis(df_holdout_val, val_preds)

    # 5. Submission
    # Threshold check
    THRESHOLD = 0.9889066475479729

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        df_test = pd.read_csv(Config.test_metadata_path)
        test_preds = inference_ensemble(trained_models, df_test, Config.device)

        # Create submission DataFrame
        submission = pd.DataFrame({"id": df_test["id"], "label": test_preds})

        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nValidation AUC ({final_auc:.6f}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
