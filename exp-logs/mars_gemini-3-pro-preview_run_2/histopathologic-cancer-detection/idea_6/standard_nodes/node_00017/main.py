import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import pearsonr
import cv2

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import load_data, PathologyDataset, get_transforms
from library.model import get_model
from library.engine import train_fold, inference_fn


def main():
    # 1. Setup
    print("Setting up environment...")
    Config.setup()
    seed_everything(Config.SEED)

    # --- Fast Baseline Configuration Overrides ---
    # We override these to ensure the run completes within the time limit
    # while still performing the required 5-fold cross-validation.
    Config.EPOCHS = 3
    Config.N_FOLDS = 5
    MAX_TRAIN_SAMPLES = 5000  # Limit training data for speed

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("\nLoading Data...")
    # Load Train Data
    # load_data handles caching automatically
    full_train_images, full_train_labels, full_train_ids = load_data(
        "train", load_cached_data=True
    )

    # Subsample training data for fast baseline
    if len(full_train_labels) > MAX_TRAIN_SAMPLES:
        print(
            f"Subsampling training data from {len(full_train_labels)} to {MAX_TRAIN_SAMPLES}..."
        )
        # Use stratified split to maintain class balance
        subset_indices, _ = train_test_split(
            np.arange(len(full_train_labels)),
            train_size=MAX_TRAIN_SAMPLES,
            stratify=full_train_labels,
            random_state=Config.SEED,
        )
        train_images = full_train_images[subset_indices]
        train_labels = full_train_labels[subset_indices]
        # train_ids = full_train_ids[subset_indices] # Not strictly needed for training
    else:
        train_images = full_train_images
        train_labels = full_train_labels

    # Load Hold-out Validation Data (Full)
    val_images, val_labels, val_ids = load_data("val", load_cached_data=True)

    # 3. Training Loop (5-Fold)
    print("\nStarting 5-Fold Cross-Validation...")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        # Split for this fold
        X_train_fold = train_images[train_idx]
        y_train_fold = train_labels[train_idx]
        X_val_fold = train_images[val_idx]
        y_val_fold = train_labels[val_idx]

        # Create Datasets
        train_ds = PathologyDataset(
            X_train_fold, y_train_fold, transforms=get_transforms("train")
        )
        val_ds = PathologyDataset(
            X_val_fold, y_val_fold, transforms=get_transforms("val")
        )

        # Create Loaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = get_model(pretrained=True).to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Train Fold
        best_auc = train_fold(
            fold,
            train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            device,
            patience=3,
        )

        # Load Best Model Weights
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        fold_models.append(model)

        # Cleanup to save memory
        del (
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            train_ds,
            val_ds,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
        )
        torch.cuda.empty_cache()

    # 4. Global Validation
    print("\nPerforming Global Validation on Hold-out Set...")

    # Create Global Val Loader
    global_val_ds = PathologyDataset(
        val_images, val_labels, transforms=get_transforms("val")
    )
    global_val_loader = torch.utils.data.DataLoader(
        global_val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference
    val_preds_accum = []
    for i, model in enumerate(fold_models):
        print(f"Inference with model fold {i}...")
        preds = inference_fn(model, global_val_loader, device)
        val_preds_accum.append(preds)

    # Average predictions
    avg_val_preds = np.mean(val_preds_accum, axis=0)

    # Calculate Metric
    final_val_auc = calculate_roc_auc(val_labels, avg_val_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_labels - avg_val_preds)

    # Calculate features for validation images
    # Normalize to 0-1
    imgs_norm = val_images.astype(np.float32) / 255.0

    # Compute stats
    brightness = imgs_norm.mean(axis=(1, 2, 3))
    contrast = imgs_norm.std(axis=(1, 2, 3))
    red_mean = imgs_norm[..., 0].mean(axis=(1, 2))
    green_mean = imgs_norm[..., 1].mean(axis=(1, 2))
    blue_mean = imgs_norm[..., 2].mean(axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Mean": red_mean,
        "Green Mean": green_mean,
        "Blue Mean": blue_mean,
    }

    print("Correlation between Error Magnitude and Image Features:")
    for name, feat_values in features.items():
        corr, p_val = pearsonr(errors, feat_values)
        print(f"  {name}: Correlation = {corr:.4f} (p-value = {p_val:.4f})")

    # 6. Submission
    THRESHOLD = 0.9889066475479729

    if final_val_auc > THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")

        # Load Test Data
        test_images, test_labels, test_ids = load_data("test", load_cached_data=True)

        test_ds = PathologyDataset(
            test_images, labels=None, transforms=get_transforms("test")
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds_accum = []
        for i, model in enumerate(fold_models):
            print(f"Inference on test set with model fold {i}...")
            preds = inference_fn(model, test_loader, device)
            test_preds_accum.append(preds)

        avg_test_preds = np.mean(test_preds_accum, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "label": avg_test_preds})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({final_val_auc}) is below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
