import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import cv2
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders
from library.model import DRModel
from library.engine import train_one_epoch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_fold(fold, train_df, val_df):
    """
    Trains a model for a single fold using Progressive Resizing and SWA.
    """
    print(f"\n=== Running Fold {fold} ===")
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DRModel(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        gem_p=Config.GEM_P,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # --- PHASE 1: Coarse Training (Lower Resolution) ---
    print(f"--- Phase 1: Res {Config.PHASE_1_RES}, Epochs {Config.PHASE_1_EPOCHS} ---")

    # Create loaders for Phase 1
    # We pass an empty DataFrame for test_df as we don't need it here
    train_loader, _, _ = get_dataloaders(
        train_df,
        val_df,
        pd.DataFrame(),
        image_size=Config.PHASE_1_RES,
        batch_size=Config.PHASE_1_BATCH_SIZE,
    )

    # Scheduler for Phase 1
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.PHASE_1_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    model.train()
    for epoch in range(Config.PHASE_1_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        # Optional: Print progress every few epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == Config.PHASE_1_EPOCHS:
            print(
                f"[Fold {fold} - P1] Epoch {epoch+1}/{Config.PHASE_1_EPOCHS} - Loss: {loss:.4f}"
            )

    # --- PHASE 2: Fine-tuning with SWA (Higher Resolution) ---
    print(
        f"--- Phase 2: Res {Config.PHASE_2_RES}, Epochs {Config.PHASE_2_EPOCHS} (SWA) ---"
    )

    # Create loaders for Phase 2
    train_loader, _, _ = get_dataloaders(
        train_df,
        val_df,
        pd.DataFrame(),
        image_size=Config.PHASE_2_RES,
        batch_size=Config.PHASE_2_BATCH_SIZE,
    )

    # Initialize SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    for epoch in range(Config.PHASE_2_EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, device, None)
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(
            f"[Fold {fold} - P2] Epoch {epoch+1}/{Config.PHASE_2_EPOCHS} - Loss: {loss:.4f}"
        )

    # Update BN statistics for SWA model
    print("Updating SWA BatchNorm statistics...")
    update_bn(train_loader, swa_model, device=device)

    # Save SWA model
    save_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth")
    torch.save(swa_model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

    # Clean up to free GPU memory
    del model, swa_model, optimizer, scheduler, swa_scheduler, train_loader
    torch.cuda.empty_cache()

    return save_path


def predict_ensemble(models, df, image_size, device):
    """
    Performs inference using an ensemble of models with TTA.
    """
    # Create a loader for the provided dataframe
    # We use get_dataloaders' test_loader logic (3rd return value)
    # We pass empty train/val dfs
    _, _, loader = get_dataloaders(
        pd.DataFrame(),
        pd.DataFrame(),
        df,
        image_size=image_size,
        batch_size=Config.BATCH_SIZE,
    )

    all_preds = []

    # Ensure models are in eval mode
    for model in models:
        model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_preds = []

            # 1. Standard Forward Pass
            for model in models:
                out = model(images)
                batch_preds.append(out.view(-1).cpu().numpy())

            # 2. Test Time Augmentation (Horizontal Flip)
            if Config.USE_TTA:
                images_flip = torch.flip(images, [3])
                for model in models:
                    out = model(images_flip)
                    batch_preds.append(out.view(-1).cpu().numpy())

            # Average predictions for this batch across all models and TTA views
            avg_preds = np.mean(batch_preds, axis=0)
            all_preds.append(avg_preds)

    return np.concatenate(all_preds)


def failure_analysis(df, preds, targets):
    """
    Analyzes correlation between error magnitude and image meta-features.
    """
    print("\n=== Failure Analysis ===")
    errors = np.abs(targets - preds)

    stats = {
        "width": [],
        "height": [],
        "file_size": [],
        "mean_intensity": [],
        "std_intensity": [],
    }

    print("Calculating meta-features for validation set...")
    for idx, row in df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            size = os.path.getsize(path)
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Image not found")

            h, w = img.shape[:2]
            mean_int = img.mean()
            std_int = img.std()

            stats["file_size"].append(size)
            stats["width"].append(w)
            stats["height"].append(h)
            stats["mean_intensity"].append(mean_int)
            stats["std_intensity"].append(std_int)
        except Exception as e:
            # Fallback for errors
            stats["file_size"].append(0)
            stats["width"].append(0)
            stats["height"].append(0)
            stats["mean_intensity"].append(0)
            stats["std_intensity"].append(0)

    # Calculate correlations
    print("Correlations between Error Magnitude and Meta-features:")
    for key in stats:
        if len(stats[key]) > 0:
            corr, _ = spearmanr(errors, stats[key])
            print(f"Correlation Error vs {key}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 2. Load Metadata
    # train.csv: Used for 5-fold CV training
    # val.csv: Hold-out set for final validation
    # test.csv: Target for submission
    train_metadata = pd.read_csv(Config.TRAIN_CSV)
    holdout_metadata = pd.read_csv(Config.VAL_CSV)
    test_metadata = pd.read_csv(Config.TEST_CSV)

    print(f"Training Data: {len(train_metadata)} samples")
    print(f"Hold-out Validation Data: {len(holdout_metadata)} samples")
    print(f"Test Data: {len(test_metadata)} samples")

    # 3. Cross-Validation Training
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    model_paths = []

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_metadata, train_metadata["diagnosis"])
    ):
        fold_train = train_metadata.iloc[train_idx].reset_index(drop=True)
        fold_val = train_metadata.iloc[val_idx].reset_index(drop=True)

        # Train model for this fold
        path = run_fold(fold, fold_train, fold_val)
        model_paths.append(path)

    # 4. Load Models for Inference
    print("\nLoading models for ensemble...")
    models = []
    for path in model_paths:
        # Re-instantiate model structure
        model = DRModel(
            backbone_name=Config.BACKBONE,
            pretrained=False,
            num_classes=Config.NUM_CLASSES,
            gem_p=Config.GEM_P,
        )

        # Load weights
        state_dict = torch.load(path, map_location=device)

        # Handle SWA state_dict keys (remove 'module.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            elif k == "n_averaged":
                continue
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    # 5. Validation on Hold-out Set
    print("\n=== Final Validation on Hold-out Set ===")
    val_preds = predict_ensemble(models, holdout_metadata, Config.PHASE_2_RES, device)
    val_targets = holdout_metadata["diagnosis"].values

    # Calculate Metric
    qwk = quadratic_weighted_kappa(val_targets, val_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {qwk}")

    # 6. Failure Analysis
    failure_analysis(holdout_metadata, val_preds, val_targets)

    # 7. Submission Generation
    threshold = 0.9207435978935975

    if qwk > threshold:
        print(
            f"\nMetric ({qwk:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = predict_ensemble(models, test_metadata, Config.PHASE_2_RES, device)

        # Post-processing: Round to nearest integer and clip
        final_preds = np.rint(test_preds).astype(int)
        final_preds = np.clip(final_preds, 0, 4)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"id_code": test_metadata["id_code"], "diagnosis": final_preds}
        )

        # Save
        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
        print("Sample head:")
        print(submission.head())
    else:
        print(
            f"\nMetric ({qwk:.6f}) <= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
