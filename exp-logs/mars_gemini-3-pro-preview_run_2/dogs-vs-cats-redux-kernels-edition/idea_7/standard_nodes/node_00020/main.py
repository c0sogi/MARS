import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.engine import fit
from library.inference import run_inference


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    seed_everything(Config.SEED)

    # Adjust Epochs to ensure completion within 2 hours
    # 10 models * 10 epochs * ~1 min/epoch = ~100 mins < 120 mins
    Config.EPOCHS = 10

    print(f"Starting execution on {Config.DEVICE}")
    print(f"Models: {Config.MODELS}")
    print(f"Folds: {Config.N_FOLDS}, Epochs: {Config.EPOCHS}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    holdout_val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # ---------------------------------------------------------
    # 3. Training Loop (Heterogeneous Ensemble)
    # ---------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for model_name in Config.MODELS:
        print(f"\n=== Training Architecture: {model_name} ===")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["label"])
        ):
            print(f"--- Fold {fold} ---")

            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold}.pth"
            )

            # Prepare Data
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            train_ds = DogCatDataset(
                fold_train_df, transforms=get_transforms("train"), mode="train"
            )
            val_ds = DogCatDataset(
                fold_val_df, transforms=get_transforms("val"), mode="val"
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model, Optimizer, Scheduler
            model = get_model(model_name, pretrained=True, num_classes=1)
            model.to(Config.DEVICE)

            optimizer = AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Train
            best_loss = fit(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                Config.DEVICE,
                checkpoint_path,
            )
            print(f"Fold {fold} Best Loss: {best_loss:.6f}")

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 4. Global Validation
    # ---------------------------------------------------------
    print("\n=== Running Global Validation ===")

    val_ds = DogCatDataset(holdout_val_df, transforms=get_transforms("val"), mode="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []

    for model_name in Config.MODELS:
        for fold in range(Config.N_FOLDS):
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold}.pth"
            )

            model = get_model(model_name, pretrained=False, num_classes=1)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=Config.DEVICE)
            )
            model.to(Config.DEVICE)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(Config.DEVICE)

                    # Original
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).squeeze(1)

                    # TTA
                    if Config.TTA_FLIP:
                        outputs_flip = model(torch.flip(images, dims=[3]))
                        probs_flip = torch.sigmoid(outputs_flip).squeeze(1)
                        probs = (probs + probs_flip) / 2.0

                    fold_preds.append(probs.cpu().numpy())

            all_preds.append(np.concatenate(fold_preds))
            del model
            torch.cuda.empty_cache()

    # Average predictions across all models (Ensemble)
    avg_preds = np.mean(all_preds, axis=0)
    y_true = holdout_val_df["label"].values
    final_metric = log_loss(y_true, avg_preds)

    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - avg_preds)

    widths, heights, ars = [], [], []
    for filepath in holdout_val_df["filepath"]:
        img = cv2.imread(os.path.join(Config.INPUT_DIR, filepath))
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            ars.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            ars.append(0)

    df_err = pd.DataFrame({"err": errors, "w": widths, "h": heights, "ar": ars})
    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Width: {df_err['err'].corr(df_err['w']):.4f}")
    print(f"  Height: {df_err['err'].corr(df_err['h']):.4f}")
    print(f"  Aspect Ratio: {df_err['err'].corr(df_err['ar']):.4f}")

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.018199009307556684
    if final_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")
        run_inference()
    else:
        print("Metric check failed. Skipping submission.")


if __name__ == "__main__":
    main()
