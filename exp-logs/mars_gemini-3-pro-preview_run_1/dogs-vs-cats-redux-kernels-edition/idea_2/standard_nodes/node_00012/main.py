import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy import stats
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from library
from library.config import Config
from library.utils import seed_everything, save_checkpoint, save_submission
from library.dataset import DogCatDataset, get_transforms
from library.model import DogCatClassifier, train_one_epoch, validate, predict_test


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"Running on device: {device}")

    # Hyperparameters for Fast Baseline
    # Reduced epochs to ensure execution within time limits while maintaining performance
    EPOCHS = Config.epochs
    BATCH_SIZE = Config.batch_size
    N_FOLDS = Config.n_folds

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    df_train_part = pd.read_csv(Config.train_metadata_path)
    df_val_part = pd.read_csv(Config.val_metadata_path)
    # Combine train and validation sets for Cross-Validation
    df_train_full = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)

    df_test = pd.read_csv(Config.test_metadata_path)

    # Test Loader (Fixed)
    test_dataset = DogCatDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Loop
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=Config.seed)

    # Storage for OOF predictions and Test accumulations
    oof_preds = np.zeros(len(df_train_full))
    test_preds_accum = np.zeros(len(df_test))
    test_ids = None

    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Split Data
        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_train_full.iloc[val_idx].reset_index(drop=True)

        # Create Datasets & Loaders
        train_ds = DogCatDataset(
            df_train_fold, transforms=get_transforms("train"), mode="train"
        )
        val_ds = DogCatDataset(
            df_val_fold, transforms=get_transforms("valid"), mode="val"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model and Optimizer
        model = DogCatClassifier().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=Config.min_lr
        )
        scaler = GradScaler()

        best_val_loss = float("inf")
        best_model_path = os.path.join(Config.model_dir, f"model_fold_{fold+1}.pth")

        # Training Loop
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, device, scaler, epoch
            )
            val_loss = validate(val_loader, model, criterion, device)
            scheduler.step()

            print(
                f"Fold {fold+1} Ep {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {"state_dict": model.state_dict(), "val_loss": val_loss},
                    is_best=True,
                    filename=f"model_fold_{fold+1}.pth",
                )

        # Load Best Model for Inference
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        # 1. OOF Inference (Validation)
        model.eval()
        fold_val_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                fold_val_preds.extend(probs.cpu().numpy())

        oof_preds[val_idx] = np.array(fold_val_preds)

        # 2. Test Inference (Accumulate for Ensemble)
        ids, preds = predict_test(test_loader, model, device)
        test_preds_accum += preds
        if test_ids is None:
            test_ids = ids

        # Cleanup
        del model, optimizer, scaler, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Validation Metric
    # -------------------------------------------------------------------------
    # Calculate Log Loss on OOF predictions
    y_true = df_train_full["label"].values
    final_metric = log_loss(y_true, oof_preds)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    errors = np.abs(y_true - oof_preds)

    widths, heights, ratios, sizes = [], [], [], []

    print("Extracting image features for correlation analysis...")
    for idx, row in df_train_full.iterrows():
        img_path = os.path.join(Config.input_dir, row["filepath"])
        try:
            size = os.path.getsize(img_path)
            img = cv2.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
                sizes.append(size)
            else:
                widths.append(0)
                heights.append(0)
                ratios.append(0)
                sizes.append(0)
        except:
            widths.append(0)
            heights.append(0)
            ratios.append(0)
            sizes.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    ratios = np.array(ratios)
    sizes = np.array(sizes)

    # Filter valid images
    mask = widths > 0

    features = {
        "Width": widths[mask],
        "Height": heights[mask],
        "Aspect Ratio": ratios[mask],
        "File Size": sizes[mask],
    }

    target_errors = errors[mask]

    print("Correlation between Error Magnitude and Input Features:")
    for name, feat_values in features.items():
        corr, p_val = stats.pearsonr(target_errors, feat_values)
        print(f"{name}: Correlation = {corr:.4f} (p={p_val:.4f})")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.013257633772229232

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        avg_test_preds = test_preds_accum / N_FOLDS
        save_submission(test_ids, avg_test_preds)
        print(
            f"Submission saved to {os.path.join(Config.submission_dir, 'submission.csv')}"
        )
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
