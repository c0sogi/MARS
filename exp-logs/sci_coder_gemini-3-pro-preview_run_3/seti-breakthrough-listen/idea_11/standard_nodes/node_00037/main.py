import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import SETIDataset
from library.model import SiameseGatedEfficientNet
from library.engine import fit


def analyze_failures(model, loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlation between error magnitude and simple image statistics.
    Returns the ROC AUC score.
    """
    model.eval()
    targets = []
    preds = []

    # Feature accumulators for correlation analysis
    feat_mean_on = []
    feat_std_on = []
    feat_max_on = []
    feat_mean_off = []

    with torch.no_grad():
        for on_img, off_img, target in loader:
            on_img = on_img.to(device)
            off_img = off_img.to(device)

            # Inference
            logits = model(on_img, off_img)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            targets.extend(target.cpu().numpy())
            preds.extend(probs)

            # Extract simple image stats (move to CPU first)
            # on_img shape: (B, 3, H, W)
            on_np = on_img.cpu().numpy()
            off_np = off_img.cpu().numpy()

            # Compute stats per sample (aggregating over channels and spatial dims)
            feat_mean_on.extend(np.mean(on_np, axis=(1, 2, 3)))
            feat_std_on.extend(np.std(on_np, axis=(1, 2, 3)))
            feat_max_on.extend(np.max(on_np, axis=(1, 2, 3)))
            feat_mean_off.extend(np.mean(off_np, axis=(1, 2, 3)))

    targets = np.array(targets)
    preds = np.array(preds)

    # Calculate Error Magnitude
    errors = np.abs(targets - preds)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_on": feat_mean_on,
            "std_on": feat_std_on,
            "max_on": feat_max_on,
            "mean_off": feat_mean_off,
            "contrast": np.array(feat_mean_on) - np.array(feat_mean_off),
        }
    )

    # Compute Correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(correlations)

    # Calculate Final Metric
    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        auc = 0.5

    return auc


def inference_tta(model, loader, device):
    """
    Performs Test Time Augmentation (TTA) inference.
    Views: Original, H-Flip, V-Flip, HV-Flip.
    Handles padding correctly during vertical flips.
    """
    model.eval()
    all_preds = []
    h_valid = 273  # Valid height before padding (288)

    with torch.no_grad():
        for on_img, off_img, _ in loader:
            on_img = on_img.to(device)
            off_img = off_img.to(device)

            # 1. Original
            p1 = torch.sigmoid(model(on_img, off_img))

            # 2. Horizontal Flip (Time - Axis 3)
            p2 = torch.sigmoid(model(on_img.flip(3), off_img.flip(3)))

            # 3. Vertical Flip (Frequency - Axis 2)
            # Must slice valid region, flip, then place back to keep padding at bottom
            on_v = on_img.clone()
            off_v = off_img.clone()

            on_v[:, :, :h_valid, :] = on_v[:, :, :h_valid, :].flip(2)
            off_v[:, :, :h_valid, :] = off_v[:, :, :h_valid, :].flip(2)

            p3 = torch.sigmoid(model(on_v, off_v))

            # 4. HV-Flip (Flip H on top of V)
            on_hv = on_v.flip(3)
            off_hv = off_v.flip(3)
            p4 = torch.sigmoid(model(on_hv, off_hv))

            # Average predictions
            avg_p = (p1 + p2 + p3 + p4) / 4.0
            all_preds.extend(avg_p.cpu().numpy().flatten())

    return np.array(all_preds)


def main():
    # --- 1. Setup ---
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- 2. Data Loading ---
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    train_dataset = SETIDataset(df_train, mode="train")
    val_dataset = SETIDataset(df_val, mode="val")

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

    # --- 3. Model Initialization ---
    print("Initializing Model...")
    model = SiameseGatedEfficientNet(backbone_name=Config.BACKBONE_NAME).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion = nn.BCEWithLogitsLoss()

    # --- 4. Training ---
    print(f"Starting training for {Config.EPOCHS} epochs...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # --- 5. Validation & Failure Analysis ---
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    val_auc = analyze_failures(model, val_loader, device)
    print(f"Final Validation Metric: {val_auc}")

    # --- 6. Submission ---
    THRESHOLD = 0.7930069652683209

    if val_auc > THRESHOLD:
        print(
            f"Validation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = SETIDataset(df_test, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        preds = inference_tta(model, test_loader, device)

        df_test["target"] = preds
        df_test[["id", "target"]].to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
