import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import LungDataset, get_transforms
from library.network import PriorPreservingDualAxisNet
from library.engine import run_training, predict_and_submit


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for faster execution within time limits
    Config.EPOCHS = 25
    Config.PATIENCE = 6

    print(f"Initializing run on {device}...")

    # 2. Data Loading
    # Train Loader
    train_dataset = LungDataset(
        metadata_path=Config.TRAIN_META_PATH,
        mode="train",
        transform=get_transforms(mode="train"),
        load_cached_data=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Loader
    val_dataset = LungDataset(
        metadata_path=Config.VAL_META_PATH,
        mode="val",
        transform=get_transforms(mode="val"),
        load_cached_data=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = PriorPreservingDualAxisNet()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    run_training(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        device,
        Config.EPOCHS,
        Config.PATIENCE,
        model_save_path,
    )

    # 5. Final Validation & Failure Analysis
    print("\nLoading best model for final evaluation and analysis...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_targets = []
    all_preds = []
    all_sigmas = []
    all_deltas = []
    # Metadata for correlation analysis
    meta_age = []
    meta_percent = []
    meta_base_fvc = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)
            w_delta = batch["week_delta"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Inference
            preds = model(img_ax, img_cor, tab)

            # Unpack predictions
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Reconstruct FVC and Confidence
            fvc_pred = base_fvc + alpha * w_delta
            sigma_pred = sigma_base + sigma_growth * torch.abs(w_delta)

            # Collect results
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(fvc_pred.cpu().numpy())
            all_sigmas.extend(sigma_pred.cpu().numpy())
            all_deltas.extend(w_delta.cpu().numpy())

            # Collect metadata (Tabular tensor: [Age, Sex, Smoking, Percent, BaseFVC])
            tab_cpu = tab.cpu().numpy()
            meta_age.extend(tab_cpu[:, 0])
            meta_percent.extend(tab_cpu[:, 3])
            meta_base_fvc.extend(tab_cpu[:, 4])

    # Compute Final Metric
    final_metric = get_score(
        np.array(all_targets), np.array(all_preds), np.array(all_sigmas)
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    errors = np.abs(targets_np - preds_np)

    print("\n=== Failure Analysis ===")
    print(f"Mean Absolute Error on Validation Set: {np.mean(errors):.4f} ml")

    # Correlation Analysis
    features = {
        "Time Delta (Weeks)": np.abs(np.array(all_deltas)),
        "Age (Normalized)": np.array(meta_age),
        "Percent (Normalized)": np.array(meta_percent),
        "Baseline FVC (Normalized)": np.array(meta_base_fvc),
    }

    print("Correlation between Absolute Error and Input Features:")
    for name, values in features.items():
        if len(np.unique(values)) > 1:
            corr, _ = pearsonr(errors, values)
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (Constant)")

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, device, Config.TEST_META_PATH, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
