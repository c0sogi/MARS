import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import SETIDataset, get_transforms
from library.model import SiameseEfficientNet
from library.engine import train_model, create_submission, validate


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.setup()

    # Use Config.EPOCHS directly (15) to ensure convergence (Cite solution_lesson_node_00024)
    device = torch.device(Config.DEVICE)

    print(
        f"Running Improved Solution: Siamese EfficientNet-B0 + GAP/GMP + Explicit Diff"
    )
    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Loading
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Dataset Initialization
    train_dataset = SETIDataset(df_train, transform=get_transforms("train"))
    val_dataset = SETIDataset(df_val, transform=get_transforms("valid"))
    test_dataset = SETIDataset(df_test, transform=get_transforms("test"))

    # DataLoader Initialization
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = SiameseEfficientNet(pretrained=Config.PRETRAINED).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Adjust T_max to match the overridden epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 4. Training
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=5,
        save_path=best_model_path,
    )

    # 5. Final Validation Metric
    # Reload best model to ensure we evaluate the best state
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # We re-run validation to get exact predictions for failure analysis
    print("Running final validation for metrics and failure analysis...")
    model.eval()
    val_preds = []
    val_targets = []

    # We also want to collect image stats for failure analysis
    # To save memory/time, we'll compute stats on the fly
    meta_stats = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)

            val_preds.extend(probs.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

            # Failure Analysis Feature Extraction
            # Images are (B, 6, H, W) where [0,2,4] are On, [1,3,5] are Off
            # We need to compute stats on CPU
            imgs_np = images.cpu().numpy()

            for img in imgs_np:
                # Split channels
                on_target = img[[0, 2, 4], :, :]
                off_target = img[[1, 3, 5], :, :]

                stats = {
                    "mean_on": np.mean(on_target),
                    "std_on": np.std(on_target),
                    "max_on": np.max(on_target),
                    "mean_off": np.mean(off_target),
                    "std_off": np.std(off_target),
                    "max_off": np.max(off_target),
                    "mean_diff": np.mean(on_target) - np.mean(off_target),
                }
                meta_stats.append(stats)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    final_auc = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_preds)

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(meta_stats)
    df_analysis["error"] = errors

    # Calculate correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # Interpretation
    top_corr = correlations.abs().idxmax()
    print(
        f"\nTop correlated feature with error: {top_corr} (Corr: {correlations[top_corr]:.4f})"
    )

    # 7. Submission
    THRESHOLD = 0.7930069652683209

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc:.5f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        create_submission(
            model=model,
            loader=test_loader,
            device=device,
            test_df=df_test,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation AUC ({final_auc:.5f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
