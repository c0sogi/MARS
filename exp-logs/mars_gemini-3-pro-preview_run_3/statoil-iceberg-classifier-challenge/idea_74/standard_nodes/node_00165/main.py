import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import get_fold_loaders
from library.model import HCICNN
from library.train_eval import Trainer, generate_submission


def main():
    # 1. Setup and Fast Baseline Configuration
    # Reduce epochs to ensure execution within time limit while maintaining reasonable performance
    Config.NUM_EPOCHS = 25
    Config.setup_directories()
    set_seed(Config.SEED)

    print(f"Starting execution on device: {Config.DEVICE}")
    print(f"Training with {Config.NUM_FOLDS} folds, {Config.NUM_EPOCHS} epochs each.")

    all_preds = []
    all_targets = []
    all_stats = []

    # 2. Cross-Validation Loop
    for fold_idx in range(Config.NUM_FOLDS):
        print(f"\n--- Fold {fold_idx} ---")

        # Load Data
        train_loader, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        # Initialize Model
        model = HCICNN().to(Config.DEVICE)

        # Initialize Optimizer & Loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Train
        trainer = Trainer(model, Config.DEVICE, optimizer, criterion)
        # Trainer.fit saves the best model to Config.CHECKPOINT_DIR
        trainer.fit(train_loader, val_loader, fold_idx)

        # Load Best Model for this fold for Validation Inference
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)
        model.eval()

        # Inference on Validation Set
        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                images = batch["image"].to(Config.DEVICE)
                angles = batch["angle"].to(Config.DEVICE)
                labels = batch["label"].to(Config.DEVICE)

                # Forward
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                fold_preds.extend(probs)
                fold_targets.extend(labels.cpu().numpy().flatten())

                # Collect features for Failure Analysis
                # images: (B, 3, 75, 75). Channel 0=HH, Channel 1=HV
                imgs_np = images.cpu().numpy()
                angles_np = angles.cpu().numpy().flatten()
                labels_np = labels.cpu().numpy().flatten()

                for i in range(len(probs)):
                    b1 = imgs_np[i, 0]
                    b2 = imgs_np[i, 1]

                    stat = {
                        "inc_angle": angles_np[i],
                        "b1_mean": np.mean(b1),
                        "b1_std": np.std(b1),
                        "b2_mean": np.mean(b2),
                        "b2_std": np.std(b2),
                        "target": labels_np[i],
                        "prediction": probs[i],
                    }
                    all_stats.append(stat)

        all_preds.extend(fold_preds)
        all_targets.extend(fold_targets)

    # 3. Global Validation Metric
    final_metric = log_loss(all_targets, all_preds)
    # Print exact format required
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(all_stats)
    # Calculate error magnitude
    df_analysis["error"] = np.abs(df_analysis["target"] - df_analysis["prediction"])

    # Compute correlation
    cols = ["inc_angle", "b1_mean", "b1_std", "b2_mean", "b2_std", "error"]
    corr = df_analysis[cols].corr()["error"].drop("error").sort_values(ascending=False)
    print("Correlation between Error Magnitude and Input Features:")
    print(corr)

    # 5. Submission Generation
    THRESHOLD = 0.17174082291273365
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
