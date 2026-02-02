import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import library modules
from library.config import Config
from library.train import Trainer
from library.dataset import SpeechCommandsDataset
from library.utils import set_seed, load_checkpoint


def main():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # Optimize for A100 GPU and 2-hour runtime limit
    Config.BATCH_SIZE = 128
    Config.EPOCHS = 12  # Sufficient for convergence with pretrained backbone
    Config.NUM_WORKERS = 12

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(
        f"Configuration configured: Batch Size={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}"
    )

    # ==========================================
    # 2. Model Training
    # ==========================================
    trainer = Trainer()
    trainer.fit()

    # ==========================================
    # 3. Validation & Metrics
    # ==========================================
    print("\nRunning Final Validation Assessment...")

    # Load the best model (EMA weights) for evaluation
    model = trainer.ema.get_model()
    device = Config.DEVICE
    model.eval()

    # Create Validation Loader
    val_dataset = SpeechCommandsDataset(subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    final_metric = accuracy_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate binary error (1 if wrong, 0 if correct)
    errors = (all_preds != all_targets).astype(int)

    # Calculate correlation between Error and Target Label Index
    # This identifies if error is correlated with specific classes
    if len(np.unique(errors)) > 1:
        correlation = np.corrcoef(errors, all_targets)[0, 1]
        print(f"Correlation between Error Magnitude and Class Label: {correlation:.6f}")
    else:
        print(
            "No errors found or all errors (perfect correlation), skipping correlation calculation."
        )

    # Optional: Print per-class accuracy for detailed log
    print("Per-class Accuracy:")
    for label_idx, label_name in Config.IDX2LABEL.items():
        mask = all_targets == label_idx
        if np.sum(mask) > 0:
            acc = accuracy_score(all_targets[mask], all_preds[mask])
            print(f"  {label_name}: {acc:.4f}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9866209549293419

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
