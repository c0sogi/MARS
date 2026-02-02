import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library.config import TrainConfig, PathConfig, DataConfig
from library.trainer import Trainer

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Configuration Overrides for Fast Baseline & Requirements
    # Optimize for A100 GPU and time constraints
    TrainConfig.EPOCHS = 15
    TrainConfig.BATCH_SIZE = 128
    TrainConfig.NUM_WORKERS = 4

    # Ensure submission path matches requirement
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    PathConfig.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Set seed for reproducibility
    set_seed(TrainConfig.SEED)

    # 2. Initialize and Train
    trainer = Trainer()

    # Run training
    # Output from trainer.fit() provides epoch logs
    trainer.fit()

    # 3. Load Best Model for Evaluation
    device = torch.device(TrainConfig.DEVICE)
    if os.path.exists(PathConfig.MODEL_SAVE_PATH):
        state_dict = torch.load(PathConfig.MODEL_SAVE_PATH, map_location=device)
        trainer.model.load_state_dict(state_dict)

    trainer.model.eval()

    # 4. Validation & Failure Analysis
    val_loader = trainer.val_loader
    criterion = nn.CrossEntropyLoss(reduction="none")

    all_losses = []
    all_signal_means = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = trainer.model(inputs)

            # Calculate loss per sample for failure analysis
            loss_per_sample = criterion(outputs, targets)
            all_losses.extend(loss_per_sample.cpu().numpy())

            # Calculate input feature stats (Mean Intensity) for correlation
            # inputs shape: (B, 1, Mels, Time)
            signal_means = inputs.view(inputs.size(0), -1).mean(dim=1)
            all_signal_means.extend(signal_means.cpu().numpy())

            # Accuracy calculation
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Compute Final Metric
    final_acc = correct / total
    print(f"Final Validation Metric: {final_acc}")

    # Compute Failure Analysis (Correlation)
    if len(all_losses) > 1:
        # Pearson correlation between Error (Loss) and Signal Intensity
        corr = np.corrcoef(all_losses, all_signal_means)[0, 1]
        print(
            f"Correlation between Error Magnitude and Input Signal Intensity: {corr:.6f}"
        )
    else:
        print("Not enough samples for failure analysis.")

    # 5. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.9754180602006689

    if final_acc > THRESHOLD:
        trainer.generate_submission()
    else:
        print(
            f"Validation accuracy {final_acc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
