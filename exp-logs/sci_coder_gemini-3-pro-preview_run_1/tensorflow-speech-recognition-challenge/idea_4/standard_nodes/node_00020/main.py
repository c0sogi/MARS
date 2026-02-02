import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
import importlib
import library.config

importlib.reload(library.config)
from library.config import Config
from library.dataset import SpeechCommandDataset
from library.model import DilatedEfficientNet
from library.trainer import Trainer
from library.utils import set_seed


def main():
    # 1. Configuration and Setup
    # Using 50 epochs as recommended in the Idea description to ensure convergence
    # with Mixup and EfficientNet-B2.
    config = Config(epochs=50)
    set_seed(config.seed)

    print(f"Running on device: {config.device}")
    print(f"Working directory: {config.working_dir}")

    # 2. Data Preparation
    # Train Dataset (Balanced, Cached)
    train_dataset = SpeechCommandDataset(
        mode="train", config=config, load_cached_data=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Dataset
    val_dataset = SpeechCommandDataset(mode="val", config=config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Test Dataset
    test_dataset = SpeechCommandDataset(mode="test", config=config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    model = DilatedEfficientNet(config)

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

    # 5. Evaluation and Failure Analysis
    print("\nStarting Evaluation and Failure Analysis...")

    # Load the best model weights
    best_model_path = os.path.join(config.working_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    # Features for failure analysis (extracted from spectrograms)
    feat_means = []
    feat_stds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config.device)
            # inputs shape: [Batch, 1, F, T]

            # Extract features from input spectrograms for analysis
            # Mean intensity (loudness proxy) and Std Dev (complexity/contrast proxy)
            # Flatten spatial dims: [B, F*T]
            flat_inputs = inputs.view(inputs.size(0), -1)
            batch_means = flat_inputs.mean(dim=1).cpu().numpy()
            batch_stds = flat_inputs.std(dim=1).cpu().numpy()

            feat_means.extend(batch_means)
            feat_stds.extend(batch_stds)

            # Inference
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    feat_means = np.array(feat_means)
    feat_stds = np.array(feat_stds)

    # Calculate Metric
    accuracy = np.mean(all_preds == all_targets)
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    # Error Magnitude = 1.0 - Probability assigned to the correct class
    # Use integer array indexing to get the prob of the target class
    rows = np.arange(len(all_targets))
    target_probs = all_probs[rows, all_targets]
    error_magnitudes = 1.0 - target_probs

    # Correlations
    # Using numpy for correlation to avoid extra dependencies
    corr_mean = np.corrcoef(error_magnitudes, feat_means)[0, 1]
    corr_std = np.corrcoef(error_magnitudes, feat_stds)[0, 1]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print(f"Input Spectrogram Mean Intensity: {corr_mean:.6f}")
    print(f"Input Spectrogram Std Deviation:  {corr_std:.6f}")

    # 6. Submission
    # Threshold from task description
    THRESHOLD = 0.9737458193979933

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nValidation accuracy ({accuracy}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
