import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.stats import pearsonr

# Append current directory to system path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.model import AudioClassifier


def main():
    # 1. Configuration & Setup
    # Limit epochs for fast baseline execution while retaining enough capacity to learn
    Config.MAX_EPOCHS = 15
    # Ensure we use the GPU
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(Config.SEED)

    print(f"Running on device: {Config.DEVICE}")
    print(f"Max Epochs: {Config.MAX_EPOCHS}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Training
    trainer = Trainer(train_loader, val_loader, test_loader)
    trainer.fit()

    # 4. Final Validation & Metric Calculation
    print("Loading best model for evaluation...")
    best_model_path = os.path.join(Config.SAVE_DIR, "best_model.pth")

    # Re-initialize model and load weights
    model = AudioClassifier(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    )
    model.to(Config.DEVICE)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: Best model not found. Using current model.")
        model = trainer.model

    model.eval()

    all_targets = []
    all_outputs = []
    all_fnames = []

    # Use reduction='none' to get per-sample loss for failure analysis
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    all_losses = []

    print("Evaluating on validation set...")
    with torch.no_grad():
        for specs, labels, fnames in val_loader:
            specs = specs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            logits = model(specs)

            # Calculate per-sample loss (mean over classes)
            loss_per_sample = criterion(logits, labels).mean(dim=1)

            all_targets.append(labels.cpu())
            all_outputs.append(logits.cpu())
            all_losses.append(loss_per_sample.cpu())
            all_fnames.extend(fnames)

    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_losses = torch.cat(all_losses, dim=0).numpy()

    # Calculate LWLRAP
    metric = calculate_lwlrap(all_targets, all_outputs)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Load validation metadata to get file paths and labels
    val_df = pd.read_csv(Config.VAL_CSV)

    # Create mappings
    # fname -> full_path
    fname_to_path = pd.Series(val_df.file_path.values, index=val_df.fname).to_dict()
    # fname -> label_count
    val_df["label_count"] = val_df["labels"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )
    fname_to_label_count = pd.Series(
        val_df.label_count.values, index=val_df.fname
    ).to_dict()

    durations = []
    label_counts = []

    for fname in all_fnames:
        # Duration
        rel_path = fname_to_path.get(fname)
        if rel_path:
            full_path = os.path.join(Config.INPUT_ROOT, rel_path)
            try:
                info = sf.info(full_path)
                durations.append(info.duration)
            except:
                durations.append(0.0)
        else:
            durations.append(0.0)

        # Label Count
        label_counts.append(fname_to_label_count.get(fname, 0))

    durations = np.array(durations)
    label_counts = np.array(label_counts)

    # Calculate correlations
    # Avoid NaN if variance is 0
    if np.std(durations) > 0:
        corr_duration, _ = pearsonr(all_losses, durations)
    else:
        corr_duration = 0.0

    if np.std(label_counts) > 0:
        corr_labels, _ = pearsonr(all_losses, label_counts)
    else:
        corr_labels = 0.0

    print(f"Correlation (Error vs Duration): {corr_duration:.4f}")
    print(f"Correlation (Error vs Label Count): {corr_labels:.4f}")

    # 6. Submission
    THRESHOLD = 0.8287354381163734
    if metric > THRESHOLD:
        print(
            f"\nMetric ({metric:.6f}) > Threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nMetric ({metric:.6f}) <= Threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
