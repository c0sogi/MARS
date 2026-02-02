import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Import provided library modules
from library.config import (
    PathConfig,
    AudioConfig,
    MelConfig,
    ModelConfig,
    TrainConfig,
    LABELS,
    IDX_TO_LABEL,
)
from library.utils import set_seed, load_checkpoint
from library.trainer import Trainer
from library.dataset import get_dataloaders


def main():
    # 1. Set Reproducibility
    set_seed(42)

    # 2. Configuration
    path_config = PathConfig()
    audio_config = AudioConfig()
    mel_config = MelConfig()
    model_config = ModelConfig()

    # Configure training for a fast but high-performance baseline
    # 20 epochs is sufficient for ResNeSt50 to converge on this dataset size
    # Batch size 128 maximizes A100 GPU utilization
    train_config = TrainConfig(
        num_epochs=20,
        batch_size=128,
        learning_rate=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_patience=5,
    )

    print(f"Running on device: {train_config.device}")

    # 3. Initialize Trainer
    trainer = Trainer(path_config, audio_config, mel_config, model_config, train_config)

    # 4. Train Model
    # Uses GPU-native augmentation and feature extraction for speed
    print("Starting training...")
    trainer.fit(load_cached_data=True)

    # 5. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model saved during training
    best_model_path = os.path.join(path_config.working_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    load_checkpoint(trainer.model, best_model_path, device=train_config.device)

    # Switch to evaluation mode (disables Dropout, BatchNorm stats update)
    trainer.model.eval()
    trainer.mel_spectrogram.eval()

    # Get Validation Loader
    # We rely on caching to make this fast
    _, val_loader, _ = get_dataloaders(
        path_config, audio_config, train_config, load_cached_data=True
    )

    all_preds = []
    all_targets = []
    device = torch.device(train_config.device)

    # Inference Loop (Optimized with no_grad)
    with torch.no_grad():
        for waveforms, targets in val_loader:
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Feature Extraction (GPU)
            specs = trainer.mel_spectrogram(waveforms)

            # Forward Pass
            logits = trainer.model(specs)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    accuracy = np.mean(all_preds == all_targets)

    # Print required metric format
    print(f"Final Validation Metric: {accuracy}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # 1. Per-Class Accuracy
    # Identifies specific commands that are problematic
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(LABELS)))
    class_counts = cm.sum(axis=1)

    # Safe division
    per_class_acc = np.divide(
        cm.diagonal(),
        class_counts,
        out=np.zeros_like(cm.diagonal(), dtype=float),
        where=class_counts != 0,
    )

    print(f"{'Label':<10} | {'Accuracy':<10} | {'Count':<8}")
    print("-" * 34)
    for idx, acc in enumerate(per_class_acc):
        print(f"{LABELS[idx]:<10} | {acc:.4f}     | {class_counts[idx]:<8}")

    # 2. Correlation Analysis
    # Correlate error magnitude (binary error) with input features (Label Index and Class Frequency)
    errors = (all_preds != all_targets).astype(int)

    if len(errors) > 0:
        # Correlation with Label Index (checks for systematic error trend across arbitrary label order)
        corr_label = np.corrcoef(all_targets, errors)[0, 1]
        print(f"\nCorrelation between Label Index and Error: {corr_label:.4f}")

        # Correlation with Class Frequency (checks if rare classes have higher error rates)
        sample_class_counts = np.array([class_counts[t] for t in all_targets])
        if np.std(sample_class_counts) > 0:
            corr_freq = np.corrcoef(sample_class_counts, errors)[0, 1]
            print(f"Correlation between Class Frequency and Error: {corr_freq:.4f}")

    # 6. Submission Generation
    # Only generate if we meet the strict threshold
    THRESHOLD = 0.9832324978392394

    if accuracy > THRESHOLD:
        print(
            f"\nValidation metric ({accuracy}) > threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({accuracy}) <= threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
