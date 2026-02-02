import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    METADATA_DIR,
    BEST_MODEL_PATH,
    SEED,
    NUM_WORKERS,
    DEVICE,
)
from library.utils import set_seed, get_device, load_checkpoint
from library.preprocessing import cache_dataset
from library.trainer import run_training, generate_submission
from library.dataset import HybridAudioDataset
from library.model import AudioClassifier


def main():
    # 1. Setup
    set_seed(SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Preprocessing
    # Pre-compute spectrograms and cache them to disk
    print("Executing preprocessing (caching features)...")
    cache_dataset(load_cached_data=True)

    # 3. Training
    # Run the full training loop using the provided trainer utility
    print("Starting training...")
    run_training(
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 4. Validation & Failure Analysis
    print("Performing validation and failure analysis...")

    # Load the best model saved during training
    model = AudioClassifier().to(device)
    try:
        load_checkpoint(BEST_MODEL_PATH, model, device=device)
        print("Best model loaded successfully.")
    except FileNotFoundError:
        print(
            f"Error: Best model not found at {BEST_MODEL_PATH}. Training may have failed."
        )
        return

    model.eval()

    # Create Validation Loader
    val_dataset = HybridAudioDataset(
        metadata_file=os.path.join(METADATA_DIR, "val.csv"),
        spec_augment=None,
        is_test=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    all_preds = []
    all_targets = []
    all_probs = []
    feature_signal_energy = []

    # Inference Loop
    with torch.no_grad():
        for spec, labels in val_loader:
            spec = spec.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(spec)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # Get probability assigned to the true class
            # gather expects index to have same dim as input, so we view labels
            true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()
            # Handle 0-d tensor edge case if batch_size=1
            if true_probs.ndim == 0:
                true_probs = true_probs.unsqueeze(0)
            all_probs.extend(true_probs.cpu().numpy())

            # For failure analysis, we can't easily get raw wave energy anymore
            # as we stopped loading it. We'll use spec energy proxy or just 0s.
            # Using spec mean energy as proxy:
            # spec: (B, 3, F, T)
            energies = spec.mean(dim=(1, 2, 3)).cpu().numpy()
            feature_signal_energy.extend(energies)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    feature_signal_energy = np.array(feature_signal_energy)

    # Calculate Metric
    correct = all_preds == all_targets
    accuracy = correct.mean()

    # Print required metric
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    # Error Magnitude = 1.0 - Probability of True Class
    error_magnitude = 1.0 - all_probs

    # Calculate correlation
    # We use Pearson correlation coefficient
    if len(error_magnitude) > 1:
        correlation_matrix = np.corrcoef(error_magnitude, feature_signal_energy)
        correlation = correlation_matrix[0, 1]
        print(
            f"Correlation between Error Magnitude and Signal Energy: {correlation:.6f}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.9832324978392394

    if accuracy > THRESHOLD:
        print(
            f"Validation accuracy meets threshold ({accuracy} > {THRESHOLD}). Generating submission..."
        )
        generate_submission(batch_size=BATCH_SIZE)
    else:
        print(
            f"Validation accuracy ({accuracy}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
