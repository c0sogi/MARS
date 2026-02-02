import os
import sys
import torch
import numpy as np
import pandas as pd

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.trainer import Trainer
from library.utils import load_checkpoint, logger


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Limit epochs to ensure fast baseline execution while sufficient for convergence
    Config.NUM_EPOCHS = 15
    set_seed(Config.SEED)

    # ==========================================
    # 2. Training
    # ==========================================
    # Initialize Trainer (handles Data, Model, Optimizer, Scheduler)
    trainer = Trainer()

    # Execute Training Loop
    trainer.fit()

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    logger.info("Performing Validation Assessment...")

    # Load the best model weights (EMA) for evaluation
    # We load into the EMA shadow model to ensure we evaluate the smoothed weights
    load_checkpoint(trainer.ema, filename="best_model.pth")
    eval_model = trainer.ema.ema_model
    eval_model.eval()

    # Access validation data from trainer
    val_data = trainer.val_data
    batch_iterator = val_data.get_iterator(Config.BATCH_SIZE)

    all_probs = []
    all_labels = []
    all_rms = []

    # Inference Loop
    with torch.no_grad():
        for waveforms, labels in batch_iterator:
            # Data is already on GPU from GPUDataset

            # Forward pass (noise_bank=None disables noise injection)
            logits = eval_model(waveforms, noise_bank=None)
            probs = torch.softmax(logits, dim=1)

            # Collect predictions and labels
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Calculate Input Feature for Failure Analysis: RMS Energy
            # waveforms shape: (Batch, Time)
            rms = torch.sqrt(torch.mean(waveforms**2, dim=1))
            all_rms.append(rms.cpu().numpy())

    # Concatenate batches
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_rms = np.concatenate(all_rms)

    # Compute Accuracy
    predictions = np.argmax(all_probs, axis=1)
    accuracy = np.mean(predictions == all_labels)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")

    # Calculate Error Magnitude: 1.0 - Probability assigned to the True Class
    # We use advanced indexing to get the prob of the true label for each sample
    true_class_probs = all_probs[np.arange(len(all_labels)), all_labels]
    error_magnitude = 1.0 - true_class_probs

    # Calculate Correlation between Error Magnitude and Input RMS Energy
    corr_matrix = np.corrcoef(error_magnitude, all_rms)
    correlation = corr_matrix[0, 1]

    print(
        f"Correlation between the model's error magnitude and the input features: {correlation}"
    )

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    threshold = 0.9867045739610335

    if accuracy > threshold:
        logger.info(
            f"Validation metric {accuracy} exceeds threshold {threshold}. Generating submission..."
        )
        trainer.generate_submission()
    else:
        logger.warning(
            f"Validation metric {accuracy} does NOT exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
