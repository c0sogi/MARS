import os
import sys
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config, utils, dataset, model, trainer


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # Override config for fast baseline execution while retaining enough capacity to learn
    # The A100 is fast enough to handle 20 epochs in a short time.
    config.NUM_EPOCHS = 20

    # 2. Training
    # Initialize the trainer
    # This will load the ResNet34 model and setup optimizer/scheduler
    model_trainer = trainer.Trainer()

    # Run training
    # fit() handles data loading, training loop, validation per epoch, and checkpointing
    model_trainer.fit(num_epochs=config.NUM_EPOCHS)

    # 3. Validation & Failure Analysis
    print("Starting Post-Training Validation and Failure Analysis...")

    # We need to get the dataloaders again to access the validation set directly
    # Using the same batch size as config
    _, val_loader, test_loader = dataset.get_dataloaders(batch_size=config.BATCH_SIZE)

    # Ensure model is in eval mode and on correct device
    model_trainer.model.eval()
    device = model_trainer.device

    all_preds = []
    all_targets = []
    all_max_vals = []  # Feature for correlation analysis (Max signal strength)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model_trainer.model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Extract simple input feature: Max value per spectrogram (proxy for peak energy)
            # inputs shape: (Batch, Channels, Freq, Time)
            # Flatten non-batch dims and take max
            batch_max_vals = inputs.view(inputs.size(0), -1).max(dim=1).values
            all_max_vals.extend(batch_max_vals.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_max_vals = np.array(all_max_vals)

    # Calculate Final Validation Metric
    val_accuracy = np.mean(all_preds == all_targets)
    print(f"Final Validation Metric: {val_accuracy}")

    # Failure Analysis
    # Calculate correlation between Error (1 for error, 0 for correct) and Input Max Value
    errors = (all_preds != all_targets).astype(int)

    if np.std(errors) > 0 and np.std(all_max_vals) > 0:
        correlation = np.corrcoef(errors, all_max_vals)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error Magnitude and Input Max Value: {correlation}")

    # 4. Submission
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.9646499567847883

    if val_accuracy > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric meets threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        model_trainer.predict(test_loader)
    else:
        print(
            f"Validation metric {val_accuracy} does not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
