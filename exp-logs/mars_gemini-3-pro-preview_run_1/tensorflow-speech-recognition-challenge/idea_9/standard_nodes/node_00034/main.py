import os
import sys
import torch
import numpy as np
import warnings
import pandas as pd

# Import provided library modules
from library.config import Config, set_seed
from library.trainer import Trainer
from library.inference import generate_submission
from library.utils import map_prediction_to_label

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration for Fast Baseline
    # Override Config attributes for optimized runtime on A100
    Config.EPOCHS = 30
    Config.BATCH_SIZE = 256

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Training
    # Initialize Trainer (creates dataloaders, model, optimizer)
    # load_cached_data=True ensures we use pre-processed Parquet/NPZ files if available
    trainer = Trainer(load_cached_data=True)

    # Execute training
    # The scheduler in Trainer.__init__ uses Config.EPOCHS, so it syncs with our override.
    trainer.fit(epochs=Config.EPOCHS)

    # 3. Validation & Failure Analysis
    print("Starting Validation and Failure Analysis...")

    # Load the best model found during training
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=Config.DEVICE)
        trainer.model.load_state_dict(state_dict)
    else:
        print("Warning: Best model not found. Using current model state.")

    trainer.model.eval()

    # Containers for analysis
    all_preds_indices = []
    all_targets_indices = []
    all_spec_means = []

    # Metric counters
    correct_comp = 0
    total_samples = 0

    val_loader = trainer.val_loader

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            outputs = trainer.model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Collect data for analysis
            # Feature: Mean intensity of the spectrogram (proxy for signal energy/loudness)
            # inputs shape: (B, 1, F, T) -> mean over (1, 2, 3)
            spec_means = inputs.mean(dim=(1, 2, 3)).cpu().numpy()

            all_preds_indices.extend(preds.cpu().numpy())
            all_targets_indices.extend(targets.cpu().numpy())
            all_spec_means.extend(spec_means)

            # Compute Metric (Multiclass Accuracy on 12 classes)
            p_cpu = preds.cpu().numpy()
            t_cpu = targets.cpu().numpy()

            for p_idx, t_idx in zip(p_cpu, t_cpu):
                # Map fine-grained indices to 12-class strings
                pred_str = map_prediction_to_label(p_idx)
                target_str = map_prediction_to_label(t_idx)

                if pred_str == target_str:
                    correct_comp += 1

            total_samples += len(p_cpu)

    # Calculate Final Metric
    final_metric = correct_comp / total_samples if total_samples > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Define Error: 1 if prediction string != target string, else 0
    errors = []
    for p_idx, t_idx in zip(all_preds_indices, all_targets_indices):
        pred_str = map_prediction_to_label(p_idx)
        target_str = map_prediction_to_label(t_idx)
        errors.append(1 if pred_str != target_str else 0)

    errors = np.array(errors)
    all_spec_means = np.array(all_spec_means)
    all_targets_indices = np.array(all_targets_indices)

    # Correlation: Error vs Spectrogram Intensity
    if np.std(errors) > 0 and np.std(all_spec_means) > 0:
        corr_intensity = np.corrcoef(errors, all_spec_means)[0, 1]
    else:
        corr_intensity = 0.0

    # Correlation: Error vs Target Class ID (Fine-grained)
    # This indicates if error is dependent on the specific command class
    if np.std(errors) > 0 and np.std(all_targets_indices) > 0:
        corr_class = np.corrcoef(errors, all_targets_indices)[0, 1]
    else:
        corr_class = 0.0

    print("Failure Analysis Correlations:")
    print(f"Error vs Input Intensity: {corr_intensity}")
    print(f"Error vs Target Class ID: {corr_class}")

    # 5. Conditional Submission
    THRESHOLD = 0.9872909698996656

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE)
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
