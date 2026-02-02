import os
import numpy as np
import torch
import pandas as pd
import sys

# Import provided library modules
import library.config as config
import library.utils as utils
import library.trainer as trainer_module


def main():
    # 1. Setup and Reproducibility
    utils.set_seed(config.SEED)

    print("Initializing Trainer...")
    # Initialize trainer with debug=False to use the full dataset for accurate metric calculation.
    # load_cached_data=True enables loading pre-processed .npy files if available in ./working.
    trainer = trainer_module.Trainer(load_cached_data=True, debug=False)

    print("Starting Training...")
    # Use config epochs (20) to allow convergence with Coordinate Attention.
    trainer.fit(epochs=config.EPOCHS, patience=5)

    print("Performing Final Validation & Failure Analysis...")
    # Load the best model weights saved during training for evaluation
    if os.path.exists(trainer.best_model_path):
        print(f"Loading best model from {trainer.best_model_path}")
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current model weights.")

    # Set model to evaluation mode
    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_errors = []

    # Storage for input features for failure analysis
    feat_means = []
    feat_stds = []
    feat_maxs = []

    # Validation Inference Loop
    # We use torch.no_grad() to optimize speed and memory
    with torch.no_grad():
        for data, target in trainer.val_loader:
            # Move data to GPU
            data = data.to(config.DEVICE)
            target = target.to(config.DEVICE)

            # Forward pass
            logits = trainer.model(data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets_np = target.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

            # Calculate absolute error for this batch
            batch_errors = np.abs(targets_np - probs)
            all_errors.extend(batch_errors)

            # Extract spectrogram features for correlation analysis
            # data shape: (Batch, 1, Freq, Time)
            # We flatten the spatial/temporal dimensions to compute stats per sample
            flat_data = data.view(data.size(0), -1)

            feat_means.extend(flat_data.mean(dim=1).cpu().numpy())
            feat_stds.extend(flat_data.std(dim=1).cpu().numpy())
            feat_maxs.extend(flat_data.max(dim=1).values.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)

    # 2. Compute Final Validation Metric
    val_auc = utils.compute_score(all_targets, all_preds)

    # Print the metric in the required format
    print(f"Final Validation Metric: {val_auc}")

    # 3. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Dictionary of features to correlate with error
    features = {
        "Spectrogram_Mean": np.array(feat_means),
        "Spectrogram_Std": np.array(feat_stds),
        "Spectrogram_Max": np.array(feat_maxs),
    }

    for name, vals in features.items():
        # Check for sufficient variance to avoid division by zero/NaN
        if np.std(vals) > 1e-9 and np.std(all_errors) > 1e-9:
            corr = np.corrcoef(all_errors, vals)[0, 1]
            print(f"Correlation between Error and {name}: {corr:.6f}")
        else:
            print(f"Correlation between Error and {name}: Undefined (low variance)")

    # 4. Submission Logic
    # Strict threshold as defined in the task
    THRESHOLD = 0.9942618903292241

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
