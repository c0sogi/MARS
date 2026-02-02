import sys
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.engine import run_training, generate_submission, predict
from library.data import prepare_data
from library.model import ESIMHybridModel


def main():
    # 1. Initialization and Reproducibility
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Training
    # We use the provided engine function which handles the training loop and early stopping.
    # We stick to Config.EPOCHS (15) with Early Stopping as defined in the library.
    # This ensures a balance between speed and performance on the A100 GPU.
    print("Initiating training process...")
    best_model_path = run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # 3. Validation Inference
    print("Loading validation data for evaluation...")
    # Load validation data (leveraging cache)
    _, val_loader, _, _ = prepare_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # Load the best model
    device = Config.DEVICE
    model = ESIMHybridModel().to(device)
    state_dict = load_checkpoint(best_model_path, device)
    if state_dict is None:
        raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")
    model.load_state_dict(state_dict)

    # Ensure model is in eval mode
    model.eval()

    # Generate probabilities on validation set
    print("Running inference on validation set...")
    val_probs = predict(model, val_loader, device)

    # Retrieve Ground Truth and Scalars
    # The dataset is not shuffled for validation, so order is preserved.
    # Accessing underlying tensors from the dataset.
    val_targets = val_loader.dataset.targets.numpy()
    val_scalars = val_loader.dataset.scalars.numpy()

    # 4. Metric Calculation
    # Calculate Log Loss (Cross Entropy)
    # sklearn log_loss handles soft labels and clipping (eps=auto)
    metric = log_loss(val_targets, val_probs)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate per-sample log loss (Cross Entropy) manually for analysis
    # Clip to avoid log(0)
    epsilon = 1e-15
    val_probs_clipped = np.clip(val_probs, epsilon, 1 - epsilon)
    # CE = - sum(y_true * log(y_pred))
    sample_losses = -np.sum(val_targets * np.log(val_probs_clipped), axis=1)

    # Input features for correlation: Log lengths of Prompt, Response A, Response B
    # These correspond to the 3 scalar features in the dataset
    feature_names = [
        "Log Prompt Length",
        "Log Response A Length",
        "Log Response B Length",
    ]

    print("Correlation between Error Magnitude (Log Loss) and Input Features:")
    for i, name in enumerate(feature_names):
        # val_scalars is (N, 3)
        feature_values = val_scalars[:, i]

        # Calculate Pearson Correlation
        if np.std(feature_values) == 0 or np.std(sample_losses) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(feature_values, sample_losses)

        print(f"{name}: {corr:.4f}")

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = 1.0392143626595562

    if metric < THRESHOLD:
        print(
            f"Validation metric {metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(best_model_path)
    else:
        print(
            f"Validation metric {metric} is NOT below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
