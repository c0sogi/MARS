import sys
import os
import numpy as np
import torch
import warnings
from scipy.stats import spearmanr

# Import classes and functions from the provided library files
from library.config import Config
from library.trainer import Trainer, validate
from library.dataset import set_seed


def main():
    # 1. Setup
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")

    # Set fixed seed for reproducibility
    set_seed(Config.SEED)

    print("Initializing Baseline Pipeline...")

    # 2. Model Training
    # Initialize the Trainer.
    # load_cached_data=True allows using preprocessed numpy files from ./working if available.
    # The dataset is small (~4.4k rows), so we use the full training set.
    trainer = Trainer(load_cached_data=True)

    # Execute the training pipeline.
    # We use 10 epochs which is sufficient for convergence on this small dataset
    # while keeping runtime very short (minutes on GPU).
    # Patience=3 ensures we stop early if validation loss plateaus.
    trainer.run(epochs=10, patience=3)

    # 3. Validation Assessment
    # The trainer automatically loads the best model weights at the end of run().
    # We now compute the final metric on the validation set for reporting.
    print("Computing Final Validation Metric...")
    val_loss, val_spearman = validate(
        trainer.model, trainer.val_loader, trainer.criterion, trainer.device
    )

    # STRICT REQUIREMENT: Print the final validation metric in this exact format.
    print(f"Final Validation Metric: {val_spearman}")

    # 4. Failure Analysis
    # We analyze if the model's error is correlated with the length of the input text.
    print("\nPerforming Failure Analysis...")
    trainer.model.eval()

    all_preds = []
    all_targets = []
    q_lens = []
    a_lens = []

    device = trainer.device

    with torch.no_grad():
        for q, a, y in trainer.val_loader:
            # Move inputs to device
            q_dev = q.to(device)
            a_dev = a.to(device)

            # Forward pass
            outputs = trainer.model(q_dev, a_dev)

            # Collect predictions and targets (move to CPU)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.numpy())

            # Calculate sequence lengths (count non-padding tokens)
            # Padding index is 0 in the Vocabulary
            q_len = (q_dev != 0).sum(dim=1).cpu().numpy()
            a_len = (a_dev != 0).sum(dim=1).cpu().numpy()

            q_lens.append(q_len)
            a_lens.append(a_len)

    # Concatenate all batches
    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)
    q_lens_arr = np.concatenate(q_lens)
    a_lens_arr = np.concatenate(a_lens)

    # Calculate Mean Absolute Error (MAE) per sample (averaged across all 30 target columns)
    # Shape: (N_samples,)
    sample_mae = np.mean(np.abs(preds_arr - targets_arr), axis=1)

    # Compute Spearman correlation between Error and Lengths
    corr_q, _ = spearmanr(sample_mae, q_lens_arr)
    corr_a, _ = spearmanr(sample_mae, a_lens_arr)

    print(f"Correlation between Error (MAE) and Question Length: {corr_q}")
    print(f"Correlation between Error (MAE) and Answer Length: {corr_a}")

    # 5. Submission
    # Generate submission only if validation metric exceeds the threshold
    if val_spearman > 0.20366743675214946:
        print(
            f"Validation metric {val_spearman} exceeds threshold. Generating submission..."
        )
        trainer.generate_submission()
        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"Submission successfully generated at: {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_spearman} did not exceed threshold. Submission skipped."
        )


if __name__ == "__main__":
    main()
