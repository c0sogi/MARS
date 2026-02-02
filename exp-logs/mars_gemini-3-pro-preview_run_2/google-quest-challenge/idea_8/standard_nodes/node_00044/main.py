import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.engine import run_training, generate_submission, eval_fn
from library.model import SiameseCoAttentionNetwork


def main():
    # 1. Setup
    seed_everything(Config.seed)
    print(f"Device: {Config.device}")

    # 2. Data Loading
    # Using cached data as per instructions. The metadata indicates a small dataset (~4.4k train),
    # so we use the full available set for the best possible baseline score within the time limit.
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Training
    # run_training handles the loop, validation monitoring, and saving the best model.
    print("Starting training...")
    best_score = run_training(train_loader, val_loader)
    print(f"Training complete. Best Validation Score reported by engine: {best_score}")

    # 4. Evaluation & Failure Analysis
    print("\n--- Starting Evaluation & Failure Analysis ---")
    device = Config.device

    # Load the best model for analysis
    model = SiameseCoAttentionNetwork()
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Re-run evaluation to get predictions and precise metric
    criterion = torch.nn.BCELoss()
    val_loss, final_metric, val_preds = eval_fn(val_loader, model, criterion, device)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlate error with input lengths
    # Extract targets from the validation dataset (order is preserved as shuffle=False)
    val_targets = val_loader.dataset.targets

    # Compute Mean Absolute Error per sample (averaged across 30 targets)
    # val_preds: (N, 30), val_targets: (N, 30)
    # We use L1 error as a proxy for "difficulty"
    sample_errors = np.abs(val_preds - val_targets).mean(axis=1)

    # Extract lengths (token counts) from attention masks
    # masks are (N, max_len), sum gives number of real tokens
    q_lens = np.sum(val_loader.dataset.q_attention_mask, axis=1)
    a_lens = np.sum(val_loader.dataset.a_attention_mask, axis=1)

    # Compute Spearman correlation between error and lengths
    corr_q, _ = spearmanr(sample_errors, q_lens)
    corr_a, _ = spearmanr(sample_errors, a_lens)

    print("\nFailure Analysis - Error Correlation with Input Features:")
    print(f"Correlation (Error vs Question Length): {corr_q}")
    print(f"Correlation (Error vs Answer Length):   {corr_a}")

    # 5. Submission
    THRESHOLD = 0.41661963777166594

    if final_metric > THRESHOLD:
        print(f"\nValidation metric {final_metric} exceeds threshold {THRESHOLD}.")
        print("Generating submission file...")
        generate_submission(test_loader)
    else:
        print(
            f"\nValidation metric {final_metric} does not exceed threshold {THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
