import sys
import os
import torch
import pandas as pd
import numpy as np
import nltk
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import compute_attribute_stats, compute_levenshtein
from library.tokenizer import get_tokenizer
from library.dataset import get_dataloaders
from library.model import AttributeContextualizedTransformer
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    # 1. Setup and Config Overrides for Fast Baseline
    Config.setup()

    # Override Config for speed constraints
    # We use a small subset for training to ensure we fit within the time limit
    # while reserving time for the mandatory full test set inference.
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2

    print("=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Preparation
    print("\n=== Data Preparation ===")
    # Ensure tokenizer and stats are ready
    tokenizer = get_tokenizer(load_cached_data=True)
    compute_attribute_stats(load_cached_data=True)

    # Get DataLoaders (using debug=True to use the small sample size defined above)
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Model Initialization
    print("\n=== Model Initialization ===")
    model = AttributeContextualizedTransformer()
    model.to(Config.DEVICE)

    # 4. Training
    print("\n=== Starting Training ===")
    trainer = Trainer(model, train_loader, val_loader, test_loader, tokenizer)
    trainer.fit()

    # 5. Final Validation Assessment
    print("\n=== Final Validation Assessment ===")
    # We run a manual inference pass on the validation set to get clean predictions
    # for metric calculation and failure analysis.
    model.eval()
    all_preds = []
    all_truths = []
    all_lens = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(Config.DEVICE)
            seq = batch["seq"].to(Config.DEVICE)
            # Actual lengths of ground truth sequences (useful for failure analysis)
            # seq_len in batch includes EOS, so it's a good proxy for complexity
            seq_lens = batch["seq_len"].numpy()

            # Predict using greedy decoding
            pred_seqs = model.predict(images, max_len=Config.MAX_LEN)

            # Decode sequences to text
            pred_texts = [tokenizer.sequence_to_text(s) for s in pred_seqs]
            truth_texts = [tokenizer.sequence_to_text(s) for s in seq]

            all_preds.extend(pred_texts)
            all_truths.extend(truth_texts)
            all_lens.extend(seq_lens)

    # Compute Metric
    final_metric = compute_levenshtein(all_preds, all_truths)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample Levenshtein distance
    lev_distances = []
    for p, t in zip(all_preds, all_truths):
        lev_distances.append(nltk.edit_distance(p, t))

    lev_distances = np.array(lev_distances)
    all_lens = np.array(all_lens)

    # Correlation: Error vs Sequence Length
    if len(lev_distances) > 1:
        corr, _ = pearsonr(lev_distances, all_lens)
        print(
            f"Correlation between Error (Levenshtein) and Target Sequence Length: {corr:.4f}"
        )

        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation between sequence length and error rate."
            )
        else:
            print(
                "Observation: Error rate is relatively independent of sequence length."
            )
    else:
        print("Not enough samples for correlation analysis.")

    # 7. Submission
    print("\n=== Submission Generation ===")
    threshold = 81.60407868615773

    if final_metric < threshold:
        print(
            f"Validation metric ({final_metric}) is better (lower) than threshold ({threshold}). Generating submission..."
        )

        # We must generate for the ENTIRE test set, so we pass debug=False here.
        # This will trigger get_dataloaders with debug=False internally for the test set.
        generate_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            load_cached_data=False,  # Force regeneration with the newly trained model
            debug=False,
        )
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
