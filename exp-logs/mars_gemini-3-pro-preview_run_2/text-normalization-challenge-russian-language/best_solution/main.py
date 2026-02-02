import pandas as pd
import numpy as np
import torch
import os
import sys
import time

# Import from library
from library.config import Config, set_seed
from library.trainer import ModelTrainer
from library.inference import HybridPredictor
from library.utils import load_metadata, is_semiotic


def main():
    # 1. Configuration
    # We override epochs to 5 to ensure the pipeline completes quickly as a baseline.
    # The A100 GPU can handle batch_size=256 easily.
    config = Config(epochs=5, batch_size=256)
    set_seed(config.seed)

    print("=== Starting Pipeline ===")

    # 2. Training
    # This trains the Tier 2 Transformer on semiotic tokens (digits/latin)
    # It automatically handles tokenization and dataset creation/caching.
    trainer = ModelTrainer(config)
    best_model_path = trainer.run()

    # 3. Validation (Hybrid: HFBB + Transformer)
    print("\n=== Running Validation ===")

    # Initialize Predictor (loads HFBB and the trained Transformer)
    predictor = HybridPredictor(config)

    # Load Validation Data
    val_df = load_metadata("val")

    # Sort to ensure context is correct (reconstruct sentence order)
    if "token_id" in val_df.columns:
        val_df["token_id_int"] = val_df["token_id"].astype(int)
        val_df.sort_values(["sentence_id", "token_id_int"], inplace=True)

    # Generate Context (Same logic as in inference.py)
    # We need prev_token and next_token for the HFBB and Transformer context
    val_df["prev_token"] = val_df["before"].shift(1).fillna("<START>")
    val_df["next_token"] = val_df["before"].shift(-1).fillna("<END>")
    val_df["prev_sent"] = val_df["sentence_id"].shift(1)
    val_df["next_sent"] = val_df["sentence_id"].shift(-1)

    # Apply sentence boundaries
    mask_start = val_df["prev_sent"] != val_df["sentence_id"]
    val_df.loc[mask_start, "prev_token"] = "<START>"

    mask_end = val_df["next_sent"] != val_df["sentence_id"]
    val_df.loc[mask_end, "next_token"] = "<END>"

    # Routing Logic
    predictions = [None] * len(val_df)
    tier2_indices = []

    # Extract records for fast iteration
    records = val_df[["before", "prev_token", "next_token"]].to_dict("records")

    # Step 1: HFBB Routing (Tier 1)
    for idx, row in enumerate(records):
        token = str(row["before"])
        prev_t = str(row["prev_token"])
        next_t = str(row["next_token"])

        # Query statistical memory
        pred, conf = predictor.hfbb.query(token, prev_t, next_t)

        if pred is not None:
            # If confidence is high (e.g. Trigram match or stable Unigram), use it
            if conf > config.hfbb_confidence_threshold:
                predictions[idx] = pred
            else:
                # Low confidence -> Route to Neural Net
                tier2_indices.append(idx)
        else:
            # OOV (Out of Vocabulary) Logic
            if is_semiotic(token):
                # Semiotic (Numbers/Latin) -> Route to Neural Net
                tier2_indices.append(idx)
            else:
                # Non-semiotic (Punctuation/Names) -> Identity Fallback
                predictions[idx] = token

    # Step 2: Tier 2 Inference (Transformer)
    if tier2_indices:
        print(f"Routing {len(tier2_indices)} validation tokens to Tier 2...")
        # Accessing protected method _run_transformer_inference to reuse batch logic
        # This method handles tokenization, batching, and greedy decoding
        tier2_preds = predictor._run_transformer_inference(val_df.iloc[tier2_indices])

        for idx, pred in zip(tier2_indices, tier2_preds):
            predictions[idx] = pred

    # Calculate Metric
    val_df["predicted"] = predictions
    # Ensure exact string matching for accuracy
    val_df["correct"] = val_df["predicted"] == val_df["after"]
    accuracy = val_df["correct"].mean()

    # Required Output Format
    print(f"Final Validation Metric: {accuracy}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = val_df[~val_df["correct"]].copy()
    print(f"Total Errors: {len(errors)} out of {len(val_df)}")

    if len(errors) > 0:
        # Correlation with input length
        val_df["len_before"] = val_df["before"].str.len()
        val_df["is_error"] = (~val_df["correct"]).astype(int)
        corr = val_df["len_before"].corr(val_df["is_error"])
        print(f"Correlation (Input Length vs Error): {corr:.4f}")

        # Error rate by class
        print("\nError Rate by Class:")
        class_counts = val_df["class"].value_counts()
        error_counts = errors["class"].value_counts()

        # Display top classes by error count
        sorted_classes = error_counts.sort_values(ascending=False).index
        for cls in sorted_classes[:10]:
            total = class_counts[cls]
            err = error_counts[cls]
            print(f"  {cls:<15}: {err/total:.4%} ({err}/{total})")

    # 5. Submission
    threshold = 0.9788071831831453
    if accuracy > threshold:
        print(
            f"\nValidation accuracy ({accuracy}) > threshold ({threshold}). Generating submission..."
        )
        predictor.generate_submission()
    else:
        print(
            f"\nValidation accuracy ({accuracy}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
