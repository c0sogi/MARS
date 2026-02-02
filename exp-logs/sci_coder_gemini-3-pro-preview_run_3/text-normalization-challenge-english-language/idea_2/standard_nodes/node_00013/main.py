import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.data import load_metadata, NormalizationDataset, collate_fn
from library.trainer import run_training
from library.inference import HybridNormalizer


def main():
    # Set seed for reproducibility
    set_seed()

    print("=== Step 1: Training Neural Model ===")
    # Train the neural model on the full 'hard' (filtered) data to maximize coverage.
    # Cite solution_lesson_node_00010: Training on full data improves "tail" performance.
    run_training(debug_sample_size=None, load_cached_data=True)

    print("\n=== Step 2: Initializing Hybrid Normalizer ===")
    # This initializes the HybridNormalizer, which:
    # 1. Builds/Loads the Tokenizer
    # 2. Fits/Loads the Symbolic Memory (N-gram tables) from the full training set
    # 3. Loads the trained Neural Model weights
    normalizer = HybridNormalizer(load_cached_data=True)

    print("\n=== Step 3: Running Validation on Full Hold-out Set ===")
    # We need to manually process the validation set because library.data.process_data
    # filters out PLAIN/PUNCT classes for 'val' split (intended for neural training),
    # but we need the full dataset to calculate the global validation metric.

    df_val = load_metadata("val")

    # Generate Context (prev/next) manually
    # Ensure sorting to respect sentence boundaries
    if "sentence_id" in df_val.columns and "token_id" in df_val.columns:
        df_val = df_val.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

    df_val["before"] = df_val["before"].fillna("").astype(str)
    df_val["after"] = df_val["after"].fillna("").astype(str)

    # Vectorized context generation
    print("Generating validation context...")
    sent_ids = df_val["sentence_id"]
    prev_series = df_val["before"].shift(1).fillna("")
    next_series = df_val["before"].shift(-1).fillna("")

    # Mask context where sentence ID changes
    is_same_prev = sent_ids == sent_ids.shift(1)
    is_same_next = sent_ids == sent_ids.shift(-1)

    df_val["prev"] = np.where(is_same_prev, prev_series, "")
    df_val["next"] = np.where(is_same_next, next_series, "")

    # Prepare for inference
    total_samples = len(df_val)
    predictions = [None] * total_samples

    # Convert to lists for fast iteration
    prevs = df_val["prev"].tolist()
    currs = df_val["before"].tolist()
    nexts = df_val["next"].tolist()
    targets = df_val["after"].tolist()

    neural_indices = []
    neural_rows = []

    print("Running Hybrid Inference on Validation Set...")
    # Hybrid Inference Loop
    for i in range(total_samples):
        p, c, n = prevs[i], currs[i], nexts[i]

        # 1. Symbolic Lookup (Trigram -> Bigram -> Unigram)
        res = normalizer.symbolic_mem.query(p, c, n)
        if res is not None:
            predictions[i] = res
            continue

        # 2. Heuristic Filter
        # If token is purely alphabetic and not in lookup, predict identity.
        # This handles rare proper nouns without risking neural hallucination.
        if c.isalpha():
            predictions[i] = c
            continue

        # 3. Neural Candidate
        # Queue complex tokens (digits, symbols) for the Seq2Seq model
        neural_indices.append(i)
        neural_rows.append({"before": c, "prev": p, "next": n})

    # Batch Neural Inference
    if neural_indices:
        print(f"Running neural inference on {len(neural_indices)} complex tokens...")
        df_neural = pd.DataFrame(neural_rows)
        # Use the tokenizer from the loaded normalizer
        ds_neural = NormalizationDataset(df_neural, normalizer.tokenizer)
        dl_neural = DataLoader(
            ds_neural,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=os.cpu_count() or 4,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Predict using the neural solver
        neural_preds = normalizer.neural_solver.predict(dl_neural)

        # Map predictions back to the main list
        for idx, pred in zip(neural_indices, neural_preds):
            predictions[idx] = pred

    # Fill remaining Nones with identity (safety fallback)
    for i in range(total_samples):
        if predictions[i] is None:
            predictions[i] = currs[i]

    # Calculate Metric
    correct_count = sum(1 for p, t in zip(predictions, targets) if p == t)
    accuracy = correct_count / total_samples

    # Print Metric with full precision as required
    print(f"Final Validation Metric: {accuracy}")

    print("\n=== Step 4: Failure Analysis ===")
    # Add predictions to dataframe for analysis
    df_val["pred"] = predictions
    df_val["is_error"] = (df_val["pred"] != df_val["after"]).astype(int)

    # Compute features for correlation analysis
    df_val["len_before"] = df_val["before"].str.len()
    df_val["num_digits"] = df_val["before"].apply(lambda x: sum(c.isdigit() for c in x))

    # Correlations
    corr_len = df_val["len_before"].corr(df_val["is_error"])
    corr_digits = df_val["num_digits"].corr(df_val["is_error"])

    print(f"Correlation (Error vs Input Length): {corr_len}")
    print(f"Correlation (Error vs Num Digits): {corr_digits}")

    print("\n=== Step 5: Submission Generation ===")
    threshold = 0.9943860453286453
    if accuracy > threshold:
        print(f"Accuracy {accuracy} > {threshold}. Generating submission...")
        # predict_dataset handles loading test data, running the pipeline, and saving the CSV
        normalizer.predict_dataset(load_cached_data=True)
    else:
        print(f"Accuracy {accuracy} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
