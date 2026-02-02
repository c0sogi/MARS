import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys

# Import library components
from library.config import Config, set_seed
from library.utils import is_semiotic
from library.data_factory import _add_context_and_filter, TransformerDataset
from library.hybrid_system import HybridPredictor


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    print(f"Execution Device: {Config.DEVICE}")

    # 2. Initialize and Train Hybrid System
    # We limit max_train_samples to 2M and epochs to 5 to ensure execution finishes < 2 hours
    # while still providing a strong baseline.
    predictor = HybridPredictor(device=Config.DEVICE)

    print("Starting Hybrid System Training...")
    predictor.train_systems(load_cached_data=True, epochs=5, max_train_samples=2000000)

    # 3. Validation Inference
    print("\n=== Running Validation ===")
    val_path = Config.VAL_CSV
    if not os.path.exists(val_path):
        print(f"Error: Validation file {val_path} not found.")
        return

    # Load and process validation data
    # is_train=False ensures we keep ALL tokens (don't filter semiotic only)
    # This matches the test set inference pipeline.
    df_val_raw = pd.read_csv(val_path)
    df_val = _add_context_and_filter(df_val_raw, is_train=False, load_cached_data=False)

    # Prepare data for iteration
    befores = df_val["before"].fillna("").astype(str).tolist()
    prevs = df_val["prev"].fillna("").astype(str).tolist()
    nexts = df_val["next"].fillna("").astype(str).tolist()
    actuals = df_val["after"].fillna("").astype(str).tolist()

    # We need id_str for the dataset class if we use it
    if "id_str" not in df_val.columns:
        df_val["id_str"] = (
            df_val["sentence_id"].astype(str) + "_" + df_val["token_id"].astype(str)
        )

    final_preds = {}
    transformer_indices = []

    print(f"Validating on {len(df_val)} tokens...")

    # Cascade Logic Step 1 & 2: HFBB and Filtering
    for idx, (curr, p, n) in enumerate(zip(befores, prevs, nexts)):
        # Query Tier 1
        res, level = predictor.hfbb.query(curr, p, n)

        # Priority: Trigram > Bigram
        if level in ["trigram", "bigram_prev", "bigram_next"]:
            final_preds[idx] = res
            continue

        # Check Tier 2 Condition
        if is_semiotic(curr):
            transformer_indices.append(idx)
            continue

        # Fallback
        if res is not None:
            final_preds[idx] = res
        else:
            final_preds[idx] = curr

    # Cascade Step 3: Transformer Inference
    if transformer_indices:
        print(
            f"Tier 2: processing {len(transformer_indices)} semiotic tokens via Transformer..."
        )

        # Subset dataframe
        df_subset = df_val.iloc[transformer_indices].copy()

        # Create Dataset/Loader
        # Note: We use is_test=True to ignore targets in the dataset __getitem__,
        # though we are validating. We just need input_ids.
        dataset = TransformerDataset(df_subset, predictor.tokenizer, is_test=True)
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        batch_preds = []
        # Run inference
        # predict_batch handles torch.no_grad()
        total_batches = len(loader)
        for i, batch in enumerate(loader):
            src = batch["input_ids"]
            decoded_texts = predictor.generator.predict_batch(src)
            batch_preds.extend(decoded_texts)

        # Map back results
        for i, original_idx in enumerate(transformer_indices):
            final_preds[original_idx] = batch_preds[i]

    # Assemble final prediction list
    pred_list = [final_preds.get(i, befores[i]) for i in range(len(df_val))]

    # 4. Evaluation and Metrics
    correct_count = sum(1 for p, a in zip(pred_list, actuals) if p == a)
    accuracy = correct_count / len(df_val)

    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    df_val["pred"] = pred_list
    df_val["is_error"] = (df_val["pred"] != df_val["after"]).astype(int)
    df_val["len_before"] = df_val["before"].apply(len)

    # Correlation
    corr = df_val["is_error"].corr(df_val["len_before"])
    print(f"Correlation (Error vs Input Length): {corr:.6f}")

    # Error by Class
    if "class" in df_val.columns:
        print("Error Rate by Class (Top 5):")
        class_errors = (
            df_val.groupby("class")["is_error"].mean().sort_values(ascending=False)
        )
        print(class_errors.head(5))

    # 5. Submission
    THRESHOLD = 0.9784022349361615
    if accuracy > THRESHOLD:
        print(f"\nValidation metric {accuracy} > {THRESHOLD}. Generating submission...")
        predictor.generate_submission(load_cached_data=True)
    else:
        print(f"\nValidation metric {accuracy} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
