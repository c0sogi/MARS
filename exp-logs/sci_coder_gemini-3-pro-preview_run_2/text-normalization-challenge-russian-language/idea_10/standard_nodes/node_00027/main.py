import pandas as pd
import numpy as np
import torch
import sys
import os

# Import library modules
from library import config
from library import utils
from library import hfbb
from library import tokenizers
from library import dataset
from library import model as model_lib
from library import trainer
from library import inference


def main():
    # =========================================================================
    # 1. Configuration Overrides for Fast Baseline
    # =========================================================================
    # Override config to ensure execution finishes within time limits
    config.DEBUG = True
    config.MAX_TRAIN_SAMPLES = 200000  # Limit samples for speed
    config.NUM_EPOCHS = 3  # Reduce epochs
    config.BATCH_SIZE = 256  # Maximize throughput on A100

    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    print(f"--- Configuration ---")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Max Train Samples: {config.MAX_TRAIN_SAMPLES}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Device: {config.DEVICE}")

    # =========================================================================
    # 2. HFBB Statistics Initialization
    # =========================================================================
    print("\n--- Step 1: Building/Loading HFBB Stats ---")
    hfbb_engine = hfbb.HFBB()
    # Build stats (will load from cache if available, or compute from train.csv)
    hfbb_engine.build_stats(load_cached_data=True)

    # =========================================================================
    # 3. Data Preparation
    # =========================================================================
    print("\n--- Step 2: Preparing Data ---")
    # Load tokenizers and datasets
    # Note: dataset.prepare_data handles filtering for semiotic classes
    train_ds, val_ds, char_tok, bpe_tok = dataset.prepare_data(load_cached_data=True)

    # Enforce sample limit if loaded from cache (which might be full size)
    if len(train_ds) > config.MAX_TRAIN_SAMPLES:
        print(
            f"Enforcing sample limit: Slicing training dataset from {len(train_ds)} to {config.MAX_TRAIN_SAMPLES}"
        )
        train_ds.df = train_ds.df.head(config.MAX_TRAIN_SAMPLES)
        # Update internal lists used by __getitem__
        train_ds.befores = train_ds.df["before"].astype(str).tolist()
        train_ds.afters = train_ds.df["after"].astype(str).tolist()
        train_ds.prevs = train_ds.df["prev"].astype(str).tolist()
        train_ds.nexts = train_ds.df["next"].astype(str).tolist()

    # =========================================================================
    # 4. Model Training
    # =========================================================================
    print("\n--- Step 3: Training Semiotic Transformer ---")
    # Train the model (saves best checkpoint to config.BEST_MODEL_PATH)
    trained_model = trainer.train_model(train_ds, val_ds, char_tok, bpe_tok)

    # =========================================================================
    # 5. Full Validation (Hybrid Pipeline)
    # =========================================================================
    print("\n--- Step 4: Running Full Validation ---")
    # Load the full validation set (including non-semiotic tokens) to calculate the final metric
    val_df = pd.read_csv(config.VAL_FILE)

    # Preprocess validation data for context (Prev/Next)
    val_df["before"] = val_df["before"].fillna("").astype(str)
    val_df["after"] = val_df["after"].fillna("").astype(str)
    # Ensure sorting
    if "sentence_id" in val_df.columns and "token_id" in val_df.columns:
        val_df.sort_values(["sentence_id", "token_id"], inplace=True)

    val_df["prev"] = val_df["before"].shift(1).fillna("<START>")
    val_df.loc[val_df["token_id"] == 0, "prev"] = "<START>"
    val_df["next"] = val_df["before"].shift(-1).fillna("<END>")
    next_token_id = val_df["token_id"].shift(-1).fillna(0)
    val_df.loc[next_token_id == 0, "next"] = "<END>"

    # Initialize Normalizer (loads the trained model)
    normalizer = inference.HybridNormalizer(load_cached_data=True)

    # Run Hybrid Inference Logic on Validation Set
    predictions = [None] * len(val_df)
    transformer_indices = []

    prevs = val_df["prev"].tolist()
    currs = val_df["before"].tolist()
    nexts = val_df["next"].tolist()

    # A. HFBB Pass
    for i, (p, c, n) in enumerate(zip(prevs, currs, nexts)):
        pred, conf, level = hfbb_engine.query(p, c, n)

        # Logic matches inference.py
        if level in ["TRIGRAM", "BIGRAM_PREV", "BIGRAM_NEXT"]:
            predictions[i] = pred
        elif level == "UNIGRAM":
            if conf > config.CONFIDENCE_THRESHOLD:
                predictions[i] = pred
            else:
                if utils.is_semiotic(c):
                    transformer_indices.append(i)
                else:
                    predictions[i] = pred
        else:  # OOV
            if utils.is_semiotic(c):
                transformer_indices.append(i)
            else:
                predictions[i] = c  # Identity fallback

    # B. Transformer Pass
    if transformer_indices:
        print(f"Routing {len(transformer_indices)} tokens to Transformer...")
        df_trans = val_df.iloc[transformer_indices].copy()
        df_trans["orig_index"] = transformer_indices

        # Use TestDataset logic
        ds = inference.TestDataset(df_trans, char_tok, config.MAX_INPUT_LEN)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=(config.DEVICE == "cuda"),
        )

        for batch in loader:
            batch_preds = normalizer.predict_transformer_batch(batch)
            orig_idxs = batch["original_index"].tolist()

            for idx, txt in zip(orig_idxs, batch_preds):
                predictions[idx] = txt

    # Fill any remaining Nones (safety)
    for i in range(len(predictions)):
        if predictions[i] is None:
            predictions[i] = currs[i]

    # Calculate Metric
    val_df["prediction"] = predictions
    val_df["correct"] = val_df["prediction"] == val_df["after"]
    accuracy = val_df["correct"].mean()

    print(f"Final Validation Metric: {accuracy}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n--- Step 5: Failure Analysis ---")
    val_df["len_before"] = val_df["before"].apply(len)
    val_df["error"] = (~val_df["correct"]).astype(int)

    # Correlation: Input Length vs Error
    corr_len = val_df["len_before"].corr(val_df["error"])
    print(f"Correlation (Input Length vs Error): {corr_len:.6f}")

    # Error Rate by Class
    if "class" in val_df.columns:
        print("Error Rate by Class (Top 5):")
        class_errors = (
            val_df.groupby("class")["error"].mean().sort_values(ascending=False)
        )
        print(class_errors.head(5))

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    print("\n--- Step 6: Submission ---")
    threshold = 0.9784022349361615

    if accuracy > threshold:
        print(f"Validation accuracy {accuracy} > {threshold}. Generating submission...")
        # Call the inference module to generate the submission file
        normalizer.generate_submission()
    else:
        print(
            f"Validation accuracy {accuracy} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
