import os
import pandas as pd
import numpy as np
import torch
from library.config import (
    ModelConfig,
    setup_environment,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
)
from library.text_utils import build_char_vocab, train_bpe_tokenizer
from library.hfbb_engine import HFBBModel
from library.dataset_factory import get_dataloaders
from library.trainer import train_model
from library.inference_engine import HybridNormalizer


def main():
    # 1. Configuration
    # We set num_epochs to 3 to ensure the pipeline completes within the 2-hour limit
    # while still allowing the Transformer to converge on the semiotic subset.
    config = ModelConfig(debug=False)
    config.num_epochs = 3

    # 2. Setup Environment
    setup_environment(seed=config.seed)

    print("=== Starting Runfile Execution ===")

    # 3. Load Metadata for Pre-processing
    print(f"Loading training metadata from {TRAIN_META_PATH}...")
    df_train = pd.read_csv(TRAIN_META_PATH)

    # Ensure 'id' column exists (required for some internal logic)
    if "id" not in df_train.columns:
        df_train["id"] = (
            df_train["sentence_id"].astype(str) + "_" + df_train["token_id"].astype(str)
        )

    # 4. Build Tokenizers
    # Tier 2 needs Character Tokenizer (Encoder) and BPE Tokenizer (Decoder)
    print("Building tokenizers...")
    char_tokenizer = build_char_vocab(df_train, vocab_size=config.char_vocab_size)
    bpe_tokenizer = train_bpe_tokenizer(df_train, vocab_size=config.bpe_vocab_size)

    # 5. Build HFBB Model (Tier 1)
    # This builds the statistical lookup tables on the FULL training set
    print("Building HFBB Model (Tier 1)...")
    hfbb = HFBBModel(config)
    hfbb.build(df_train, load_cached_data=True)

    # 6. Train Transformer (Tier 2)
    # The data loader factory handles the creation of the density-maximized (balanced) dataset
    print("Preparing Transformer training (Tier 2)...")
    train_loader, val_loader = get_dataloaders(
        config, char_tokenizer, bpe_tokenizer, load_cached_data=True
    )

    print("Starting Transformer training...")
    train_model(
        config,
        train_loader,
        val_loader,
        char_vocab_size=char_tokenizer.vocab_size,
        bpe_vocab_size=len(bpe_tokenizer),
    )

    # 7. Validation & Failure Analysis
    print("\n=== Running Validation Inference ===")

    # Load Validation Data
    df_val = pd.read_csv(VAL_META_PATH)
    if "id" not in df_val.columns:
        df_val["id"] = (
            df_val["sentence_id"].astype(str) + "_" + df_val["token_id"].astype(str)
        )

    # Initialize Hybrid Engine
    # This loads the best checkpoint and the HFBB cache
    engine = HybridNormalizer(config)
    engine.load_resources()

    # Predict on Validation Set
    # Note: This generates a submission file, but we use the returned DataFrame
    print("Generating validation predictions...")
    val_preds_df = engine.predict(df_val)

    # Merge with Ground Truth
    # df_val has 'after', val_preds_df has 'after' (prediction)
    print("Calculating metrics...")
    comparison_df = df_val.merge(val_preds_df, on="id", suffixes=("_true", "_pred"))

    # Ensure strings
    comparison_df["after_true"] = comparison_df["after_true"].fillna("").astype(str)
    comparison_df["after_pred"] = comparison_df["after_pred"].fillna("").astype(str)

    # Calculate Accuracy
    comparison_df["correct"] = (
        comparison_df["after_true"] == comparison_df["after_pred"]
    )
    accuracy = comparison_df["correct"].mean()

    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    comparison_df["error"] = (~comparison_df["correct"]).astype(int)
    comparison_df["len_before"] = (
        comparison_df["before"].fillna("").astype(str).apply(len)
    )

    # Calculate correlation between Error and Input Length
    corr_len = comparison_df["error"].corr(comparison_df["len_before"])
    print(f"Correlation (Error vs Input Length): {corr_len:.10f}")

    # 8. Submission Generation
    THRESHOLD = 0.9788071831831453

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy} > {THRESHOLD}. Generating final submission..."
        )

        # Load Test Data
        df_test = pd.read_csv(TEST_META_PATH)

        # Predict
        engine.predict(df_test)
        print("Submission generated successfully at ./submission/submission.csv")
    else:
        print(f"\nValidation accuracy {accuracy} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
