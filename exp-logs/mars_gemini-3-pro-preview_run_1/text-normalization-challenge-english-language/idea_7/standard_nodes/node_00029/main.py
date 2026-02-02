import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import (
    prepare_data,
    collate_tagger,
    collate_seq2seq,
    collate_submission,
    SubmissionDataset,
)
from library.models import AttentionBiLSTMTagger, TransformerSeq2Seq
from library.engine import (
    TaggerEngine,
    Seq2SeqEngine,
    get_class_weights,
    generate_submission,
)


def validate_full_pipeline(
    tagger_engine, seq2seq_engine, val_loader, kb, vocab, val_metadata_path
):
    """
    Runs the full inference pipeline on the validation set and calculates exact match accuracy.
    """
    print("Running Full Pipeline Validation...")

    # 1. Redirect submission path to a temp file
    original_sub_path = Config.SUBMISSION_PATH
    temp_sub_path = os.path.join(Config.WORKING_DIR, "val_predictions.csv")
    Config.SUBMISSION_PATH = temp_sub_path

    # 2. Generate predictions using the standard submission logic
    try:
        generate_submission(tagger_engine, seq2seq_engine, val_loader, kb, vocab)
    finally:
        # Restore path
        Config.SUBMISSION_PATH = original_sub_path

    # 3. Load Predictions
    df_pred = pd.read_csv(temp_sub_path, dtype=str, keep_default_na=False)

    # 4. Load Ground Truth
    # We read the raw validation metadata file to get the 'after' column
    df_gt = pd.read_csv(val_metadata_path, dtype=str, keep_default_na=False)

    # Ensure IDs match
    # The submission generation preserves order and IDs, but let's merge on ID to be safe
    df_merged = pd.merge(df_gt, df_pred, on="id", suffixes=("_true", "_pred"))

    # 5. Calculate Accuracy
    # Exact string match required
    correct = (df_merged["after_true"] == df_merged["after_pred"]).sum()
    total = len(df_merged)
    accuracy = correct / total

    return accuracy, df_merged


def perform_failure_analysis(df_results, vocab):
    """
    Correlates error magnitude with input features.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate Error (1 if wrong, 0 if correct)
    df_results["is_error"] = (
        df_results["after_true"] != df_results["after_pred"]
    ).astype(int)

    # 2. Extract Features
    # Token Length
    df_results["token_len"] = df_results["before"].astype(str).apply(len)

    # Class ID (Map class string to ID)
    # df_results has 'class' column from val.csv
    df_results["class_id"] = (
        df_results["class"].map(vocab.class2id).fillna(-1).astype(int)
    )

    # 3. Calculate Correlations
    # We use numpy for simple correlation
    features = ["token_len", "class_id"]
    correlations = {}

    for feat in features:
        if feat in df_results.columns:
            # Drop NaNs if any
            valid = df_results[[feat, "is_error"]].dropna()
            if len(valid) > 0:
                corr = np.corrcoef(valid[feat], valid["is_error"])[0, 1]
                correlations[feat] = corr
            else:
                correlations[feat] = 0.0

    print("Correlation between Error and Input Features:")
    for feat, corr in correlations.items():
        print(f"   {feat}: {corr:.4f}")

    # Additional Insight: Error rate by class
    print("\nError Rate by Class (Top 5 worst):")
    class_errors = (
        df_results.groupby("class")["is_error"].mean().sort_values(ascending=False)
    )
    print(class_errors.head(5).to_string())


def main():
    # 1. Setup
    Config.setup()
    seed_everything()
    print(f"Device: {Config.DEVICE}")

    # 2. Data Preparation
    # load_cached_data=True to use existing artifacts if available
    vocab, kb, train_d_tag, val_d_tag, train_d_s2s, val_d_s2s, test_d = prepare_data(
        load_cached_data=True
    )

    # Create DataLoaders
    # Tagger Loaders
    train_loader_tag = DataLoader(
        train_d_tag,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_tagger,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader_tag = DataLoader(
        val_d_tag,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_tagger,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Seq2Seq Loaders
    train_loader_s2s = DataLoader(
        train_d_s2s,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_seq2seq,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader_s2s = DataLoader(
        val_d_s2s,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_seq2seq,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    num_tokens = len(vocab.token2id)
    num_chars = len(vocab.char2id)
    num_classes = len(vocab.class2id)

    print(
        f"Initializing Models | Tokens: {num_tokens}, Chars: {num_chars}, Classes: {num_classes}"
    )

    tagger_model = AttentionBiLSTMTagger(num_tokens, num_chars, num_classes)
    seq2seq_model = TransformerSeq2Seq(num_chars, num_classes)

    # 4. Train Tagger
    print("\n=== Training Tagger ===")
    class_weights = get_class_weights(train_d_tag, vocab, Config.DEVICE)
    tagger_engine = TaggerEngine(
        tagger_model, Config.DEVICE, train_loader_tag, val_loader_tag, class_weights
    )

    # Train
    tagger_engine.fit(epochs=Config.EPOCHS)

    # 5. Train Seq2Seq
    print("\n=== Training Seq2Seq Fallback ===")
    seq2seq_engine = Seq2SeqEngine(
        seq2seq_model, Config.DEVICE, train_loader_s2s, val_loader_s2s
    )

    # Train
    seq2seq_engine.fit(epochs=Config.EPOCHS)

    # 6. Full Pipeline Validation
    # Create a submission-style loader for the validation set
    # val_d_tag.data is the grouped dataframe (sentence level)
    val_grouped_df = val_d_tag.data
    val_submission_ds = SubmissionDataset(val_grouped_df, vocab)
    val_submission_loader = DataLoader(
        val_submission_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_submission,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_metric, df_val_results = validate_full_pipeline(
        tagger_engine, seq2seq_engine, val_submission_loader, kb, vocab, Config.VAL_FILE
    )

    print(f"Final Validation Metric: {val_metric:.16f}")

    # 7. Failure Analysis
    perform_failure_analysis(df_val_results, vocab)

    # 8. Submission
    THRESHOLD = 0.9949142925818993
    if val_metric > THRESHOLD:
        print(
            f"\nMetric ({val_metric:.6f}) > Threshold ({THRESHOLD:.6f}). Generating Submission..."
        )

        test_loader = DataLoader(
            test_d,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_submission,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(tagger_engine, seq2seq_engine, test_loader, kb, vocab)
    else:
        print(
            f"\nMetric ({val_metric:.6f}) did not meet threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
