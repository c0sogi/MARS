import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
import library.config as config
import library.trainer as trainer
import library.utils as utils
import library.vocabulary as vocab_module
import library.models as models
import library.data_loader as data_loader
import library.inference as inference


def main():
    # =========================================================================
    # 1. CONFIGURATION & OPTIMIZATION
    # =========================================================================
    # Monkey-patch config for fast baseline execution
    config.TAGGER_EPOCHS = 1
    config.SEQ2SEQ_EPOCHS = 1
    trainer.TAGGER_EPOCHS = 1
    trainer.SEQ2SEQ_EPOCHS = 1

    # Limits for training steps
    MAX_TAGGER_BATCHES = 2000
    MAX_SEQ2SEQ_BATCHES = 500

    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. DATA LOADING
    # =========================================================================
    print("Loading Vocabularies...")
    vocab_words, vocab_chars, vocab_classes = vocab_module.build_vocabularies(
        df_train=None, load_cached_data=True
    )

    print("Loading DataLoaders...")
    # This handles signal-dense filtering automatically
    loaders = data_loader.get_dataloaders(
        vocab_words, vocab_chars, vocab_classes, load_cached_data=True
    )

    # Wrapper to limit training iterations
    class LimitedLoader:
        def __init__(self, loader, limit):
            self.loader = loader
            self.limit = limit
            self.dataset = loader.dataset

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.limit:
                    break
                yield batch

        def __len__(self):
            return min(len(self.loader), self.limit)

    train_loader_tagger = LimitedLoader(loaders["tagger_train"], MAX_TAGGER_BATCHES)
    val_loader_tagger = loaders["tagger_val"]  # Full validation for monitoring

    train_loader_seq2seq = LimitedLoader(loaders["seq2seq_train"], MAX_SEQ2SEQ_BATCHES)
    val_loader_seq2seq = loaders["seq2seq_val"]

    # =========================================================================
    # 3. MODEL TRAINING
    # =========================================================================
    print("Initializing Models...")
    tagger = models.RegexBiLSTMTagger(
        len(vocab_words), len(vocab_chars), len(vocab_classes)
    ).to(device)

    seq2seq = models.CharLSTMSeq2Seq(
        len(vocab_chars),
        len(vocab_classes),
        vocab_chars.token2id[config.SOS_TOKEN],
        vocab_chars.token2id[config.EOS_TOKEN],
    ).to(device)

    print("Training Tagger...")
    trainer.train_tagger(
        tagger, train_loader_tagger, val_loader_tagger, len(vocab_classes)
    )

    print("Training Seq2Seq...")
    trainer.train_seq2seq(
        seq2seq, train_loader_seq2seq, val_loader_seq2seq, len(vocab_chars)
    )

    # =========================================================================
    # 4. VALIDATION (FULL INFERENCE)
    # =========================================================================
    print("Running Full Validation Inference...")

    # Trick: Use InferencePipeline but point it to the validation data
    original_test_path = config.TEST_DATA_PATH
    original_sub_path = config.SUBMISSION_PATH

    # Temporary paths
    val_pred_path = os.path.join(config.WORKING_DIR, "val_prediction.csv")
    config.TEST_DATA_PATH = config.VAL_DATA_PATH
    config.SUBMISSION_PATH = val_pred_path

    # Run inference
    # Note: InferencePipeline reloads models from checkpoints, which we just saved.
    pipeline = inference.InferencePipeline()
    pipeline.run()

    # Restore paths
    config.TEST_DATA_PATH = original_test_path
    config.SUBMISSION_PATH = original_sub_path

    # Calculate Metric
    print("Calculating Validation Metric...")
    df_val_true = pd.read_csv(config.VAL_DATA_PATH, dtype=str, keep_default_na=False)
    df_val_pred = pd.read_csv(val_pred_path, dtype=str, keep_default_na=False)

    # Merge on ID
    df_merged = pd.merge(df_val_true, df_val_pred, on="id", suffixes=("_true", "_pred"))

    # Handle NaNs if any (though keep_default_na=False should prevent this)
    df_merged["after_true"] = df_merged["after_true"].fillna("")
    df_merged["after_pred"] = df_merged["after_pred"].fillna("")

    # Accuracy
    correct_mask = df_merged["after_true"] == df_merged["after_pred"]
    accuracy = correct_mask.mean()

    print(f"Final Validation Metric: {accuracy}")

    # =========================================================================
    # 5. FAILURE ANALYSIS
    # =========================================================================
    print("\nFailure Analysis:")
    df_errors = df_merged[~correct_mask].copy()

    if len(df_errors) > 0:
        # 1. Correlation with Input Length
        df_merged["input_len"] = df_merged["before"].str.len()
        df_merged["is_error"] = (~correct_mask).astype(int)

        corr = df_merged["input_len"].corr(df_merged["is_error"])
        print(f"Correlation between Input Length and Error: {corr:.4f}")

        # 2. Error Rates by Class
        print("\nTop 5 Classes with most errors:")
        print(df_errors["class"].value_counts().head(5).to_string())

        print("\nError Rate by Class (Top 5):")
        class_counts = df_merged["class"].value_counts()
        error_counts = df_errors["class"].value_counts()
        for cls in class_counts.head(5).index:
            total = class_counts[cls]
            err = error_counts.get(cls, 0)
            print(f"{cls}: {err}/{total} ({err/total:.2%})")
    else:
        print("No errors found in validation set.")

    # =========================================================================
    # 6. SUBMISSION
    # =========================================================================
    THRESHOLD = 0.9949142925818993

    if accuracy > THRESHOLD:
        print(f"\nValidation metric {accuracy} > {THRESHOLD}. Generating submission...")
        # Run inference on actual test set
        pipeline = inference.InferencePipeline()
        pipeline.run()
    else:
        print(f"\nValidation metric {accuracy} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
