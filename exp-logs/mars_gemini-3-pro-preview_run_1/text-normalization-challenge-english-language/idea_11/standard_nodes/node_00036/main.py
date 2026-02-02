import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import logging
import csv

# 1. Patch tqdm to be silent (Requirement: Do not print progress bars)
import tqdm.auto


def no_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.auto.tqdm = no_tqdm
# Also patch the main tqdm module just in case
import tqdm

tqdm.tqdm = no_tqdm

# 2. Imports from library
from library.config import Config
from library.utils import set_seed, setup_logger, calculate_accuracy
from library.data_loader import get_dataloaders, KnowledgeBase
from library.engine import train_tagger, train_seq2seq
from library.inference import InferencePipeline, run_inference
from library.vocab import VocabManager
from library.features import GlobalPriorManager


def main():
    # Setup
    set_seed(Config.SEED)
    # Setup logger but suppress library loggers to minimize output
    logger = setup_logger("main_script")
    logging.getLogger("features").setLevel(logging.ERROR)
    logging.getLogger("vocab").setLevel(logging.ERROR)
    logging.getLogger("data_loader").setLevel(logging.ERROR)
    logging.getLogger("engine").setLevel(logging.ERROR)
    logging.getLogger("inference").setLevel(logging.ERROR)

    logger.info("Starting Fast Baseline Script")

    # Define paths
    FULL_TRAIN_PATH = Config.TRAIN_FILE
    SAMPLED_TRAIN_PATH = os.path.join(Config.WORK_DIR, "train_sampled.csv")

    # ---------------------------------------------------------
    # Step 1: Build Artifacts (Vocab, KB, Priors) from FULL Data
    # ---------------------------------------------------------
    # We use the full dataset to build the Knowledge Base and Vocabularies
    # to ensure maximum coverage, even if we train on a subset.
    logger.info("Loading full training data for artifact generation...")
    df_full = pd.read_csv(FULL_TRAIN_PATH)

    # Build Vocabs
    logger.info("Building Vocabularies from full data...")
    vocab_manager = VocabManager()
    # Force build from scratch using current Config.TRAIN_FILE (which is full)
    vocab_manager.build_or_load(load_cached_data=False)

    # Build Priors
    logger.info("Building Global Priors from full data...")
    prior_manager = GlobalPriorManager()
    prior_manager.build_or_load(df_full, load_cached_data=False)

    # Build Knowledge Base
    logger.info("Building Knowledge Base from full data...")
    kb = KnowledgeBase()
    kb.build(df_full, save=True)

    # ---------------------------------------------------------
    # Step 2: Prepare Sampled Data for Fast Training
    # ---------------------------------------------------------
    logger.info("Creating sampled training set...")
    # Sample 10% of sentences to preserve context and ensure speed
    unique_sents = df_full["sentence_id"].unique()
    np.random.seed(Config.SEED)
    # Sample ~10%
    sampled_sents = np.random.choice(
        unique_sents, size=int(len(unique_sents) * 0.1), replace=False
    )
    df_sampled = df_full[df_full["sentence_id"].isin(sampled_sents)].copy()

    df_sampled.to_csv(SAMPLED_TRAIN_PATH, index=False)
    logger.info(
        f"Sampled train data saved to {SAMPLED_TRAIN_PATH} ({len(df_sampled)} rows)"
    )

    # Free memory
    del df_full, df_sampled
    import gc

    gc.collect()

    # ---------------------------------------------------------
    # Step 3: Configure for Training
    # ---------------------------------------------------------
    # Override Config to point to sampled data
    Config.TRAIN_FILE = SAMPLED_TRAIN_PATH
    # Reduce epochs for fast baseline
    Config.EPOCHS = 3
    Config.BATCH_SIZE = 256

    # Get Dataloaders
    # load_cached_data=True will load the artifacts we just built (Vocab, KB, Priors)
    # but will create new dataset caches from the SAMPLED_TRAIN_PATH
    logger.info("Initializing Dataloaders...")
    dataloaders = get_dataloaders(load_cached_data=True)

    # ---------------------------------------------------------
    # Step 4: Train Models
    # ---------------------------------------------------------
    logger.info("Training Tagger...")
    # Train Tagger
    train_tagger(dataloaders, vocab_manager, prior_manager, load_cached_data=True)

    logger.info("Training Seq2Seq...")
    # Train Seq2Seq
    train_seq2seq(dataloaders, vocab_manager)

    # ---------------------------------------------------------
    # Step 5: Validation
    # ---------------------------------------------------------
    logger.info("Running Validation on full validation set...")

    # Initialize Pipeline (loads trained models from disk)
    pipeline = InferencePipeline(load_cached_data=True)

    # Predict on Validation File (Config.VAL_FILE is the full original validation file)
    val_preds = pipeline.predict(test_file=Config.VAL_FILE)

    # Load Ground Truth
    df_val = pd.read_csv(Config.VAL_FILE)
    # Ensure ID mapping matches submission format
    if "id" not in df_val.columns:
        df_val["id"] = (
            df_val["sentence_id"].astype(str) + "_" + df_val["token_id"].astype(str)
        )

    # Convert preds to dictionary for mapping
    pred_map = dict(val_preds)

    # Map predictions to validation dataframe
    df_val["pred"] = df_val["id"].map(pred_map).fillna("")

    # Calculate Exact Match Accuracy
    df_val["correct"] = df_val["pred"] == df_val["after"].astype(str)
    accuracy = df_val["correct"].mean()

    # Print Metric in required format
    print(f"Final Validation Metric: {accuracy}")

    # ---------------------------------------------------------
    # Step 6: Failure Analysis
    # ---------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate Correlation between Error and Input Length
    df_val["error"] = (~df_val["correct"]).astype(int)
    df_val["len_before"] = df_val["before"].astype(str).apply(len)

    # Calculate correlation
    if len(df_val) > 1 and df_val["error"].std() > 0:
        corr = np.corrcoef(df_val["error"], df_val["len_before"])[0, 1]
        print(f"Correlation between Error and Input Length: {corr:.4f}")
    else:
        print("Correlation between Error and Input Length: Undefined (no variance)")

    # Print Error Rate by Class
    print("Error Rate by Class (Top 5):")
    class_errors = df_val.groupby("class")["error"].mean().sort_values(ascending=False)
    print(class_errors.head(5).to_string())

    # ---------------------------------------------------------
    # Step 7: Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.9949142925818993

    if accuracy > THRESHOLD:
        logger.info(
            f"Validation accuracy {accuracy} > {THRESHOLD}. Generating submission..."
        )
        # Run inference on Test Set
        run_inference(test_file=Config.TEST_FILE, output_path=Config.SUBMISSION_PATH)
    else:
        logger.info(
            f"Validation accuracy {accuracy} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
