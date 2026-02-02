import os
import sys
import pandas as pd
import numpy as np
import torch
import nltk
from nltk.metrics import edit_distance
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import get_dataloaders, DynamicMaskingDataset
from library.trainer import Trainer
from library.predictor import Predictor


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for Fast Baseline Execution
    # Reducing sample sizes to ensure completion within 2 hours
    Config.MAX_TRAIN_SAMPLES = 100_000
    Config.VAL_SAMPLES = 2_000
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 64
    Config.VAL_BATCH_SIZE = 64
    Config.TEST_BATCH_SIZE = 128

    # Setup Logger
    logger = setup_logger("Runfile", os.path.join(Config.OUTPUT_DIR, "runfile.log"))
    logger.info("Starting end-to-end pipeline execution...")
    logger.info(
        f"Configuration: Train Samples={Config.MAX_TRAIN_SAMPLES}, Val Samples={Config.VAL_SAMPLES}, Epochs={Config.EPOCHS}"
    )

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # load_cached_data=False ensures we generate the new smaller subsets defined above
    logger.info("Loading data...")
    train_loader, val_loader_loss, test_loader = get_dataloaders(
        load_cached_data=False, debug=False
    )

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    logger.info("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader_loss)

    logger.info("Starting Training Phase...")
    trainer.train()

    # ---------------------------------------------------------
    # 4. Validation & Metric Calculation
    # ---------------------------------------------------------
    logger.info("Starting Validation Phase (Levenshtein Metric)...")

    # Load validation data and sample it for evaluation
    df_val_full = pd.read_parquet(Config.VAL_PATH)
    df_val_eval = df_val_full.sample(
        n=Config.VAL_SAMPLES, random_state=Config.SEED
    ).reset_index(drop=True)

    # Simulate Test Scenario: Remove one word per sentence
    val_ids = []
    val_input_sentences = []
    val_original_sentences = []

    rng = np.random.RandomState(Config.SEED)

    for idx, row in df_val_eval.iterrows():
        original_sent = row["sentence"]
        words = original_sent.split()

        # Skip sentences that are too short to have a middle gap
        if len(words) < 3:
            continue

        # Pick a word to remove (uniformly random, excluding first and last)
        gap_idx = rng.randint(1, len(words) - 1)

        # Create input sentence by removing the word
        # We join with space to approximate the input format
        pre_gap = words[:gap_idx]
        post_gap = words[gap_idx + 1 :]
        input_sent = " ".join(pre_gap + post_gap)

        val_ids.append(idx)
        val_input_sentences.append(input_sent)
        val_original_sentences.append(original_sent)

    # Create DataFrame for Predictor
    df_val_simulated = pd.DataFrame({"id": val_ids, "sentence": val_input_sentences})

    # Initialize Predictor (loads best model from checkpoint)
    predictor = Predictor()

    # Create DataLoader for simulated validation
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    val_sim_dataset = DynamicMaskingDataset(df_val_simulated, tokenizer, mode="test")
    val_sim_loader = DataLoader(
        val_sim_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference
    logger.info(f"Running inference on {len(df_val_simulated)} validation samples...")
    predictions = []

    # Predictor.predict_batch returns list of (id, predicted_sentence)
    # We iterate manually to avoid re-initializing the predictor or logger inside generate_submission
    for batch in val_sim_loader:
        batch_preds = predictor.predict_batch(batch)
        predictions.extend(batch_preds)

    # Compute Levenshtein Distance
    logger.info("Computing Levenshtein distances...")
    pred_map = {p_id: p_sent for p_id, p_sent in predictions}

    total_distance = 0
    count = 0
    distances = []
    lengths = []
    word_counts = []

    for i, original_sent in enumerate(val_original_sentences):
        p_id = val_ids[i]
        if p_id in pred_map:
            predicted_sent = pred_map[p_id]

            # Compute distance
            dist = edit_distance(original_sent, predicted_sent)

            total_distance += dist
            count += 1

            # Store for failure analysis
            distances.append(dist)
            lengths.append(len(original_sent))
            word_counts.append(len(original_sent.split()))

    mean_levenshtein = total_distance / count if count > 0 else float("inf")

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {mean_levenshtein}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    logger.info("Performing Failure Analysis...")
    if count > 0:
        # Calculate correlations
        corr_len = np.corrcoef(lengths, distances)[0, 1]
        corr_words = np.corrcoef(word_counts, distances)[0, 1]

        print(f"Correlation (Error vs Char Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Word Count): {corr_words:.4f}")

        logger.info(f"Failure Analysis - Correlation with Char Length: {corr_len:.4f}")
        logger.info(f"Failure Analysis - Correlation with Word Count: {corr_words:.4f}")
    else:
        logger.warning("No predictions generated for failure analysis.")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 12.211422845691382

    if mean_levenshtein < THRESHOLD:
        logger.info(
            f"Metric {mean_levenshtein} < Threshold {THRESHOLD}. Generating submission..."
        )
        predictor.generate_submission(test_loader)
    else:
        logger.info(
            f"Metric {mean_levenshtein} >= Threshold {THRESHOLD}. Submission skipped."
        )

    logger.info("Pipeline execution complete.")

    # Reprint the metric at the end to ensure it is visible to the grading system
    print(f"Final Validation Metric: {mean_levenshtein}")


if __name__ == "__main__":
    main()
