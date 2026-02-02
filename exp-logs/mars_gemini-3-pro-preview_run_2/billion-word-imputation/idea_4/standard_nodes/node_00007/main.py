import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# 1. Configuration Setup (Must be done before importing engine/pipeline)
# -------------------------------------------------------------------------
from library.config import Config

# Override Config for Fast Baseline
Config.DEBUG = True
Config.DEBUG_SIZE = 50000  # Train on 50k samples
Config.LOCATOR_EPOCHS = 1
Config.INFILLER_EPOCHS = 1
Config.LOCATOR_BATCH_SIZE = 256
Config.INFILLER_BATCH_SIZE = 128

# Now import the rest of the library
from library.utils import set_seed, compute_levenshtein_distance, setup_logger
from library.data_factory import load_data, TestDataset
from library.pipeline import run_training, generate_submission, Predictor

# Setup Logger
logger = setup_logger("runfile", os.path.join(Config.WORKING_DIR, "runfile.log"))


def create_synthetic_validation_set(df, sample_size=5000):
    """
    Creates a synthetic test set from validation data by removing one word.
    Returns a dataframe for the Predictor and the ground truth strings.
    """
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)

    ids = []
    masked_sentences = []
    original_sentences = []
    meta_info = []  # Store lengths, removed indices for failure analysis

    for idx, row in df.iterrows():
        text = row["sentence"]
        words = text.split()

        # Skip sentences too short to have a middle word removed
        if len(words) < 3:
            continue

        # Select random word index (not first or last)
        remove_idx = np.random.randint(1, len(words) - 1)

        # Find character span of the word to remove
        current_pos = 0
        word_spans = []
        for w in words:
            start = text.find(w, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(w)
            word_spans.append((start, end))
            current_pos = end

        start_char, end_char = word_spans[remove_idx]

        # Create masked text (remove word)
        # We simply concatenate the parts around the word.
        # Normalization (clean_text) in utils might handle spacing,
        # but here we just ensure we don't leave double spaces if possible.
        new_text_raw = text[:start_char] + text[end_char:]
        new_text = " ".join(new_text_raw.split())

        ids.append(row["id"])
        masked_sentences.append(new_text)
        original_sentences.append(text)

        meta_info.append(
            {
                "char_len": len(text),
                "word_count": len(words),
                "removed_idx_norm": remove_idx / len(words),
            }
        )

    df_synth = pd.DataFrame({"id": ids, "sentence": masked_sentences})
    return df_synth, original_sentences, pd.DataFrame(meta_info)


def main():
    set_seed(Config.SEED)
    logger.info("Starting Runfile Execution...")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    logger.info("Initiating Training Phase...")
    locator_path, infiller_path = run_training()

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    logger.info("Loading Validation Data for Evaluation...")
    # Load raw validation data
    df_val = load_data(
        Config.VAL_METADATA,
        "val_cache",
        load_cached_data=True,
        debug=True,
        debug_size=20000,
    )

    # Prepare synthetic test set
    logger.info("Creating synthetic validation set...")
    val_subset_df, ground_truth, meta_df = create_synthetic_validation_set(
        df_val, sample_size=5000
    )

    # Initialize Predictor
    predictor = Predictor(locator_path, infiller_path)

    # Create DataLoader for Predictor
    # We use the tokenizer from the predictor's locator model
    val_ds = TestDataset(val_subset_df, predictor.locator_tokenizer, Config.MAX_LEN)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    logger.info("Running Inference on Validation Set...")
    # Predict
    df_preds = predictor.predict(val_loader)

    # Align predictions with ground truth
    # df_preds has 'id' and 'sentence'. val_subset_df has 'id'.
    # We merge to ensure order, though predict preserves order usually.
    df_preds = df_preds.set_index("id")
    val_subset_df = val_subset_df.set_index("id")

    # Calculate Levenshtein Distance
    lev_distances = []

    # Iterate through the subset IDs to match ground truth order
    # ground_truth list corresponds to the order in val_subset_df before set_index
    # Let's reconstruct the order carefully
    ordered_preds = []
    ordered_gt = []

    # We iterate over the original list of IDs used to create ground_truth
    original_ids = val_subset_df.index.tolist()

    for i, uid in enumerate(original_ids):
        if uid in df_preds.index:
            pred_sent = df_preds.loc[uid, "sentence"]
            gt_sent = ground_truth[i]

            dist = compute_levenshtein_distance(pred_sent, gt_sent)
            lev_distances.append(dist)
        else:
            # Should not happen
            lev_distances.append(len(ground_truth[i]))  # Max penalty

    final_metric = np.mean(lev_distances)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate correlations
    # meta_df aligns with ground_truth list
    meta_df["lev_dist"] = lev_distances

    corr_len = np.corrcoef(meta_df["char_len"], meta_df["lev_dist"])[0, 1]
    corr_idx = np.corrcoef(meta_df["removed_idx_norm"], meta_df["lev_dist"])[0, 1]

    print("-" * 30)
    print("Failure Analysis (Correlation with Error):")
    print(f"Sentence Length vs Error: {corr_len:.4f}")
    print(f"Relative Missing Word Position vs Error: {corr_idx:.4f}")
    print("-" * 30)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 7.8943

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric ({final_metric:.4f}) is better than threshold ({THRESHOLD}). Generating Submission..."
        )
        generate_submission(locator_path, infiller_path, run_train=False)
    else:
        logger.info(
            f"Validation metric ({final_metric:.4f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
