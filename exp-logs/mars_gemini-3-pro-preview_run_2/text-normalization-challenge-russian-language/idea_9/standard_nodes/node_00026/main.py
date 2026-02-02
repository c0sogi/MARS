import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pointbiserialr

# Import from provided library
from library.config import Config
from library.utils import set_seed, is_semiotic
from library.train import train_model
from library.inference import CascadePredictor
from library.dataset import ContextWindowDataset
from library.hfbb import HFBBModel


def evaluate_cascade(df, predictor):
    """
    Evaluates the cascade model on a dataframe containing 'before' and 'after' columns.
    Replicates the logic from CascadePredictor.generate_submission but for validation.
    """
    print("Evaluating Cascade on validation set...")

    # Prepare context
    # Ensure string types
    df["before"] = df["before"].fillna("").astype(str)
    df["after"] = df["after"].fillna("").astype(str)

    # Shift for context respecting sentence boundaries
    df["prev"] = df["before"].shift(1).fillna("<start>")
    mask_start = df["sentence_id"] != df["sentence_id"].shift(1)
    df.loc[mask_start, "prev"] = "<start>"

    df["next"] = df["before"].shift(-1).fillna("<end>")
    mask_end = df["sentence_id"] != df["sentence_id"].shift(-1)
    df.loc[mask_end, "next"] = "<end>"

    # Initialize predictions
    predictions = [None] * len(df)
    tier2_indices = []

    # Tier 1: HFBB
    prev_arr = df["prev"].values
    curr_arr = df["before"].values
    next_arr = df["next"].values

    trigram_map = predictor.hfbb_model.trigram_map
    bigram_prev_map = predictor.hfbb_model.bigram_prev_map
    bigram_next_map = predictor.hfbb_model.bigram_next_map
    unigram_map = predictor.hfbb_model.unigram_map

    confidence_threshold = Config.CONFIDENCE_THRESHOLD

    # Store confidence for failure analysis
    confidences = np.zeros(len(df))

    for idx, (p, c, n) in enumerate(zip(prev_arr, curr_arr, next_arr)):
        pred = None
        source = "none"
        conf = 0.0

        if (p, c, n) in trigram_map:
            pred = trigram_map[(p, c, n)]
            source = "trigram"
            conf = 1.0
        elif (p, c) in bigram_prev_map:
            pred = bigram_prev_map[(p, c)]
            source = "bigram_prev"
            conf = 1.0
        elif (c, n) in bigram_next_map:
            pred = bigram_next_map[(c, n)]
            source = "bigram_next"
            conf = 1.0
        elif c in unigram_map:
            pred, conf = unigram_map[c]
            source = "unigram"

        confidences[idx] = conf

        route_to_tier2 = False

        if source in ["trigram", "bigram_prev", "bigram_next"]:
            predictions[idx] = pred
        elif source == "unigram":
            if conf > confidence_threshold:
                predictions[idx] = pred
            else:
                if is_semiotic(c):
                    route_to_tier2 = True
                else:
                    predictions[idx] = pred
        else:
            if is_semiotic(c):
                route_to_tier2 = True
            else:
                predictions[idx] = c

        if route_to_tier2:
            tier2_indices.append(idx)

    # Tier 2: Transformer
    if tier2_indices:
        print(f"Routing {len(tier2_indices)} tokens to Tier 2 Transformer...")
        dataset = ContextWindowDataset(
            df=df,
            indices=np.array(tier2_indices),
            char_tokenizer=predictor.char_tokenizer,
            target_tokenizer=None,
            mode="test",
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        with torch.no_grad():
            for batch in loader:
                src = batch["src_ids"].to(predictor.device)
                original_indices = batch["id"].numpy()

                generated_ids = predictor._greedy_decode(
                    src, max_len=Config.MAX_LEN_SUBWORD
                )
                generated_lists = generated_ids.cpu().tolist()

                decoded_strs = [
                    predictor.target_tokenizer.decode(ids, remove_special_tokens=True)
                    for ids in generated_lists
                ]

                for i, original_idx in enumerate(original_indices):
                    predictions[original_idx] = decoded_strs[i]

    # Fill any remaining Nones (safety)
    for i, p in enumerate(predictions):
        if p is None:
            predictions[i] = df.iloc[i]["before"]

    # Calculate Accuracy
    actuals = df["after"].values
    correct_mask = np.array(predictions) == actuals
    accuracy = np.mean(correct_mask)

    return accuracy, correct_mask, confidences


def perform_failure_analysis(df, correct_mask, confidences):
    """
    Analyzes failure patterns.
    """
    print("\n=== Failure Analysis ===")

    # 0 is error, 1 is correct. We want correlation with Error (1 if error, 0 if correct)
    error_mask = (~correct_mask).astype(int)

    # 1. Token Length
    token_lengths = df["before"].astype(str).apply(len).values
    corr_len, _ = pointbiserialr(error_mask, token_lengths)
    print(f"Correlation (Error vs Token Length): {corr_len:.4f}")

    # 2. Is Semiotic
    is_semiotic_vals = df["before"].astype(str).apply(is_semiotic).astype(int).values
    corr_sem, _ = pointbiserialr(error_mask, is_semiotic_vals)
    print(f"Correlation (Error vs Is_Semiotic): {corr_sem:.4f}")

    # 3. HFBB Confidence
    corr_conf, _ = pointbiserialr(error_mask, confidences)
    print(f"Correlation (Error vs Unigram Confidence): {corr_conf:.4f}")

    # 4. Class Breakdown (if available)
    if "class" in df.columns:
        print("\nError Rate by Class:")
        df["is_error"] = error_mask
        class_errors = (
            df.groupby("class")["is_error"].mean().sort_values(ascending=False)
        )
        print(class_errors.head(10))


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline
    Config.EPOCHS = 3
    Config.DEBUG = True
    Config.DEBUG_SIZE = 150000  # Limit training data for speed

    # Ensure reproducibility
    set_seed(Config.SEED)
    Config.setup()

    print("Configuration set for fast baseline execution.")
    print(
        f"Epochs: {Config.EPOCHS}, Debug Mode: {Config.DEBUG}, Debug Size: {Config.DEBUG_SIZE}"
    )

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n--- Starting Training Phase ---")
    # We use load_cached_data=False to ensure the datasets are regenerated
    # with the DEBUG/Curriculum settings defined above.
    model = train_model(load_cached_data=False)

    # ==========================================
    # 3. Validation
    # ==========================================
    print("\n--- Starting Validation Phase ---")

    # Load the FULL validation set (ignoring Config.DEBUG truncation for validation metric)
    print(f"Loading full validation set from {Config.VAL_DATA_PATH}...")
    df_val = pd.read_csv(Config.VAL_DATA_PATH)

    # Initialize Predictor (loads the trained model and HFBB)
    predictor = CascadePredictor()
    predictor.load_resources()

    # Evaluate
    accuracy, correct_mask, confidences = evaluate_cascade(df_val, predictor)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    perform_failure_analysis(df_val, correct_mask, confidences)

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 0.9784022349361615
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
