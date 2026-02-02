import pandas as pd
import numpy as np
import torch
import os
import sys
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Import library modules
from library.config import Config, set_seed
from library.utils import get_logger, get_device
from library.dataset import NormalizationDataset
from library.trainer import (
    run_training_pipeline,
    generate_submission,
    custom_collate_fn,
)
from library.model import TransformerTokenClassifier
from library.label_manager import LabelEngineer
from library.transformations import TransformationRegistry

# Setup logger
logger = get_logger("runfile")


def generate_predictions(model, dataset, batch_size=128):
    """
    Runs inference on a dataset and returns a DataFrame with ['id', 'pred_after'].
    Used for validation inference to calculate metrics and perform analysis.
    """
    device = get_device()
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
    )

    # Setup transformation logic to map predicted IDs back to strings
    label_engineer = LabelEngineer()
    label_engineer._load_or_create_label_encoder()
    id_to_name = {i: name for i, name in enumerate(label_engineer.label_names)}
    registry = TransformationRegistry()

    results = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            # Iterate through batch
            for i in range(len(batch["submission_ids"])):
                sample_preds = preds[i]
                word_ids = batch["word_ids"][i]
                raw_tokens = batch["raw_tokens"][i]
                submission_ids = batch["submission_ids"][i]

                previous_word_idx = None
                processed_indices = set()

                for seq_idx, word_idx in enumerate(word_ids):
                    if word_idx == -1:
                        continue

                    if word_idx != previous_word_idx:
                        # Process the first sub-token of each word
                        if word_idx < len(raw_tokens):
                            raw_token = raw_tokens[word_idx]
                            sub_id = submission_ids[word_idx]
                            pred_label_id = sample_preds[seq_idx]

                            # Apply transformation
                            label_name = id_to_name.get(pred_label_id, "TRANS_PLAIN")
                            normalized_text = registry.apply(label_name, raw_token)

                            results.append(
                                {"id": sub_id, "pred_after": normalized_text}
                            )
                            processed_indices.add(word_idx)

                    previous_word_idx = word_idx

                # Fallback for tokens not covered (e.g., due to truncation)
                for idx in range(len(raw_tokens)):
                    if idx not in processed_indices:
                        results.append(
                            {"id": submission_ids[idx], "pred_after": raw_tokens[idx]}
                        )

    return pd.DataFrame(results)


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = get_device()
    logger.info(f"Using device: {device}")

    # 2. Load Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. Prepare Datasets
    logger.info("Preparing datasets...")
    # Initialize datasets (handles caching internally)
    train_ds = NormalizationDataset("train", tokenizer)
    val_ds = NormalizationDataset("val", tokenizer)

    # Optimization: Limit training size for fast baseline execution
    # While the dataset class performs hard-negative sampling, the result might still be large.
    # We subset to 200,000 sentences to ensure completion within 2 hours.
    MAX_TRAIN_SAMPLES = 200000
    if len(train_ds) > MAX_TRAIN_SAMPLES:
        logger.info(
            f"Subsetting training set from {len(train_ds)} to {MAX_TRAIN_SAMPLES} for speed."
        )
        indices = np.random.choice(len(train_ds), MAX_TRAIN_SAMPLES, replace=False)
        train_ds = Subset(train_ds, indices)

    # 4. Training
    logger.info("Starting training pipeline...")
    # Run training for 1 epoch to satisfy "fast baseline" requirement
    trainer = run_training_pipeline(
        train_ds, val_ds, epochs=1, batch_size=Config.TRAIN_BATCH_SIZE
    )

    # 5. Validation & Failure Analysis
    logger.info("Performing validation and failure analysis...")

    # Load validation metadata to get ground truth 'after' and 'class'
    val_meta_path = Config.VAL_METADATA
    val_df = pd.read_csv(val_meta_path, keep_default_na=False)

    # Generate predictions on validation set
    val_preds_df = generate_predictions(
        trainer.model, val_ds, batch_size=Config.VAL_BATCH_SIZE
    )

    # Merge predictions with ground truth
    analysis_df = val_df.merge(val_preds_df, on="id", how="left")

    # Fill missing predictions with raw text (fallback)
    analysis_df["pred_after"] = analysis_df["pred_after"].fillna(analysis_df["before"])

    # Calculate Metric (Exact String Match)
    analysis_df["is_correct"] = analysis_df["after"] == analysis_df["pred_after"]
    accuracy = analysis_df["is_correct"].mean()

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    logger.info("Running failure analysis...")
    analysis_df["is_error"] = (~analysis_df["is_correct"]).astype(int)
    analysis_df["token_len"] = analysis_df["before"].str.len()

    # Correlation between error and token length
    corr_len = analysis_df["is_error"].corr(analysis_df["token_len"])
    print(f"Correlation (Error vs Token Length): {corr_len:.4f}")

    # Error rate by class (Top 10)
    logger.info("Error rate by class:")
    class_errors = (
        analysis_df.groupby("class")["is_error"].mean().sort_values(ascending=False)
    )
    print(class_errors.head(10))

    # 6. Submission
    THRESHOLD = 0.973229717044087
    if accuracy > THRESHOLD:
        logger.info(
            f"Validation metric {accuracy} > {THRESHOLD}. Generating submission..."
        )
        test_ds = NormalizationDataset("test", tokenizer)
        generate_submission(test_ds)
    else:
        logger.info(
            f"Validation metric {accuracy} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
