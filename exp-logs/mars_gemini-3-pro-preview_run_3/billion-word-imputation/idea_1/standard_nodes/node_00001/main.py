import os
import sys
import pandas as pd
import numpy as np
import torch
import random
import nltk
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.inference import InferencePipeline
from library.utils import set_seed, setup_logger
from library.models import LocatorNetwork, FillerNetwork


class InMemoryDataset(Dataset):
    """
    Custom dataset for running inference on validation data in memory.
    """

    def __init__(self, data, tokenizer, max_len):
        self.data = data  # List of dicts: {'id': int, 'sentence': str}
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence = row["sentence"]

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "id": row["id"],
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def prepare_validation_data(sample_size=2000):
    """
    Loads validation data, creates synthetic gaps, and returns:
    - val_loader: DataLoader for inference
    - ground_truth: Dict mapping id -> original_sentence
    - metadata: DataFrame with 'id', 'length', etc. for analysis
    """
    print("Preparing validation data...")
    df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Sample for speed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)

    input_data = []
    ground_truth = {}
    meta_data = []

    for idx, row in df.iterrows():
        original_sentence = row["sentence"]
        words = original_sentence.split()

        # Skip sentences too short to have a middle word removed
        if len(words) < 3:
            continue

        # Create gap (mimic test set: never first or last)
        remove_idx = random.randint(1, len(words) - 2)

        # Construct gapped sentence
        gapped_words = words[:remove_idx] + words[remove_idx + 1 :]
        gapped_sentence = " ".join(gapped_words)

        # Store
        uid = idx  # Use index as ID
        input_data.append({"id": uid, "sentence": gapped_sentence})
        ground_truth[uid] = original_sentence
        meta_data.append({"id": uid, "length": len(original_sentence)})

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
    dataset = InMemoryDataset(input_data, tokenizer, Config.MAX_LEN)
    loader = DataLoader(
        dataset,
        batch_size=Config.FILLER_PARAMS["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    return loader, ground_truth, pd.DataFrame(meta_data)


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for Fast Baseline
    Config.DEBUG = True  # Use subset for training

    # Reduce epochs to 1 for speed
    Config.LOCATOR_PARAMS["epochs"] = 1
    Config.FILLER_PARAMS["epochs"] = 1

    # Increase batch size for A100 efficiency
    Config.LOCATOR_PARAMS["batch_size"] = 256
    Config.FILLER_PARAMS["batch_size"] = 128

    set_seed(Config.SEED)
    logger = setup_logger("RunFile")

    logger.info("Configuration set for fast baseline.")

    # ---------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------
    logger.info("Initializing Trainer...")
    trainer = Trainer(debug=Config.DEBUG)

    # Train Locator
    trainer.train_locator()

    # Train Filler
    trainer.train_filler()

    # ---------------------------------------------------------
    # 3. Validation & Evaluation
    # ---------------------------------------------------------
    logger.info("Starting Validation Evaluation...")

    # Prepare validation data
    val_loader, ground_truth, meta_df = prepare_validation_data(sample_size=2000)

    # Initialize Inference Pipeline (reusing its model loading and predict logic)
    pipeline = InferencePipeline(
        debug=False
    )  # debug=False here just means don't reload data from disk
    pipeline.load_models()

    # Run Inference
    # pipeline.predict expects a loader yielding 'id', 'input_ids', 'attention_mask'
    predictions = pipeline.predict(val_loader)

    # Compute Metrics
    levenshtein_distances = []
    ids_processed = []

    for pred in predictions:
        uid = pred["id"]
        pred_sent = pred["sentence"]
        true_sent = ground_truth[uid]

        # Calculate Levenshtein Distance
        dist = nltk.edit_distance(pred_sent, true_sent)
        levenshtein_distances.append(dist)
        ids_processed.append(uid)

    mean_levenshtein = np.mean(levenshtein_distances)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {mean_levenshtein}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Merge metrics with metadata
    results_df = pd.DataFrame(
        {"id": ids_processed, "levenshtein": levenshtein_distances}
    )

    analysis_df = pd.merge(results_df, meta_df, on="id")

    # Calculate Correlation
    correlation = analysis_df["levenshtein"].corr(analysis_df["length"])

    print(
        f"Correlation between Levenshtein Distance and Sentence Length: {correlation:.4f}"
    )

    if correlation > 0.3:
        print(
            "Analysis: Positive correlation indicates the model struggles more with longer sentences."
        )
    elif correlation < -0.3:
        print(
            "Analysis: Negative correlation indicates the model struggles more with shorter sentences."
        )
    else:
        print(
            "Analysis: Weak correlation suggests error is independent of sentence length."
        )

    # ---------------------------------------------------------
    # 5. Submission Generation
    # ---------------------------------------------------------
    logger.info("Generating Final Submission...")

    # Reset debug flag for submission to process the full test set
    # Note: Config.DEBUG is global, but Trainer/Pipeline accept debug arg.
    # We create a new pipeline instance with debug=False to ensure full data loading.

    submission_pipeline = InferencePipeline(debug=False)
    submission_pipeline.generate_submission()

    logger.info("Process Complete.")


if __name__ == "__main__":
    main()
