import sys
import os
import pandas as pd
import numpy as np
import torch
import nltk
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from transformers import AutoTokenizer, logging as hf_logging

# Suppress warnings and progress bars
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()

# Disable tqdm globally
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Add current directory to path
sys.path.append(os.getcwd())

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_levenshtein
from library.data import process_data, LocatorDataset, InfillerDataset, TestDataset
from library.engine import Trainer
from library.inference import BeamPipeline, run_inference


def main():
    # 1. Setup and Configuration Override
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    # Reducing sample size and epochs to ensure completion within 2 hours
    Config.TRAIN_SAMPLE_SIZE = 50000
    Config.VAL_SAMPLE_SIZE = 5000
    Config.LOCATOR_EPOCHS = 1
    Config.INFILLER_EPOCHS = 1

    # 2. Data Preparation
    # Load data using the library function
    # We slice the dataframes manually after loading to strictly enforce the fast baseline limits
    # in case the cache contained larger datasets.
    locator_train_df, locator_val_df, infiller_train_df, infiller_val_df = process_data(
        load_cached_data=True
    )

    # Apply strict slicing
    if len(locator_train_df) > Config.TRAIN_SAMPLE_SIZE:
        locator_train_df = locator_train_df.iloc[: Config.TRAIN_SAMPLE_SIZE]
    if len(infiller_train_df) > Config.TRAIN_SAMPLE_SIZE:
        infiller_train_df = infiller_train_df.iloc[: Config.TRAIN_SAMPLE_SIZE]
    if len(locator_val_df) > Config.VAL_SAMPLE_SIZE:
        locator_val_df = locator_val_df.iloc[: Config.VAL_SAMPLE_SIZE]
    if len(infiller_val_df) > Config.VAL_SAMPLE_SIZE:
        infiller_val_df = infiller_val_df.iloc[: Config.VAL_SAMPLE_SIZE]

    # Initialize Tokenizers
    try:
        loc_tokenizer = AutoTokenizer.from_pretrained(
            Config.LOCATOR_MODEL_NAME, use_fast=True
        )
    except:
        loc_tokenizer = AutoTokenizer.from_pretrained(
            Config.LOCATOR_MODEL_NAME, use_fast=False
        )

    try:
        inf_tokenizer = AutoTokenizer.from_pretrained(
            Config.INFILLER_MODEL_NAME, use_fast=True
        )
    except:
        inf_tokenizer = AutoTokenizer.from_pretrained(
            Config.INFILLER_MODEL_NAME, use_fast=False
        )

    # Create Datasets
    loc_train_ds = LocatorDataset(locator_train_df, loc_tokenizer)
    loc_val_ds = LocatorDataset(locator_val_df, loc_tokenizer)
    inf_train_ds = InfillerDataset(infiller_train_df, inf_tokenizer)
    inf_val_ds = InfillerDataset(infiller_val_df, inf_tokenizer)

    # Create DataLoaders
    loc_train_loader = DataLoader(
        loc_train_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    loc_val_loader = DataLoader(
        loc_val_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    inf_train_loader = DataLoader(
        inf_train_ds,
        batch_size=Config.INFILLER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    inf_val_loader = DataLoader(
        inf_val_ds,
        batch_size=Config.INFILLER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Training
    trainer = Trainer()
    # Train Locator
    trainer.train_locator(loc_train_loader, loc_val_loader)
    # Train In-Filler
    trainer.train_infiller(inf_train_loader, inf_val_loader)

    # 4. Validation Inference & Metrics
    # We use the trained pipeline to predict on the validation set to compute the task metric.

    # Prepare validation data for inference pipeline
    val_inference_df = locator_val_df.copy()
    val_inference_df["id"] = val_inference_df.index

    # Reconstruct Ground Truths
    ground_truths = []
    for _, row in val_inference_df.iterrows():
        words = row["sentence"].split()
        # gap_index is the word index BEFORE the gap.
        # We insert the missing word at gap_index + 1
        words.insert(row["gap_index"] + 1, row["missing_word"])
        ground_truths.append(" ".join(words))

    # Create Inference Dataset
    # We use a dummy tokenizer here as the pipeline re-tokenizes internally
    dummy_tokenizer = AutoTokenizer.from_pretrained(Config.LOCATOR_MODEL_NAME)
    val_inf_ds = TestDataset(val_inference_df, dummy_tokenizer)
    val_inf_loader = DataLoader(
        val_inf_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Pipeline
    pipeline = BeamPipeline()
    predictions_raw = pipeline.predict(val_inf_loader)

    # Map predictions back to order
    predictions_map = {idx: sent for idx, sent in predictions_raw}
    hypotheses = [predictions_map[idx] for idx in val_inference_df.index]

    # Compute Metric
    lev_score = compute_levenshtein(ground_truths, hypotheses)
    print(f"Final Validation Metric: {lev_score}")

    # 5. Failure Analysis
    # Calculate individual distances
    distances = []
    lengths = []

    for ref, hyp in zip(ground_truths, hypotheses):
        d = nltk.edit_distance(ref, hyp)
        distances.append(d)
        lengths.append(len(ref.split()))

    # Compute correlation
    if len(distances) > 1:
        corr, _ = pearsonr(distances, lengths)
        print(f"Correlation between Error (Levenshtein) and Sentence Length: {corr}")
    else:
        print("Not enough samples for correlation analysis.")

    # 6. Submission
    THRESHOLD = 6.149349349349349
    if lev_score < THRESHOLD:
        run_inference()


if __name__ == "__main__":
    main()
