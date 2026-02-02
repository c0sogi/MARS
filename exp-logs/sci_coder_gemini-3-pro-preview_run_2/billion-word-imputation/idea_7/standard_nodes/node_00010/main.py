import os
import sys
import pandas as pd
import numpy as np
import torch
import nltk
import random
from scipy.stats import pearsonr
import warnings
import csv

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, TestDataset
from library.models import LocatorModel, InfillerModel, VerifierModel
from library.engine import Trainer
from library.pipeline import InferencePipeline, generate_submission


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    print("Configuring for Fast Baseline...")

    # Modify Config for speed and resource efficiency on A100
    Config.TRAIN_SIZE = 100_000  # Reduced from 2M to fit time limit
    Config.VAL_SIZE = 5_000  # Reduced validation set size

    Config.LOCATOR_EPOCHS = 2
    Config.INFILLER_EPOCHS = 2
    Config.VERIFIER_EPOCHS = 2

    # Optimize Batch Sizes for A100 40GB
    Config.LOCATOR_BATCH_SIZE = 128
    Config.INFILLER_BATCH_SIZE = 64
    Config.VERIFIER_BATCH_SIZE = 64

    # Reduce Beam Width for faster inference (Trade-off: Speed vs Accuracy)
    Config.LOCATOR_TOP_K = 3

    # ---------------------------------------------------------
    # 2. Training Phase
    # ---------------------------------------------------------
    print("\n=== Starting Training Phase ===")

    # Force fresh data generation to respect the new TRAIN_SIZE
    # We pass load_cached_data=False to ensure we don't load old large datasets
    dataloaders = get_dataloaders(load_cached_data=False)

    # Train Locator
    if not os.path.exists(Config.LOCATOR_CKPT_PATH):
        print("Training Locator...")
        locator = LocatorModel(pretrained=True)
        train_loader, val_loader = dataloaders["locator"]
        Trainer.train_locator(locator, train_loader, val_loader)
        # Free memory
        del locator, train_loader, val_loader
        torch.cuda.empty_cache()
    else:
        print("Locator checkpoint found. Skipping training.")

    # Train Infiller
    if not os.path.exists(Config.INFILLER_CKPT_PATH):
        print("Training Infiller...")
        infiller = InfillerModel(pretrained=True)
        train_loader, val_loader = dataloaders["infiller"]
        Trainer.train_infiller(infiller, train_loader, val_loader)
        del infiller, train_loader, val_loader
        torch.cuda.empty_cache()
    else:
        print("Infiller checkpoint found. Skipping training.")

    # Train Verifier
    if not os.path.exists(Config.VERIFIER_CKPT_PATH):
        print("Training Verifier...")
        verifier = VerifierModel(pretrained=True)
        train_loader, val_loader = dataloaders["verifier"]
        Trainer.train_verifier(verifier, train_loader, val_loader)
        del verifier, train_loader, val_loader
        torch.cuda.empty_cache()
    else:
        print("Verifier checkpoint found. Skipping training.")

    # ---------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Starting Evaluation Phase ===")

    # Load raw validation metadata
    df_val_full = pd.read_parquet(Config.VAL_META_PATH)

    # Sample a subset for evaluation to save time
    eval_size = 2000
    df_eval = df_val_full.sample(n=eval_size, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Prepare Evaluation Data: Synthetically remove one word
    masked_data = []
    ground_truth = []

    for idx, row in df_eval.iterrows():
        words = row["sentence"].split()
        # Test set logic: never first or last word. Needs at least 3 words.
        if len(words) >= 3:
            remove_idx = random.randint(1, len(words) - 2)
            mod_words = words[:remove_idx] + words[remove_idx + 1 :]
            masked_sentence = " ".join(mod_words)

            masked_data.append(
                {
                    "id": idx,  # Use temp ID
                    "sentence": masked_sentence,
                    "char_len": len(masked_sentence),
                }
            )
            ground_truth.append(row["sentence"])

    df_masked = pd.DataFrame(masked_data)

    # Prepare DataLoader for InferencePipeline
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # We need the locator tokenizer for the TestDataset class
    tokenizer_loc = AutoTokenizer.from_pretrained(
        Config.LOCATOR_MODEL_NAME, use_fast=True
    )

    eval_ds = TestDataset(df_masked, tokenizer_loc, Config.MAX_LENGTH)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference Pipeline
    pipeline = InferencePipeline(load_models=True)
    results = pipeline.run_inference(eval_loader)

    # Map results back to order
    pred_map = {r[0]: r[1] for r in results}
    predictions = [pred_map[row_id] for row_id in df_masked["id"]]

    # Compute Levenshtein Distance
    distances = []
    total_dist = 0
    for pred, target in zip(predictions, ground_truth):
        d = nltk.edit_distance(pred, target)
        distances.append(d)
        total_dist += d

    final_metric = total_dist / len(predictions)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    lengths = df_masked["char_len"].tolist()
    if len(distances) > 1:
        corr, _ = pearsonr(distances, lengths)
        print(f"Correlation between Error and Sentence Length: {corr:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # ---------------------------------------------------------
    # 4. Submission Generation
    # ---------------------------------------------------------
    threshold = 3.708908908908909

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric:.4f}) is better than threshold ({threshold:.4f}). Generating submission..."
        )

        # We can reuse the pipeline instance, but generate_submission creates its own.
        # To save memory, let's delete our current pipeline and let generate_submission handle it,
        # or just call pipeline.run_inference on the test set manually.
        # Calling generate_submission() is safer as it handles file I/O as per spec.
        del pipeline
        torch.cuda.empty_cache()

        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric:.4f}) did not beat threshold ({threshold:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
