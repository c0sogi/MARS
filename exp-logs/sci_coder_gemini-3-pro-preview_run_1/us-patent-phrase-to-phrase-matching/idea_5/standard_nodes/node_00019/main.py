import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as transformers_logging

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_score, get_logger
from library.dataset import get_cpc_texts, get_folds, PearsonDataset
from library.engine import run_fold, predict_and_submit, inference_fn
from library.model import CustomModel

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()


def analyze_failures(df, y_true, y_pred):
    """
    Performs failure analysis by correlating error with input features.
    """
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Create analysis dataframe
    analysis_df = df.copy()
    analysis_df["error"] = errors
    analysis_df["score_true"] = y_true

    # Feature engineering for analysis
    # Lengths
    analysis_df["len_anchor"] = analysis_df["anchor"].astype(str).apply(len)
    analysis_df["len_target"] = analysis_df["target"].astype(str).apply(len)
    analysis_df["len_diff"] = abs(analysis_df["len_anchor"] - analysis_df["len_target"])

    # Jaccard Similarity (Simple token overlap)
    def get_jaccard(s1, s2):
        a = set(str(s1).lower().split())
        b = set(str(s2).lower().split())
        c = a.intersection(b)
        if len(a) + len(b) - len(c) == 0:
            return 0.0
        return float(len(c)) / (len(a) + len(b) - len(c))

    analysis_df["jaccard"] = analysis_df.apply(
        lambda x: get_jaccard(x["anchor"], x["target"]), axis=1
    )

    # Calculate correlations with error
    features = ["score_true", "len_anchor", "len_target", "len_diff", "jaccard"]
    print("Correlation between Absolute Error and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["error"])
            print(f"{feat}: {corr:.4f}")
    print("=" * 30 + "\n")


def main():
    # 1. Setup
    Config.setup_system()
    logger = get_logger("main_script")

    # Adjust Config for Fast Baseline
    # We limit data size and epochs to ensure completion within 2 hours
    Config.epochs = 2
    SAMPLE_SIZE = 8000  # Subsample training data

    logger.info(f"Starting Fast Baseline Run")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Epochs: {Config.epochs}, Train Sample Size: {SAMPLE_SIZE}")

    # 2. Data Loading & Preprocessing
    # Load Metadata
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    test_df = pd.read_csv(Config.test_path)

    # Subsample training data for speed (Stratified sample to keep distribution)
    if len(train_df) > SAMPLE_SIZE:
        # Simple stratified sampling by score
        train_df = (
            train_df.groupby("score", group_keys=False)
            .apply(
                lambda x: x.sample(
                    int(np.rint(SAMPLE_SIZE * len(x) / len(train_df))),
                    random_state=Config.seed,
                )
            )
            .reset_index(drop=True)
        )
        logger.info(f"Subsampled train data to {len(train_df)} rows.")

    # Load CPC Contexts and Map
    cpc_texts = get_cpc_texts(load_cached_data=True)

    def map_context(df):
        df["context_text"] = df["context"].map(cpc_texts)
        # Fill missing with raw code if description not found
        df["context_text"] = df["context_text"].fillna(df["context"])
        return df

    train_df = map_context(train_df)
    val_df = map_context(val_df)
    test_df = map_context(test_df)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 3. Training Loop (K-Fold)
    # Generate Folds
    train_df = get_folds(train_df, n_splits=Config.num_folds)

    # We will store models to disk (handled by engine.py), but we need to ensure they are created.
    for fold in range(Config.num_folds):
        logger.info(f"--- Preparing Fold {fold} ---")

        # Split Data
        trn_idx = train_df[train_df["fold"] != fold].index
        val_idx = train_df[train_df["fold"] == fold].index

        df_train_fold = train_df.loc[trn_idx].reset_index(drop=True)
        df_valid_fold = train_df.loc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = PearsonDataset(df_train_fold, tokenizer, Config.max_len)
        valid_dataset = PearsonDataset(df_valid_fold, tokenizer, Config.max_len)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Run Training
        run_fold(fold, train_loader, valid_loader, Config.device)

        # Clear memory
        del (
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            df_train_fold,
            df_valid_fold,
        )
        torch.cuda.empty_cache()

    # 4. Hold-out Validation Evaluation
    logger.info("Running evaluation on hold-out validation set...")

    # Prepare Validation Loader
    val_dataset = PearsonDataset(val_df, tokenizer, Config.max_len, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Ensemble Inference
    all_preds = []
    for fold in range(Config.num_folds):
        model_path = os.path.join(Config.models_dir, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            continue

        model = CustomModel()
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        preds = inference_fn(model, val_loader, Config.device)
        all_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    if not all_preds:
        logger.error("No models found for validation.")
        return

    avg_preds = np.mean(all_preds, axis=0)
    y_true = val_df["score"].values

    # Compute Metric
    final_metric = get_score(y_true, avg_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    analyze_failures(val_df, y_true, avg_preds)

    # 6. Submission
    THRESHOLD = 0.8673
    if final_metric > THRESHOLD:
        logger.info(
            f"Metric {final_metric:.4f} > {THRESHOLD}. Generating submission..."
        )

        # Prepare Test Loader
        test_dataset = PearsonDataset(test_df, tokenizer, Config.max_len, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Use engine function to predict and save
        # Note: predict_and_submit loads models internally, so we just pass the loader
        predict_and_submit(test_loader, Config.device)

    else:
        logger.warning(f"Metric {final_metric:.4f} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
