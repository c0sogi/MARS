import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
from functools import partial


# --- 1. Suppress TQDM and Logging ---
class noop_tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass


import tqdm
import tqdm.auto

tqdm.tqdm = noop_tqdm
tqdm.auto.tqdm = noop_tqdm

# --- 2. Imports from Library ---
from library.config import Config
from library.utils import seed_everything, compute_qwk, get_logger
from library.data import (
    preprocess_data,
    FeatureEngineer,
    EssayDataset,
    get_tokenizer,
    make_folds,
)
from library.trainer import run_training, inference_fn, CustomModel
from library.stacking import StackingModel

# --- 3. Configuration Overrides for Fast Baseline ---
logger = get_logger("RunFile")
# Set logging to ERROR to minimize output as requested
logging.getLogger("Trainer").setLevel(logging.ERROR)
logging.getLogger("DataModule").setLevel(logging.ERROR)
logging.getLogger("Stacking").setLevel(logging.ERROR)

# Adjust Config for speed
Config.epochs = 1
Config.mlm_epochs = 0  # Skip MLM
Config.num_folds = 3  # Use 3 folds instead of 5
Config.train_batch_size = (
    4  # A100 can handle larger batch size than 1 for DeBERTa Large
)
Config.gradient_accumulation_steps = 4  # Maintain effective batch size ~16
Config.debug = False


def main():
    seed_everything(Config.seed)

    # --- 4. Data Loading & Subsampling ---
    # Load data (fresh load to ensure we control the subsampling)
    train_df, test_df = preprocess_data(load_cached_data=False)

    # Subsample to 5000 samples for speed (approx 15-20 mins training time)
    if len(train_df) > 5000:
        train_df = train_df.iloc[:5000].reset_index(drop=True)

    # Re-create stratified folds on the subsampled data
    train_df = make_folds(train_df, num_folds=Config.num_folds, seed=Config.seed)

    # --- 5. Level 1 Training (DeBERTa) ---
    # run_training returns OOF dataframe and a rounded submission df (which we won't use directly for stacking inference)
    oof_df, _ = run_training(train_df, test_df)

    # --- 6. Level 2 Training (Stacking) ---
    # Train Stacker on OOF predictions
    stacker = StackingModel()
    stacker.train(oof_df, train_df)

    # --- 7. Validation on Hold-out Set ---
    val_path = os.path.join(Config.metadata_dir, "val.csv")
    if not os.path.exists(val_path):
        print(f"Error: Validation file not found at {val_path}")
        return

    val_df = pd.read_csv(val_path)

    # Feature Engineering on Validation Set
    fe = FeatureEngineer()
    val_df = fe.extract_features(val_df)

    # Level 1 Inference on Validation Set
    tokenizer = get_tokenizer()
    val_dataset = EssayDataset(val_df, tokenizer, Config.max_length, is_test=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.eval_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    l1_val_preds_accum = np.zeros(len(val_df))

    # Iterate through saved fold models and average predictions
    for fold in range(Config.num_folds):
        model_path = os.path.join(Config.model_dir, f"model_fold_{fold}.pth")

        # Load Model
        model = CustomModel(
            pretrained=False
        )  # Backbone weights loaded from config, but we overwrite state_dict
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        # Inference
        preds = inference_fn(model, val_loader)
        l1_val_preds_accum += preds

        # Cleanup
        del model
        torch.cuda.empty_cache()

    l1_val_preds_avg = l1_val_preds_accum / Config.num_folds

    # Level 2 Inference on Validation Set
    final_val_preds = stacker.predict(val_df, l1_val_preds_avg)

    # --- 8. Validation Metric ---
    val_qwk = compute_qwk(val_df["score"].values, final_val_preds)
    print(f"Final Validation Metric: {val_qwk}")

    # --- 9. Failure Analysis ---
    # Calculate absolute error
    # Note: final_val_preds are floats. We round them for QWK but use raw/rounded diff for analysis?
    # Usually failure analysis uses the metric-aligned predictions.
    val_preds_rounded = np.round(final_val_preds).clip(1, 6).astype(int)
    val_df["error"] = np.abs(val_df["score"] - val_preds_rounded)

    print("\nFailure Analysis (Correlation with Error):")
    for feature in Config.meta_features:
        if feature in val_df.columns:
            corr = val_df["error"].corr(val_df[feature])
            print(f"{feature}: {corr:.4f}")

    # --- 10. Submission Generation ---
    THRESHOLD = 0.8274925140324321

    if val_qwk > THRESHOLD:
        # Generate Test Predictions
        test_dataset = EssayDataset(test_df, tokenizer, Config.max_length, is_test=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.eval_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        l1_test_preds_accum = np.zeros(len(test_df))

        for fold in range(Config.num_folds):
            model_path = os.path.join(Config.model_dir, f"model_fold_{fold}.pth")
            model = CustomModel(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            model.to(Config.device)
            model.eval()

            preds = inference_fn(model, test_loader)
            l1_test_preds_accum += preds

            del model
            torch.cuda.empty_cache()

        l1_test_preds_avg = l1_test_preds_accum / Config.num_folds

        # Stacking Prediction on Test
        final_test_preds = stacker.predict(test_df, l1_test_preds_avg)

        # Create Submission File
        submission = pd.DataFrame(
            {"essay_id": test_df["essay_id"], "score": final_test_preds}
        )

        # Round and Clip
        submission["score"] = np.round(submission["score"]).clip(1, 6).astype(int)

        submission.to_csv(Config.submission_path, index=False)


if __name__ == "__main__":
    main()
