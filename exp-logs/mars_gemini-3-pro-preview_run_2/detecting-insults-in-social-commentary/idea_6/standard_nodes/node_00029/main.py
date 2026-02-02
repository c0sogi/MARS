import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_and_process_data, InsultDataset
from library.model import InsultModel
from library.train import run_training, inference_fn
from library.tapt import run_tapt
from library.inference import predict


def main():
    # =========================================================================
    # 1. Configuration Adjustments for Fast Baseline
    # =========================================================================
    # We reduce epochs to ensure the pipeline completes within the time limit.
    # The dataset is small (~3k), so 3 epochs should be sufficient for convergence
    # on a pre-trained large model.
    Config.epochs = 3
    Config.tapt_epochs = 3

    # Optimize batch sizes for 16GB GPU
    Config.train_batch_size = 4
    Config.tapt_batch_size = 4
    Config.valid_batch_size = 16

    # =========================================================================
    # 2. Setup
    # =========================================================================
    seed_everything(Config.seed)
    Config.setup()

    print("Starting End-to-End Pipeline...")
    print(f"Configuration: Epochs={Config.epochs}, TAPT Epochs={Config.tapt_epochs}")

    # =========================================================================
    # 3. Stage 1: Task-Adaptive Pre-Training (TAPT)
    # =========================================================================
    print("\n=== Stage 1: Task-Adaptive Pre-Training ===")
    run_tapt()

    # =========================================================================
    # 4. Stage 2: Supervised Training (5-Fold CV)
    # =========================================================================
    print("\n=== Stage 2: Supervised Training ===")
    run_training()

    # =========================================================================
    # 5. Validation Assessment
    # =========================================================================
    print("\n=== Stage 3: Validation Assessment ===")

    # We need to calculate the metric on the hold-out validation set (metadata/val.csv).
    # Since the training pipeline merges train and val for CV, we must reconstruct
    # the Out-Of-Fold (OOF) predictions specifically for the rows in val.csv.

    # Load data
    df_train, df_val, _ = load_and_process_data(load_cached_data=True)

    # Reconstruct the full dataframe used in training to match indices
    df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

    # The hold-out validation set corresponds to the last len(df_val) rows
    val_start_idx = len(df_train)
    val_indices_global = set(range(val_start_idx, len(df_full)))

    # Array to store predictions for df_val
    val_preds = np.zeros(len(df_val))
    val_targets = df_val[Config.target_col].values

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Replicate the StratifiedKFold split used in training
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    device = Config.device

    print("Generating OOF predictions for validation set...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full[Config.target_col])
    ):
        # Identify which rows in this fold's validation set belong to our hold-out val set
        # Intersection of fold-val-indices and global-val-indices
        current_fold_val_indices = [i for i in val_idx if i in val_indices_global]

        if not current_fold_val_indices:
            continue

        # Load the model trained for this fold
        model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model for fold {fold} not found at {model_path}")
            continue

        # Initialize and load model
        model = InsultModel(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Prepare data for inference
        subset_df = df_full.iloc[current_fold_val_indices].reset_index(drop=True)
        dataset = InsultDataset(subset_df, tokenizer, Config.max_len)
        loader = DataLoader(
            dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Run inference
        fold_preds = inference_fn(model, loader, device)

        # Map predictions back to the validation array
        for global_idx, pred in zip(current_fold_val_indices, fold_preds):
            local_idx = global_idx - val_start_idx
            val_preds[local_idx] = pred

        # Clean up to save memory
        del model, dataset, loader, fold_preds
        torch.cuda.empty_cache()

    # Calculate Final Metric
    final_metric = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n=== Stage 4: Failure Analysis ===")

    # Calculate error magnitude
    errors = np.abs(val_targets - val_preds)

    # Extract features from validation text
    # Note: df_val['Comment'] is already cleaned
    texts = df_val[Config.text_col].astype(str).tolist()
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]

    # Calculate correlations
    corr_char = np.corrcoef(errors, char_lens)[0, 1]
    corr_word = np.corrcoef(errors, word_lens)[0, 1]

    print(f"Correlation (Error vs Char Length): {corr_char}")
    print(f"Correlation (Error vs Word Count): {corr_word}")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    print("\n=== Stage 5: Submission ===")
    threshold = 0.9632101806239738

    if final_metric > threshold:
        print(
            f"Metric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        predict(load_cached_data=True)
    else:
        print(
            f"Metric ({final_metric}) does not exceed threshold ({threshold}). Submission skipped."
        )

    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
