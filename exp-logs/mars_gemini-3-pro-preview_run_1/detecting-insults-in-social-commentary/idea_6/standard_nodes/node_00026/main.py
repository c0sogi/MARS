import pandas as pd
import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_score
from library.data_processing import load_data, get_structural_features, InsultDataset
from library.trainer import run_mlm, run_fold, predict
from library.model import HybridDeberta


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading & Feature Engineering
    # ==========================================
    print("Loading Data...")
    train_df, val_df, test_df = load_data()

    print("Generating Structural Features...")
    train_svd, val_svd, test_svd = get_structural_features(
        train_df["Comment"].tolist(),
        val_df["Comment"].tolist(),
        test_df["Comment"].tolist(),
        load_cached_data=True,
    )

    # ==========================================
    # 3. Stage 1: Domain Adaptation (MLM)
    # ==========================================
    # Only run if not already present to save time
    if not os.path.exists(Config.MLM_MODEL_PATH):
        print("Starting Stage 1: MLM Domain Adaptation...")
        run_mlm(train_df["Comment"], val_df["Comment"], test_df["Comment"])
    else:
        print(f"MLM model found at {Config.MLM_MODEL_PATH}, skipping Stage 1.")

    # ==========================================
    # 4. Stage 2: Supervised Training
    # ==========================================
    print("Starting Stage 2: Supervised Training...")

    # Prepare data for run_fold (which expects combined data + indices)
    combined_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    combined_svd = np.concatenate([train_svd, val_svd], axis=0)

    # Define indices for the fixed split
    train_idx = np.arange(len(train_df))
    val_idx = np.arange(len(train_df), len(combined_df))

    # Determine which tokenizer/backbone to use
    model_source_path = (
        Config.MLM_MODEL_PATH
        if os.path.exists(Config.MLM_MODEL_PATH)
        else Config.MODEL_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source_path)

    # Run training using the library function
    # We treat this as "Fold 0"
    test_preds, best_val_score = run_fold(
        fold=0,
        train_idx=train_idx,
        val_idx=val_idx,
        df=combined_df,
        svd_features=combined_svd,
        tokenizer=tokenizer,
        test_df=test_df,
        test_svd=test_svd,
    )

    # Report Metric
    print(f"Final Validation Metric: {best_val_score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nRunning Failure Analysis...")

    # Load the best model saved by run_fold
    model_path = os.path.join(Config.WORKING_DIR, "model_fold_0.bin")
    model = HybridDeberta(pretrained_model_name_or_path=model_source_path)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # Create Validation DataLoader for Fold 0
    # We use the specific validation split from Fold 0 for analysis
    val_data_fold0 = combined_df.iloc[val_idx].reset_index(drop=True)
    val_svd_fold0 = combined_svd[val_idx]

    val_dataset = InsultDataset(
        val_data_fold0["Comment"].values, val_svd_fold0, tokenizer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Inference
    val_probs = predict(model, val_loader, Config.DEVICE)
    val_labels = val_data_fold0["Insult"].values

    # Compute Error
    errors = np.abs(val_labels - val_probs)

    # Compute Features
    val_data_fold0["char_count"] = val_data_fold0["Comment"].apply(
        lambda x: len(str(x))
    )
    val_data_fold0["word_count"] = val_data_fold0["Comment"].apply(
        lambda x: len(str(x).split())
    )

    # Compute Correlations
    corr_char = np.corrcoef(errors, val_data_fold0["char_count"])[0, 1]
    corr_word = np.corrcoef(errors, val_data_fold0["word_count"])[0, 1]

    print("-" * 30)
    print("Failure Analysis Report:")
    print(f"Correlation (Error vs Char Count): {corr_char:.4f}")
    print(f"Correlation (Error vs Word Count): {corr_word:.4f}")
    print("-" * 30)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.9586453201970443

    if best_val_score > THRESHOLD:
        print(
            f"Validation score ({best_val_score:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        sample_sub_path = os.path.join("./input", "sample_submission_null.csv")

        if os.path.exists(sample_sub_path):
            submission_df = pd.read_csv(sample_sub_path)
            # Ensure length matches
            if len(submission_df) == len(test_preds):
                submission_df["Insult"] = test_preds
            else:
                print(
                    f"Warning: Sample submission length ({len(submission_df)}) != Test preds length ({len(test_preds)}). Creating new DF."
                )
                submission_df = pd.DataFrame({"Insult": test_preds})
        else:
            submission_df = pd.DataFrame({"Insult": test_preds})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation score ({best_val_score:.6f}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
