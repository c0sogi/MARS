import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
from library.utils import set_seed, get_device
from library.data import (
    load_and_preprocess_data,
    prepare_augmented_data,
    create_dataloaders,
)
from library.model import DebertaV3Classifier
from library.engine import train_runner, inference_fn


def main():
    # 1. Setup
    device = get_device()

    # Ensure output dirs exist
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # 2. Load Data
    # We load cached data if available to speed up the process
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=True)

    # 3. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 4. Ensemble Training
    # We train multiple models with different seeds to create a robust ensemble.
    # Cite solution_lesson_node_00013: Simplicity Over Sophistication (removing Teacher-Student loop)
    # Cite solution_lesson_node_00027: Prioritize base model optimization (BS=32) over pipeline complexity.

    val_preds = []
    test_preds = []

    # We need to keep track of validation labels for the metric calculation
    val_labels = val_df["Insult"].values

    for seed in Config.seeds:
        set_seed(seed)

        # Create DataLoaders for this seed
        train_loader, val_loader, test_loader = create_dataloaders(
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            load_cached_data=True,
        )

        # Initialize Model
        model = DebertaV3Classifier(Config.model_name)
        model.to(device)

        # Train Model
        save_name = f"model_seed_{seed}.bin"
        model, best_auc = train_runner(
            train_loader, val_loader, model, save_name=save_name
        )

        # Inference on Validation (for Ensemble Metric)
        val_probs = inference_fn(val_loader, model, device)
        val_preds.append(val_probs)

        # Inference on Test
        test_probs = inference_fn(test_loader, model, device)
        test_preds.append(test_probs)

        # Cleanup to save memory
        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # 5. Validation Assessment & Failure Analysis
    # Average predictions from the ensemble
    avg_val_preds = np.mean(val_preds, axis=0)

    # Calculate Final Validation Metric
    final_val_auc = roc_auc_score(val_labels, avg_val_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis: Correlation between error magnitude and comment length
    errors = np.abs(val_labels - avg_val_preds)
    # Ensure we use the decoded length
    lengths = val_df["Comment"].apply(len).values

    correlation = np.corrcoef(errors, lengths)[0, 1]
    print(f"Correlation between Error Magnitude and Comment Length: {correlation}")

    # Check Threshold before proceeding
    THRESHOLD = 0.9639490968801314
    if final_val_auc <= THRESHOLD:
        print(
            f"Validation metric {final_val_auc} is not higher than threshold {THRESHOLD}. Exiting."
        )
        return

    # 6. Submission
    avg_test_preds = np.mean(test_preds, axis=0)

    # Load sample submission to ensure correct format
    sample_sub_path = "./input/sample_submission_null.csv"
    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path)
        sub_df["Insult"] = avg_test_preds
    else:
        # Fallback if sample file is missing
        sub_df = test_df.copy()
        sub_df["Insult"] = avg_test_preds
        # Ensure Insult is the first column
        cols = ["Insult"] + [c for c in sub_df.columns if c != "Insult"]
        sub_df = sub_df[cols]

    # Save Submission
    sub_df.to_csv(Config.submission_path, index=False)


if __name__ == "__main__":
    main()
