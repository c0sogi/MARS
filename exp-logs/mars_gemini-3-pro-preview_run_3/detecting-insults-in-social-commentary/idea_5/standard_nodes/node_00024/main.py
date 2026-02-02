import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library import (
    utils,
    data_factory,
    pretrain_mlm,
    train_supervised,
    inference,
    modeling,
)


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    print("Initializing pipeline...")
    utils.seed_everything(Config.seeds[0])

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Clear specific cache files to prevent data leakage or stale data issues.
    # We specifically want to control when "supervised_data.parquet" (Train+Val) is created.
    supervised_cache = os.path.join(Config.cache_dir, "supervised_data.parquet")
    if os.path.exists(supervised_cache):
        os.remove(supervised_cache)
        print("Cleared previous supervised data cache.")

    # ==========================================
    # 2. Domain-Adaptive Pre-training (DAPT)
    # ==========================================
    # We run this once. It uses Train+Val+Test (unlabeled), which is permitted.
    # This creates the domain-adapted backbone used for all subsequent fine-tuning.
    if not os.path.exists(Config.dapt_model_output_dir):
        pretrain_mlm.run_domain_adaptation()
    else:
        print(
            f"DAPT model already exists at {Config.dapt_model_output_dir}. Skipping DAPT."
        )

    # ==========================================
    # 3. Validation Phase
    # ==========================================
    # Strategy: Train a single model on the Training split ONLY, then evaluate on the Validation split.
    # This ensures a fair calculation of the validation metric.

    print("\n" + "=" * 40)
    print("STARTING VALIDATION PHASE")
    print("=" * 40)

    # Monkey-patch data_factory.load_supervised_data to return only the training set
    # This forces train_supervised.py to train on the train split only.
    original_load_supervised_data = data_factory.load_supervised_data

    def load_train_only(load_cached_data=True):
        print("Validation Phase: Loading TRAIN split only...")
        df_train = pd.read_csv(Config.train_path)
        df_train["Comment"] = df_train["Comment"].apply(utils.decode_text)
        return df_train

    # Apply patch
    data_factory.load_supervised_data = load_train_only

    # Train one seed (Seed 42) for validation
    val_seed = Config.seeds[0]
    train_supervised.train_seed(val_seed)

    # Restore original data loader
    data_factory.load_supervised_data = original_load_supervised_data

    # Perform Inference on Validation Set
    print("Running inference on Validation Set...")

    # Load validation data manually
    df_val = pd.read_csv(Config.val_path)
    val_texts = df_val["Comment"].apply(utils.decode_text).values
    val_labels = df_val["Insult"].values

    # Load the model trained on train split
    model_path = os.path.join(Config.working_dir, f"model_seed_{val_seed}.bin")
    model = modeling.InsultModel(pretrained=False)
    state_dict = torch.load(model_path, map_location=Config.device)
    model.load_state_dict(state_dict)
    model.to(Config.device)
    model.eval()

    # Create DataLoader for validation inference
    tokenizer = data_factory.get_tokenizer()
    val_dataset = data_factory.InsultDataset(
        texts=val_texts, labels=None, tokenizer=tokenizer, max_length=Config.max_length
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Predict
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            val_preds.extend(probs)

    val_preds = np.array(val_preds)

    # Calculate Metric
    final_metric = roc_auc_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate absolute error
    errors = np.abs(val_labels - val_preds)

    # Feature: Text Length (Character count)
    text_lengths = np.array([len(t) for t in val_texts])

    # Correlation
    corr, _ = pearsonr(errors, text_lengths)
    print(f"Correlation between Error Magnitude and Comment Length: {corr:.10f}")

    # Additional Analysis: Error by Class
    mean_error_0 = errors[val_labels == 0].mean()
    mean_error_1 = errors[val_labels == 1].mean()
    print(f"Mean Error for Neutral Comments (0): {mean_error_0:.6f}")
    print(f"Mean Error for Insulting Comments (1): {mean_error_1:.6f}")

    # Clean up validation model from memory
    del model, state_dict, val_loader
    torch.cuda.empty_cache()

    # ==========================================
    # 5. Submission Phase
    # ==========================================
    threshold = 0.9639490968801314

    if final_metric > threshold:
        print("\n" + "=" * 40)
        print("VALIDATION PASSED - PROCEEDING TO SUBMISSION")
        print("=" * 40)

        # Strategy: Train on FULL Data (Train + Val) for maximum performance.
        # We must retrain the first seed (since it only saw Train split) and train the others.

        # Ensure cache is cleared so load_supervised_data creates the concatenated dataset
        if os.path.exists(supervised_cache):
            os.remove(supervised_cache)

        # Train all seeds
        for seed in Config.seeds:
            # Note: This overwrites the validation model for seed 42 with the full-data model
            train_supervised.train_seed(seed)

        # Generate Submission
        inference.generate_submission()

    else:
        print(
            f"\nValidation Metric {final_metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
