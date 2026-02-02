import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import warnings

# Import from provided library files
from library.config import Config, seed_everything
from library.dataset import get_tokenizer, QuestDataset
from library.model import SharedBottomSplitTopRoBERTa
from library.engine import run_training, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    Config.setup()
    device = Config.device
    print(f"Using device: {device}")

    # 2. Load Data
    # We use the metadata files which are already subsets/splits suitable for this task
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Initialize Tokenizer
    tokenizer = get_tokenizer()

    # Create Datasets
    train_ds = QuestDataset(train_df, tokenizer, mode="train")
    val_ds = QuestDataset(val_df, tokenizer, mode="val")
    test_ds = QuestDataset(test_df, tokenizer, mode="test")

    # Create DataLoaders
    # Using drop_last=True for train to maintain consistent batch shapes
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Initialize Model
    print("Initializing model...")
    model = SharedBottomSplitTopRoBERTa()
    model.to(device)

    # 4. Training
    # run_training handles the loop, validation, and saving the best model
    print("Starting training...")
    best_score = run_training(model, train_loader, val_loader, device)

    # 5. Final Validation & Failure Analysis
    print("\nRunning Failure Analysis...")

    # Load best model for analysis
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()

    val_preds = []
    val_targets = []

    # Inference loop for validation analysis
    with torch.no_grad():
        for batch in val_loader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            targets = batch["targets"].cpu().numpy()

            outputs = model(q_ids, q_mask, a_ids, a_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()

            val_preds.append(probs)
            val_targets.append(targets)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Spearman Score manually to ensure we print the exact required format
    spearman_scores = []
    for i in range(val_targets.shape[1]):
        if np.std(val_targets[:, i]) < 1e-9 or np.std(val_preds[:, i]) < 1e-9:
            score = 0.0
        else:
            score = spearmanr(val_targets[:, i], val_preds[:, i]).correlation
        spearman_scores.append(score)

    final_metric = np.nanmean(spearman_scores)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Text Length
    # Error metric: Mean Absolute Error per sample (averaged across 30 targets)
    sample_mae = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Features
    q_lengths = val_df["question_body"].fillna("").astype(str).str.len().values
    a_lengths = val_df["answer"].fillna("").astype(str).str.len().values

    # Correlations
    corr_q, _ = spearmanr(sample_mae, q_lengths)
    corr_a, _ = spearmanr(sample_mae, a_lengths)

    print(f"Failure Analysis - Error vs Question Length Correlation: {corr_q:.4f}")
    print(f"Failure Analysis - Error vs Answer Length Correlation: {corr_a:.4f}")

    # 6. Submission
    threshold = 0.4129102292393905
    if final_metric > threshold:
        print(
            f"\nValidation score ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        predict_and_submit(model, test_loader, test_df, device)
    else:
        print(
            f"\nValidation score ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
