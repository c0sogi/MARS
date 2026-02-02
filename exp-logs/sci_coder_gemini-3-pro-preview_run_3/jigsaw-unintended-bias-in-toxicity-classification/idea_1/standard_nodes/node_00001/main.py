import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.metrics import calculate_jigsaw_metrics
from library.dataset import ToxicityDataset, collate_batch
from library.data_utils import clean_and_tokenize


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_validation_inference(trainer, val_df):
    """
    Runs inference on the validation set using the trained model.
    Returns the dataframe with a new 'prediction' column.
    """
    device = trainer.device
    model = trainer.model
    vocab = trainer.vocab

    # Ensure model is in eval mode
    model.eval()

    # Create dataset and loader
    val_dataset = ToxicityDataset(
        texts=val_df[Config.TEXT_COL].tolist(),
        targets=val_df[Config.TARGET_COL].tolist(),
        vocab=vocab,
        is_training=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    all_preds = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Check if batch contains targets (tuple) or just text
            if isinstance(batch, tuple) and len(batch) == 3:
                texts, offsets, _ = batch
            else:
                texts, offsets = batch

            texts = texts.to(device)
            offsets = offsets.to(device)

            outputs = model(texts, offsets).squeeze()

            # Handle edge case for single-item batch
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            all_preds.extend(outputs.cpu().numpy())

    val_df_pred = val_df.copy()
    val_df_pred["prediction"] = all_preds
    return val_df_pred


def perform_failure_analysis(val_df):
    """
    Analyzes correlations between error magnitude and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    # Target is fractional, prediction is probability.
    val_df["error"] = (val_df[Config.TARGET_COL] - val_df["prediction"]).abs()

    # Feature 1: Text Length
    val_df["text_len"] = val_df[Config.TEXT_COL].fillna("").astype(str).apply(len)

    # Correlation with Text Length
    len_corr = val_df["error"].corr(val_df["text_len"])
    print(f"Correlation between Error and Text Length: {len_corr:.4f}")

    # Correlation with Identity Attributes
    print("Correlation between Error and Identity Attributes:")
    identity_corrs = {}
    for col in Config.IDENTITY_COLUMNS:
        if col in val_df.columns:
            # We calculate correlation on the subset where identity is annotated (non-NaN)
            # or fillna(0) if we assume NaN means not present.
            # Given the task description implies fractional values, we use them directly.
            # We fill NaN with 0 for correlation calculation over the whole set.
            subset = val_df.copy()
            subset[col] = subset[col].fillna(0.0)
            corr = subset["error"].corr(subset[col])
            identity_corrs[col] = corr
            print(f"  {col}: {corr:.4f}")

    return identity_corrs


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Load Data
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Fast Baseline: Limit training samples
    # Using 200,000 samples to ensure execution finishes well within the time limit
    # while providing enough data for the NBOW model to converge.
    SAMPLE_SIZE = 200000
    if len(train_df) > SAMPLE_SIZE:
        print(
            f"Subsampling training data from {len(train_df)} to {SAMPLE_SIZE} for fast baseline..."
        )
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED).reset_index(
            drop=True
        )

    # 3. Train Model
    print("Initializing Trainer...")
    trainer = Trainer(device=device)

    # We use fewer epochs for the baseline run to ensure speed,
    # relying on early stopping to save the best model.
    print("Starting training...")
    trainer.train(
        train_df=train_df,
        val_df=val_df,
        batch_size=Config.BATCH_SIZE,
        epochs=5,  # Reduced from Config default for fast baseline
        lr=Config.LEARNING_RATE,
        patience=Config.PATIENCE,
    )

    # 4. Validation Assessment
    # Run inference on full validation set using the best restored model
    val_df_with_preds = run_validation_inference(trainer, val_df)

    # Calculate Metrics
    metrics = calculate_jigsaw_metrics(val_df_with_preds, prediction_col="prediction")

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {metrics['final_score']:.16f}")
    print(f"  Overall AUC: {metrics['overall_auc']:.6f}")
    print(f"  Subgroup AUC Mean: {metrics['subgroup_auc_mean']:.6f}")
    print(f"  BPSN AUC Mean: {metrics['bpsn_auc_mean']:.6f}")
    print(f"  BNSP AUC Mean: {metrics['bnsp_auc_mean']:.6f}")

    # 5. Failure Analysis
    perform_failure_analysis(val_df_with_preds)

    # 6. Generate Submission
    print("\nGenerating submission for test set...")
    trainer.predict(test_df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
