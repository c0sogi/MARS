import os
import pandas as pd
import numpy as np
import torch

from library.config import Config
from library.utils import seed_everything, print_metric
from library.data_loader import get_data_loaders
from library.model import ToxicityClassifier
from library.trainer import Trainer
from library.metrics import compute_final_metric


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Optimize for A100 GPU and Time Limit
    Config.TRAIN_BATCH_SIZE = 128
    Config.VALID_BATCH_SIZE = 256
    Config.EPOCHS = (
        2  # Needed to trigger unfreezing in Epoch 2 based on provided Trainer logic
    )

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    print("Configuration updated for runtime optimization:")
    print(f"  Batch Size: {Config.TRAIN_BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading Data...")
    # Load DataLoaders (cached or processed)
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # Load raw validation dataframe for metrics and analysis
    val_df = pd.read_csv(Config.VAL_PATH)

    # ==========================================
    # 3. Model & Training
    # ==========================================
    print("\nInitializing Model...")
    model = ToxicityClassifier(Config.MODEL_NAME)
    trainer = Trainer(model, device=Config.DEVICE)

    print("\nStarting Training...")
    # fit() handles the training loop, validation monitoring, and model saving
    trainer.fit(train_loader, val_loader, val_df)

    # ==========================================
    # 4. Final Evaluation
    # ==========================================
    print("\nRunning Final Evaluation...")
    # We predict on the validation set to get the exact predictions for analysis
    val_preds = trainer.predict(val_loader)

    # Ensure alignment
    if len(val_preds) != len(val_df):
        print(f"Warning: Prediction length {len(val_preds)} != DF length {len(val_df)}")
        val_df_eval = val_df.iloc[: len(val_preds)].copy()
    else:
        val_df_eval = val_df.copy()

    val_df_eval["prediction"] = val_preds

    # Calculate and print the required metric
    final_score = compute_final_metric(
        val_df_eval, "prediction", Config.TARGET_COL, verbose=False
    )
    print(f"Final Validation Metric: {final_score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nFailure Analysis:")
    # Calculate absolute error
    val_df_eval["error"] = (
        val_df_eval[Config.TARGET_COL] - val_df_eval["prediction"]
    ).abs()

    # Add text length as a feature
    val_df_eval["text_len"] = val_df_eval["comment_text"].fillna("").str.len()

    # Features to analyze
    analysis_features = Config.IDENTITY_COLUMNS + ["text_len", Config.TARGET_COL]

    print("Correlation between Error Magnitude and Features:")
    for feature in analysis_features:
        if feature in val_df_eval.columns:
            # Handle NaNs in identity columns by filling with 0 (assuming not mentioned)
            series = val_df_eval[feature].fillna(0)
            if pd.api.types.is_numeric_dtype(series):
                corr = series.corr(val_df_eval["error"])
                print(f"  {feature}: {corr:.4f}")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.9022848229047395

    if final_score > THRESHOLD:
        print(
            f"\nScore ({final_score:.5f}) meets threshold ({THRESHOLD:.5f}). Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nScore ({final_score:.5f}) does not meet threshold ({THRESHOLD:.5f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
