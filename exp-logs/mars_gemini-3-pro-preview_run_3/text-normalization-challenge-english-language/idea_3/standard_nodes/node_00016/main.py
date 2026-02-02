import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_manager import load_parquet_data
from library.trainer import train_neural_model
from library.inference_pipeline import HybridNormalizer, generate_submission
from library.symbolic_solver import SymbolicModel


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Optimize hyperparameters for A100 GPU and 2-hour time limit
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 128  # Reduced to fit 16GB VRAM
    Config.NUM_WORKERS = 12  # Utilize all vCPUs
    Config.DEBUG = False  # Ensure full training

    Config.setup()
    seed_everything(Config.SEED)

    print("=== Starting End-to-End Workflow ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Train Neural Model
    # ==========================================
    print("\n--- Step 1: Training Neural Model ---")
    # This function handles:
    # - Loading raw data
    # - Building/Loading Tokenizer
    # - Preparing 'Hard' dataset (filtering easy tokens)
    # - Training Transformer
    # - Saving best model to Config.MODEL_PATH
    model, tokenizer = train_neural_model(load_cached_data=True)

    # ==========================================
    # 3. Validation Inference
    # ==========================================
    print("\n--- Step 2: Validation ---")
    # Load validation data
    df_val = load_parquet_data("val")

    # Initialize Hybrid Normalizer
    # This loads the symbolic stats and the neural model we just trained
    normalizer = HybridNormalizer(device=Config.DEVICE)
    normalizer.initialize_models()

    # Run inference
    print(f"Running inference on {len(df_val)} validation samples...")
    # predict_batch adds a 'predicted' column to the dataframe
    df_val_pred = normalizer.predict_batch(df_val)

    # Fill any remaining NaNs with the original text (Identity fallback)
    # The pipeline handles most, but this is a safety net
    if df_val_pred["predicted"].isna().any():
        nan_count = df_val_pred["predicted"].isna().sum()
        print(f"Filling {nan_count} remaining NaNs with identity mapping.")
        df_val_pred.loc[df_val_pred["predicted"].isna(), "predicted"] = df_val_pred.loc[
            df_val_pred["predicted"].isna(), "before"
        ]

    # ==========================================
    # 4. Metric Calculation
    # ==========================================
    # Exact string match accuracy
    correct_predictions = df_val_pred["predicted"] == df_val_pred["after"]
    accuracy = correct_predictions.mean()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n--- Step 3: Failure Analysis ---")
    df_val_pred["is_error"] = ~correct_predictions
    df_val_pred["error_int"] = df_val_pred["is_error"].astype(int)

    # Feature 1: Input Length
    df_val_pred["len_before"] = df_val_pred["before"].astype(str).apply(len)
    corr_len = df_val_pred["len_before"].corr(df_val_pred["error_int"])

    # Feature 2: Class (Encoded)
    corr_class = 0.0
    if "class" in df_val_pred.columns:
        # Factorize class to numeric codes
        df_val_pred["class_code"] = pd.factorize(df_val_pred["class"])[0]
        corr_class = df_val_pred["class_code"].corr(df_val_pred["error_int"])

        # Print error rates by class
        print("\nTop 5 Classes by Error Rate:")
        class_error_rates = (
            df_val_pred.groupby("class")["is_error"].mean().sort_values(ascending=False)
        )
        print(class_error_rates.head(5))

    print(f"\nCorrelation [Error vs Input Length]: {corr_len}")
    print(f"Correlation [Error vs Class]: {corr_class}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Step 4: Submission Check ---")
    THRESHOLD = 0.9943860453286453

    if accuracy > THRESHOLD:
        print(f"Validation accuracy ({accuracy}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        generate_submission()
    else:
        print(
            f"Validation accuracy ({accuracy}) did not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")

    print("\nWorkflow complete.")


if __name__ == "__main__":
    main()
