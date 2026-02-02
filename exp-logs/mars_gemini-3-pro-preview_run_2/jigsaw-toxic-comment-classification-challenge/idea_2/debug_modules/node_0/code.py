import sys
import os
import numpy as np
import pandas as pd
import torch

# Import from library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_datasets
from library.tfidf_processor import TfidfVectorizationPipeline
from library.linear_model import LinearEnsembleTrainer
from library.transformer_data import create_dataloaders
from library.transformer_trainer import TransformerTrainer


def main():
    print("=== Starting Pipeline Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample size for quick execution

    # Reduce training intensity
    Config.EPOCHS = 1
    Config.MAX_LEN = 64  # Reduce sequence length for speed

    # Simplify Linear Ensemble
    Config.LR_C_VALUES = [1.0]  # Use single C value instead of list
    Config.TFIDF_WORD_MAX_FEATURES = 1000
    Config.TFIDF_CHAR_MAX_FEATURES = 1000

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[2] Loading Datasets...")

    # We force load_cached_data=False to demonstrate the loading logic from CSV
    # and to ensure we are working with the fresh debug slice.
    train_df, val_df, test_df = load_datasets(load_cached_data=False)

    # Verify Data Loading
    print(f"Train Shape: {train_df.shape}")
    print(f"Val Shape:   {val_df.shape}")
    print(f"Test Shape:  {test_df.shape}")

    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val size mismatch"
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test size mismatch"
    assert not train_df[Config.TEXT_COL].isnull().any(), "Null values in train text"

    # --------------------------------------------------------------------------
    # 3. Branch A: Linear Model (TF-IDF + Logistic Regression)
    # --------------------------------------------------------------------------
    print("\n[3] Executing Branch A: Linear Ensemble...")

    # 3a. Feature Extraction
    tfidf_pipeline = TfidfVectorizationPipeline()
    # Force reload to verify computation logic
    X_train, X_val, X_test = tfidf_pipeline.run(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify Feature Shapes
    assert X_train.shape[0] == len(train_df)
    assert X_val.shape[0] == len(val_df)
    assert X_test.shape[0] == len(test_df)
    print(f"TF-IDF Matrix Shape: {X_train.shape}")

    # 3b. Training and Prediction
    linear_trainer = LinearEnsembleTrainer()
    y_train = train_df[Config.LABEL_COLS]
    y_val = val_df[Config.LABEL_COLS]

    val_preds_linear, test_preds_linear = linear_trainer.train_and_predict(
        X_train, y_train, X_val, y_val, X_test
    )

    # Verify Predictions
    assert val_preds_linear.shape == (len(val_df), 6)
    assert test_preds_linear.shape == (len(test_df), 6)
    assert np.all(
        (test_preds_linear >= 0) & (test_preds_linear <= 1)
    ), "Linear preds out of bounds"
    print("Linear Branch execution successful.")

    # --------------------------------------------------------------------------
    # 4. Branch B: Deep Learning (Transformer)
    # --------------------------------------------------------------------------
    print("\n[4] Executing Branch B: Transformer Model...")

    # 4a. Data Preparation
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df
    )

    # 4b. Training
    transformer_trainer = TransformerTrainer()

    # Train for 1 epoch (as configured above)
    best_auc = transformer_trainer.train(train_loader, val_loader)
    print(f"Transformer Training Completed. Best AUC: {best_auc:.4f}")

    # 4c. Inference
    test_preds_transformer = transformer_trainer.predict(test_loader)

    # Verify Predictions
    assert test_preds_transformer.shape == (len(test_df), 6)
    assert np.all(
        (test_preds_transformer >= 0) & (test_preds_transformer <= 1)
    ), "Transformer preds out of bounds"
    print("Transformer Branch execution successful.")

    # --------------------------------------------------------------------------
    # 5. Ensemble and Submission
    # --------------------------------------------------------------------------
    print("\n[5] Ensembling and Generating Submission...")

    # Weighted Average
    w_linear = Config.ENSEMBLE_WEIGHTS["linear"]
    w_transformer = Config.ENSEMBLE_WEIGHTS["transformer"]

    print(f"Weights -> Linear: {w_linear}, Transformer: {w_transformer}")

    final_test_preds = (w_linear * test_preds_linear) + (
        w_transformer * test_preds_transformer
    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(final_test_preds, columns=Config.LABEL_COLS)
    submission_df.insert(0, "id", test_df["id"])

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = Config.SUBMISSION_CSV
    submission_df.to_csv(submission_path, index=False)

    # Final Verification
    print(f"\nSaved submission to: {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    assert os.path.exists(submission_path)
    loaded_sub = pd.read_csv(submission_path)
    assert loaded_sub.shape == (Config.DEBUG_SAMPLE_SIZE, 7)
    assert list(loaded_sub.columns) == ["id"] + Config.LABEL_COLS

    print("\n=== Pipeline Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
