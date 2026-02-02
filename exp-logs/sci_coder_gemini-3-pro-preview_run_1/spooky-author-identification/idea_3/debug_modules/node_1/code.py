import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Import Config and Apply Overrides for Speed/Demo
from library.config import Config

print(">>> Configuring environment for fast demonstration...")
Config.DEBUG = True
Config.EPOCHS = 1
Config.TRAIN_BATCH_SIZE = 8
Config.VALID_BATCH_SIZE = 8
Config.MAX_LENGTH = 32  # Reduce sequence length for speed
Config.N_FOLDS = 2  # Note: Default args in function definitions might still use 5, but DEBUG data size makes this fast enough.

# Redirect outputs to a demo directory to avoid clutter
Config.WORKING_DIR = "./working/demo_run"
Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
Config.SUBMISSION_DIR = "./working/demo_submission"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

# Reduce complexity of sub-models
Config.TFIDF_WORD_PARAMS["max_features"] = 500
Config.TFIDF_CHAR_PARAMS["max_features"] = 500
Config.XGB_NUM_ROUNDS = 5
Config.XGB_EARLY_STOPPING_ROUNDS = 2

# Clean up previous demo runs if they exist
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
if os.path.exists(Config.SUBMISSION_DIR):
    shutil.rmtree(Config.SUBMISSION_DIR)

# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Import remaining modules after Config setup
from library.utils import seed_everything
from library.data_processing import load_data, MetaFeatureExtractor
from library.model_tfidf import TfidfExpert
from library.model_transformer import Trainer
from library.model_stacking import StackingMetaLearner
from library.pipeline_manager import PipelineManager


def run_demo():
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[1/5] Verifying Data Loading...")
    train_df, val_df, test_df = load_data()

    # In DEBUG mode, load_data returns 200 training samples
    assert (
        len(train_df) == 200
    ), f"Expected 200 training samples in DEBUG mode, got {len(train_df)}"
    assert (
        len(val_df) == 50
    ), f"Expected 50 validation samples in DEBUG mode, got {len(val_df)}"
    print("Data loaded successfully.")

    # -------------------------------------------------------------------------
    # 2. Feature Extraction Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Meta-Feature Extraction...")
    extractor = MetaFeatureExtractor()
    # Test on a small batch
    sample_texts = train_df["text"].iloc[:10].tolist()
    features = extractor.extract(sample_texts)

    # Expecting 3 features: char_len, word_count, punct_count
    assert features.shape == (10, 3), f"Expected shape (10, 3), got {features.shape}"
    print("Meta-features extracted successfully.")

    # -------------------------------------------------------------------------
    # 3. TF-IDF Expert Verification
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying TF-IDF Expert...")
    tfidf_model = TfidfExpert()

    # Manually trigger feature generation (usually handled by pipeline)
    # We use the small debug dataframes
    X_train, X_val, X_test = tfidf_model.get_features(
        train_df["text"], val_df["text"], test_df["text"], load_cached_data=False
    )

    y_train = train_df["author"].map(Config.LABEL2ID).values
    y_val = val_df["author"].map(Config.LABEL2ID).values

    tfidf_model.fit(X_train, y_train)

    # Validate
    loss, acc = tfidf_model.validate(X_val, y_val)
    probs = tfidf_model.predict_proba(X_test)

    assert probs.shape == (len(test_df), 3), "TF-IDF prediction shape mismatch"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of bounds"
    print(f"TF-IDF Expert verified. Val Loss: {loss:.4f}")

    # -------------------------------------------------------------------------
    # 4. Transformer Expert Verification
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying Transformer Expert (Trainer)...")
    # Use an even smaller subset for the transformer demo to ensure speed
    t_train = train_df.iloc[:16]
    t_val = val_df.iloc[:8]

    trainer = Trainer()

    # Fit for 1 epoch (Config.EPOCHS=1)
    best_loss = trainer.fit(
        t_train["text"],
        t_train["author"],
        t_val["text"],
        t_val["author"],
        fold_idx="demo",
    )

    # Predict
    preds = trainer.predict(t_val["text"])
    assert preds.shape == (8, 3), "Transformer prediction shape mismatch"

    # Check if model file was saved
    model_path = os.path.join(Config.WORKING_DIR, "transformer_fold_demo.pt")
    assert os.path.exists(model_path), "Transformer model file not saved"
    print(f"Transformer Expert verified. Best Loss: {best_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Pipeline Manager (End-to-End) Verification
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Full Pipeline Manager...")
    # The PipelineManager handles CV, Stacking, and Final Submission
    # It will reload data, so it uses the DEBUG size (200 rows)
    pm = PipelineManager()

    # Execute the full pipeline
    pm.execute()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (
        len(test_df),
        4,
    ), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == [
        "id",
        "EAP",
        "HPL",
        "MWS",
    ], "Submission columns mismatch"

    print("Full Pipeline verified successfully.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_demo()
