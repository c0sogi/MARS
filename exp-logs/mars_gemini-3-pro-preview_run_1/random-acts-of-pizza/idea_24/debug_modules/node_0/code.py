import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_engineering as fe
import library.model_rf as model_rf
import library.model_mlp as model_mlp
import library.trainer as trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of Pizza Request Prediction Pipeline ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Set seeds
    utils.seed_everything(42)

    # Modify global config for speed (Monkey-patching)
    print("   - Reducing model hyperparameters for speed...")
    config.RF_PARAMS["n_estimators"] = 10  # Reduce trees
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead in small demo

    config.MLP_PARAMS["epochs"] = 1  # 1 Epoch only
    config.MLP_PARAMS["batch_size"] = 8  # Small batch
    config.MLP_PARAMS["hidden_dim"] = 16  # Small hidden dim
    config.MLP_PARAMS["patience"] = 1  # Fail fast

    # We will use a separate working directory for the demo to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    config.WORKING_DIR = demo_working_dir
    os.makedirs(demo_working_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Subsampling
    # -------------------------------------------------------------------------
    print("\n2. Loading and Subsampling Data...")

    # Load original data
    # We force load_cached_data=False to ensure we read from metadata CSVs
    train_full, val_full, test_full = data_loader.load_data(load_cached_data=False)

    # Subsample for demo speed (50 train, 20 val, 20 test)
    train_subset = train_full.head(50).copy().reset_index(drop=True)
    val_subset = val_full.head(20).copy().reset_index(drop=True)
    test_subset = test_full.head(20).copy().reset_index(drop=True)

    print(f"   - Train subset shape: {train_subset.shape}")
    print(f"   - Val subset shape: {val_subset.shape}")
    print(f"   - Test subset shape: {test_subset.shape}")

    # Monkey-patch load_data so that internal calls in trainer.py use our subsets
    # This is crucial for the high-level pipeline demo later
    def mocked_load_data(load_cached_data=True):
        print("   [Mock] Returning subsampled dataframes.")
        return train_subset, val_subset, test_subset

    data_loader.load_data = mocked_load_data

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Components
    # -------------------------------------------------------------------------
    print("\n3. Verifying Feature Engineering Components...")

    # A. MetadataExtractor
    print("   A. Testing MetadataExtractor...")
    meta_extractor = fe.MetadataExtractor()
    meta_train = meta_extractor.process(
        train_subset, "train_demo", load_cached_data=False
    )

    assert isinstance(meta_train, pd.DataFrame), "Metadata output should be a DataFrame"
    assert (
        "upvote_ratio" in meta_train.columns
    ), "Engineered feature 'upvote_ratio' missing"
    assert len(meta_train) == len(train_subset), "Metadata row count mismatch"
    print("      -> MetadataExtractor passed.")

    # B. TextProcessor (TF-IDF)
    print("   B. Testing TextProcessor...")
    text_proc = fe.TextProcessor()
    tfidf_train = text_proc.fit_transform(
        train_subset, "train_demo", load_cached_data=False
    )
    tfidf_test = text_proc.transform(test_subset, "test_demo", load_cached_data=False)

    assert scipy.sparse.issparse(tfidf_train), "TF-IDF output should be sparse matrix"
    assert tfidf_train.shape[0] == len(train_subset), "TF-IDF row count mismatch"
    assert (
        tfidf_train.shape[1] == tfidf_test.shape[1]
    ), "Feature dimension mismatch between train/test"
    print("      -> TextProcessor passed.")

    # C. BayesianHistoryEncoder
    print("   C. Testing BayesianHistoryEncoder...")
    bayes_enc = fe.BayesianHistoryEncoder()
    # Test fit_transform_cv (for training)
    hist_train = bayes_enc.fit_transform_train(
        train_subset, "train_demo", load_cached_data=False
    )
    # Test transform (for inference)
    hist_test = bayes_enc.transform(test_subset, "test_demo", load_cached_data=False)

    assert "hist_mean_success" in hist_train.columns, "Bayesian feature missing"
    assert not hist_train.isnull().values.any(), "Bayesian features contain NaNs"
    assert len(hist_train) == len(train_subset), "Bayesian row count mismatch"
    print("      -> BayesianHistoryEncoder passed.")

    # D. SBERTHandler
    print("   D. Testing SBERTHandler (Embeddings)...")
    sbert = fe.SBERTHandler()
    # Mock the model loading to avoid downloading/loading heavy weights if possible?
    # No, we must use the real one provided. It's pre-installed.
    # We'll rely on the small dataset size to keep it fast.

    req_emb = sbert.encode_requests(train_subset, "train_demo", load_cached_data=False)
    hist_emb = sbert.encode_history(train_subset, "train_demo", load_cached_data=False)

    assert req_emb.shape == (
        len(train_subset),
        config.SBERT_EMBEDDING_DIM,
    ), "Request embedding shape incorrect"
    assert hist_emb.shape == (
        len(train_subset),
        config.MAX_HISTORY_LEN,
        config.SBERT_EMBEDDING_DIM,
    ), "History embedding shape incorrect"
    print("      -> SBERTHandler passed.")

    # -------------------------------------------------------------------------
    # 4. Model Stream A: Random Forest
    # -------------------------------------------------------------------------
    print("\n4. Verifying RF Stream...")

    rf_model_wrapper = model_rf.RFModel()

    # Train
    print("   - Training RF...")
    rf_auc = rf_model_wrapper.train(train_subset, val_subset, load_cached_data=False)
    assert isinstance(rf_auc, float), "AUC should be a float"
    assert 0 <= rf_auc <= 1, "AUC should be between 0 and 1"

    # Predict
    print("   - Predicting with RF...")
    rf_preds = rf_model_wrapper.predict_proba(test_subset, load_cached_data=False)
    assert len(rf_preds) == len(test_subset), "Prediction count mismatch"
    assert np.all((rf_preds >= 0) & (rf_preds <= 1)), "Probabilities out of range"
    print("      -> RF Stream passed.")

    # -------------------------------------------------------------------------
    # 5. Model Stream B: MLP (Gated Fusion)
    # -------------------------------------------------------------------------
    print("\n5. Verifying MLP Stream...")

    mlp_model_wrapper = model_mlp.MLPModel()

    # Train
    print("   - Training MLP...")
    mlp_auc = mlp_model_wrapper.train(train_subset, val_subset, load_cached_data=False)
    assert isinstance(mlp_auc, float), "AUC should be a float"

    # Predict
    print("   - Predicting with MLP...")
    mlp_preds = mlp_model_wrapper.predict_proba(test_subset, load_cached_data=False)
    assert len(mlp_preds) == len(test_subset), "Prediction count mismatch"
    assert mlp_preds.shape == (len(test_subset),), "Prediction shape mismatch"
    print("      -> MLP Stream passed.")

    # -------------------------------------------------------------------------
    # 6. High-Level Pipeline Integration
    # -------------------------------------------------------------------------
    print("\n6. Verifying High-Level Trainer Pipeline...")

    # We use the mocked load_data here to run the full pipeline quickly
    # This tests the integration of all parts via trainer.py

    # Override submission path to demo directory
    original_sub_path = config.SUBMISSION_PATH
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "demo_submission.csv")

    try:
        trainer.run_training_pipeline(load_cached_data=False)

        # Check if submission file was created
        assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

        # Verify content
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        assert len(sub_df) == len(test_subset), "Submission row count mismatch"
        assert config.ID_COL in sub_df.columns, "Request ID column missing"
        assert config.TARGET_COL in sub_df.columns, "Target column missing"

        print("      -> Trainer Pipeline passed.")

    finally:
        # Restore config
        config.SUBMISSION_PATH = original_sub_path

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
