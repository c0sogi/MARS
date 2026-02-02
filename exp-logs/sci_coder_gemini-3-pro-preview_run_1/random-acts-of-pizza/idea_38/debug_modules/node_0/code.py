import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_json_data, save_submission
from library.text_encoders import SBERTEmbedder, TFIDFHandler, SentimentAnalyzer
from library.feature_engine import (
    MetadataProcessor,
    HistoryProcessor,
    PrototypeComputer,
)
from library.rf_learner import RFLearner
from library.mlp_learner import MLPLearner


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Demo Configuration...")

    # Override Config for speed (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for quick verification

    # Use a specific directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "output", "demo_submission.csv")

    # Reduce Model Complexity for Demo
    Config.RF_PARAMS["n_estimators"] = 10
    Config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead in small demo

    Config.MLP_PARAMS["epochs"] = 2
    Config.MLP_PARAMS["batch_size"] = 8
    Config.MLP_PARAMS["hidden_dim"] = 32
    Config.MLP_PARAMS["patience"] = 1

    # Set global seed
    set_seed(Config.RANDOM_SEED)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nLoading Data...")
    # Load data (this will use the DEBUG_SAMPLE_SIZE)
    train_df, val_df, test_df = load_json_data(Config, load_cached_data=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Assertions
    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE
    assert "requester_received_pizza" in train_df.columns
    assert "request_text_edit_aware" in test_df.columns

    # Extract labels
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    # ==========================================
    # 3. Feature Engineering
    # ==========================================
    print("\nRunning Feature Engineering...")

    # A. Text Embeddings (SBERT)
    print("-> Generating SBERT Embeddings...")
    sbert = SBERTEmbedder(cache_dir=Config.WORKING_DIR)

    # Title
    train_title_emb = sbert.encode_text(
        train_df, Config.TEXT_COL_TITLE, "demo_train", load_cached_data=False
    )
    val_title_emb = sbert.encode_text(
        val_df, Config.TEXT_COL_TITLE, "demo_val", load_cached_data=False
    )
    test_title_emb = sbert.encode_text(
        test_df, Config.TEXT_COL_TITLE, "demo_test", load_cached_data=False
    )

    # Body
    train_body_emb = sbert.encode_text(
        train_df, Config.TEXT_COL_BODY, "demo_train", load_cached_data=False
    )
    val_body_emb = sbert.encode_text(
        val_df, Config.TEXT_COL_BODY, "demo_val", load_cached_data=False
    )
    test_body_emb = sbert.encode_text(
        test_df, Config.TEXT_COL_BODY, "demo_test", load_cached_data=False
    )

    # History (Sequence of embeddings)
    # Note: We limit max_len for speed in demo
    print("-> Generating History Embeddings...")
    hist_col = "requester_subreddits_at_request"
    train_hist_emb, train_hist_mask = sbert.encode_history(
        train_df, hist_col, "demo_train", load_cached_data=False, max_len=10
    )
    val_hist_emb, val_hist_mask = sbert.encode_history(
        val_df, hist_col, "demo_val", load_cached_data=False, max_len=10
    )
    test_hist_emb, test_hist_mask = sbert.encode_history(
        test_df, hist_col, "demo_test", load_cached_data=False, max_len=10
    )

    # B. TF-IDF
    print("-> Generating TF-IDF Features...")
    tfidf = TFIDFHandler(
        vocab_size=100, cache_dir=Config.WORKING_DIR
    )  # Small vocab for demo
    train_tfidf, val_tfidf, test_tfidf = tfidf.process(
        train_df, val_df, test_df, load_cached_data=False
    )

    # C. Sentiment
    print("-> Generating Sentiment Features...")
    sentiment = SentimentAnalyzer(cache_dir=Config.WORKING_DIR)
    train_sent = sentiment.process(train_df, "demo_train", load_cached_data=False)
    val_sent = sentiment.process(val_df, "demo_val", load_cached_data=False)
    test_sent = sentiment.process(test_df, "demo_test", load_cached_data=False)

    # D. Metadata
    print("-> Processing Metadata...")
    meta_proc = MetadataProcessor(cache_dir=Config.WORKING_DIR)
    (rf_meta_train, rf_meta_val, rf_meta_test), (
        mlp_meta_train,
        mlp_meta_val,
        mlp_meta_test,
    ) = meta_proc.process(train_df, val_df, test_df, load_cached_data=False)

    # E. History Top-K
    print("-> Processing History Top-K...")
    hist_proc = HistoryProcessor(top_k=10, cache_dir=Config.WORKING_DIR)
    train_topk, val_topk, test_topk = hist_proc.process(
        train_df, val_df, test_df, load_cached_data=False
    )

    # F. Prototypes
    print("-> Computing Prototypes...")
    proto = PrototypeComputer(cache_dir=Config.WORKING_DIR)
    train_proto, val_proto, test_proto = proto.process(
        train_body_emb,
        train_hist_emb,
        train_hist_mask,
        y_train,
        val_body_emb,
        val_hist_emb,
        val_hist_mask,
        test_body_emb,
        test_hist_emb,
        test_hist_mask,
        load_cached_data=False,
    )

    # ==========================================
    # 4. Stream A: Random Forest
    # ==========================================
    print("\nTraining Stream A: Random Forest...")

    rf_train_comps = {
        "tfidf": train_tfidf,
        "metadata": rf_meta_train,
        "top_k": train_topk,
        "prototypes": train_proto,
        "sentiment": train_sent,
    }
    rf_val_comps = {
        "tfidf": val_tfidf,
        "metadata": rf_meta_val,
        "top_k": val_topk,
        "prototypes": val_proto,
        "sentiment": val_sent,
    }
    rf_test_comps = {
        "tfidf": test_tfidf,
        "metadata": rf_meta_test,
        "top_k": test_topk,
        "prototypes": test_proto,
        "sentiment": test_sent,
    }

    rf_learner = RFLearner(cache_dir=Config.WORKING_DIR)
    rf_model = rf_learner.train(
        rf_train_comps, y_train, rf_val_comps, y_val, load_cached_data=False
    )

    rf_preds = rf_learner.predict(rf_test_comps, load_cached_data=False)

    assert len(rf_preds) == len(test_df)
    assert np.all((rf_preds >= 0) & (rf_preds <= 1))
    print(f"RF Predictions (first 5): {rf_preds[:5]}")

    # ==========================================
    # 5. Stream B: MLP (Dual Query Network)
    # ==========================================
    print("\nTraining Stream B: MLP...")

    mlp_train_comps = {
        "title_emb": train_title_emb,
        "body_emb": train_body_emb,
        "hist_emb": train_hist_emb,
        "hist_mask": train_hist_mask,
        "metadata": mlp_meta_train,
        "prototypes": train_proto,
    }
    mlp_val_comps = {
        "title_emb": val_title_emb,
        "body_emb": val_body_emb,
        "hist_emb": val_hist_emb,
        "hist_mask": val_hist_mask,
        "metadata": mlp_meta_val,
        "prototypes": val_proto,
    }
    mlp_test_comps = {
        "title_emb": test_title_emb,
        "body_emb": test_body_emb,
        "hist_emb": test_hist_emb,
        "hist_mask": test_hist_mask,
        "metadata": mlp_meta_test,
        "prototypes": test_proto,
    }

    mlp_learner = MLPLearner(cache_dir=Config.WORKING_DIR)
    mlp_model = mlp_learner.train(mlp_train_comps, y_train, mlp_val_comps, y_val)

    mlp_preds = mlp_learner.predict(mlp_test_comps)

    assert len(mlp_preds) == len(test_df)
    assert np.all((mlp_preds >= 0) & (mlp_preds <= 1))
    print(f"MLP Predictions (first 5): {mlp_preds[:5]}")

    # ==========================================
    # 6. Ensemble & Submission
    # ==========================================
    print("\nGenerating Ensemble Submission...")

    final_preds = (Config.ENSEMBLE_WEIGHT_RF * rf_preds) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_preds
    )

    save_submission(final_preds, test_df["request_id"].values, Config)

    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    # Verify file exists
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file loaded. Rows: {len(df_sub)}")
        assert len(df_sub) == len(test_df)
        assert "request_id" in df_sub.columns
        assert "requester_received_pizza" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo Completed Successfully!")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    run_demo()
