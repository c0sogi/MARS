import os
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from library
from library.utils import set_seed, get_device
from library.data_manager import get_clean_data
from library.feature_engineers import (
    MetadataExtractor,
    TopKSubredditEncoder,
    TextProcessor,
)
from library.torch_dataset import get_pizza_datasets, PizzaDataset
from library.neural_model import GatedPizzaNetwork, train_neural_model
from library.rf_model import RFPipeline
from library.execution import evaluate_ensemble


def run_demo():
    print("=== Starting Demonstration of Pizza Request Prediction Library ===")

    # 1. Setup
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    # Define debug parameters
    DEBUG_SIZE = 10
    CACHE_DIR = "./working/idea_25/"

    # Clean up cache for fresh demo to ensure we test computation logic
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    # ---------------------------------------------------------
    # 2. Data Manager
    # ---------------------------------------------------------
    print("\n--- Testing Data Manager ---")
    # We use load_cached_data=False to force loading from raw metadata and processing
    df_train, df_val, df_test = get_clean_data(
        load_cached_data=False, debug_mode=True, debug_size=DEBUG_SIZE
    )

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    assert len(df_train) == DEBUG_SIZE
    assert "requester_received_pizza" in df_train.columns
    assert "request_text_edit_aware" in df_train.columns

    # ---------------------------------------------------------
    # 3. Feature Engineers
    # ---------------------------------------------------------
    print("\n--- Testing Feature Engineers ---")

    # A. Metadata Extractor
    print("Testing MetadataExtractor...")
    meta_extractor = MetadataExtractor()
    # Force computation
    meta_train, meta_val, meta_test = meta_extractor.process(
        df_train, df_val, df_test, load_cached_data=False
    )

    assert len(meta_train) == DEBUG_SIZE
    assert "text_word_count" in meta_train.columns
    assert "upvote_ratio" in meta_train.columns

    # B. TopK Subreddit Encoder
    print("Testing TopKSubredditEncoder...")
    topk_encoder = TopKSubredditEncoder(k=5)  # Small k for debug
    topk_train, topk_val, topk_test = topk_encoder.process(
        df_train, df_val, df_test, load_cached_data=False
    )

    assert len(topk_train) == DEBUG_SIZE
    # Columns should start with 'sub_' if any subreddits were found
    if topk_train.shape[1] > 0:
        assert topk_train.columns[0].startswith("sub_")

    # C. Text Processor
    print("Testing TextProcessor...")
    text_processor = TextProcessor()

    # TF-IDF
    print("Computing TF-IDF...")
    tfidf_train, tfidf_val, tfidf_test = text_processor.process_tfidf(
        df_train, df_val, df_test, load_cached_data=False
    )
    assert tfidf_train.shape[0] == DEBUG_SIZE
    assert isinstance(tfidf_train, np.ndarray)

    # SBERT Request
    print("Computing SBERT Request Embeddings...")
    req_train, req_val, req_test = text_processor.process_sbert_request(
        df_train, df_val, df_test, load_cached_data=False
    )
    assert req_train.shape == (DEBUG_SIZE, 384)

    # SBERT History
    print("Computing SBERT History Embeddings...")
    hist_train, hist_val, hist_test = text_processor.process_sbert_history(
        df_train, df_val, df_test, load_cached_data=False
    )
    assert hist_train.shape == (DEBUG_SIZE, 20, 384)

    # ---------------------------------------------------------
    # 4. Torch Dataset & DataLoader
    # ---------------------------------------------------------
    print("\n--- Testing Torch Dataset ---")

    # Get scaled metadata for the dataset
    X_meta_train, _, _ = meta_extractor.get_scaled_features(
        meta_train, meta_val, meta_test
    )
    y_train = df_train["requester_received_pizza"].astype(int).values

    train_ds = PizzaDataset(
        request_emb=req_train,
        history_seq=hist_train,
        metadata=X_meta_train,
        labels=y_train,
    )

    assert len(train_ds) == DEBUG_SIZE
    sample = train_ds[0]
    assert "request_emb" in sample
    assert "history_seq" in sample
    assert "history_mask" in sample
    assert "metadata" in sample
    assert "label" in sample
    assert sample["request_emb"].shape == (384,)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    batch = next(iter(train_loader))
    assert batch["request_emb"].shape[0] == 2

    # ---------------------------------------------------------
    # 5. Neural Model
    # ---------------------------------------------------------
    print("\n--- Testing Neural Model ---")

    input_dims = {"text_dim": 384, "meta_dim": X_meta_train.shape[1]}

    model = GatedPizzaNetwork(
        text_dim=input_dims["text_dim"],
        meta_dim=input_dims["meta_dim"],
        hidden_dim=32,
        dropout_rate=0.1,
    ).to(device)

    # Forward pass check
    req = batch["request_emb"].to(device)
    hist = batch["history_seq"].to(device)
    mask = batch["history_mask"].to(device)
    meta = batch["metadata"].to(device)

    with torch.no_grad():
        logits = model(req, hist, mask, meta)

    assert logits.shape == (2, 1)

    # Training Loop Check
    print("Testing training loop (1 epoch)...")
    # Need val loader
    # Using simple transformation for validation metadata to match train
    val_meta_scaled = meta_extractor.scaler.transform(np.arcsinh(meta_val))
    y_val = df_val["requester_received_pizza"].astype(int).values

    val_ds = PizzaDataset(req_val, hist_val, val_meta_scaled, y_val)
    val_loader = DataLoader(val_ds, batch_size=2)

    config = {
        "lr": 1e-3,
        "epochs": 1,
        "patience": 1,
        "hidden_dim": 32,
        "dropout": 0.1,
        "weight_decay": 1e-4,
    }

    trained_model, history = train_neural_model(
        train_loader, val_loader, input_dims, config
    )
    assert "train_loss" in history
    assert "val_auc" in history

    # ---------------------------------------------------------
    # 6. Random Forest Pipeline
    # ---------------------------------------------------------
    print("\n--- Testing Random Forest Pipeline ---")

    rf_pipeline = RFPipeline(n_estimators=10, min_samples_leaf=1)

    # We use the pipeline's get_data which aggregates everything
    # Using debug_mode=True ensures we don't process the whole dataset
    rf_data = rf_pipeline.get_data(
        load_cached_data=False, debug_mode=True, debug_size=DEBUG_SIZE
    )

    assert rf_data["X_train"].shape[0] == DEBUG_SIZE

    # Train
    auc = rf_pipeline.train(
        rf_data["X_train"], rf_data["y_train"], rf_data["X_val"], rf_data["y_val"]
    )
    print(f"RF Debug AUC: {auc}")

    # Predict
    preds = rf_pipeline.predict(rf_data["X_test"])
    assert len(preds) == DEBUG_SIZE

    # ---------------------------------------------------------
    # 7. Full Execution Ensemble
    # ---------------------------------------------------------
    print("\n--- Testing Full Ensemble Execution ---")

    # This runs both pipelines end-to-end using the library's orchestration
    metrics = evaluate_ensemble(
        load_cached_data=False,
        debug_mode=True,
        debug_size=DEBUG_SIZE,
        epochs=1,
        patience=1,
    )

    assert "rf_auc" in metrics
    assert "mlp_auc" in metrics
    assert os.path.exists("./submission/submission.csv")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
