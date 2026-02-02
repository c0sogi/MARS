import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

# Import library components
from library.config import Config, set_seed
from library.data_loader import load_raw_data
from library.feature_engine import (
    MetadataExtractor,
    TextEmbedder,
    TfidfProcessor,
    InteractionProcessor,
)
from library.dataset import PizzaDataset
from library.neural_net import OrthogonalSkipGatedMLP
from library.tree_model import InteractionProjectedRF
from library.utils import save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demo Script...")

    # 1. Setup and Reproducibility
    set_seed(42)

    # Override Config for speed
    Config.MLP_EPOCHS = 1
    Config.MLP_BATCH_SIZE = 4
    Config.RF_PARAMS["n_estimators"] = 10  # Reduce trees for demo
    Config.VOCAB_SIZE_TFIDF = 100  # Reduce vocab for demo

    # 2. Load Data Subset
    print("\n--- Loading Data Subset ---")
    # We load the raw csvs but only take a few rows to make feature engineering fast
    train_df = load_raw_data(Config.TRAIN_PATH).head(20)
    val_df = load_raw_data(Config.VAL_PATH).head(10)
    test_df = load_raw_data(Config.TEST_PATH).head(10)

    print(f"Train subset shape: {train_df.shape}")
    print(f"Val subset shape: {val_df.shape}")

    # 3. Feature Engineering Demonstration
    print("\n--- Feature Engineering Demo ---")

    # A. Metadata Extractor
    print("Running MetadataExtractor...")
    meta_extractor = MetadataExtractor()
    meta_extractor.fit(train_df)

    train_dense, train_ratio, train_num_raw = meta_extractor.transform(train_df)
    test_dense, test_ratio, test_num_raw = meta_extractor.transform(test_df)

    assert train_dense.shape[0] == 20
    assert train_dense.shape[1] == 6  # 6 numerical columns
    assert train_ratio.shape == (20, 1)
    print("Metadata features verified.")

    # B. Text Embedder (SBERT)
    print("Running TextEmbedder (this may take a moment to load SBERT)...")
    text_embedder = TextEmbedder()

    # Encode titles and bodies
    train_title_emb = text_embedder.encode(train_df["request_title"])
    train_body_emb = text_embedder.encode(train_df["request_text_edit_aware"])
    test_title_emb = text_embedder.encode(test_df["request_title"])
    test_body_emb = text_embedder.encode(test_df["request_text_edit_aware"])

    assert train_title_emb.shape == (20, 384)
    assert train_body_emb.shape == (20, 384)

    # Process History
    train_hist, train_mask, train_cent = text_embedder.process_history(train_df)
    test_hist, test_mask, test_cent = text_embedder.process_history(test_df)

    assert train_hist.shape == (20, 20, 384)  # (B, Seq, Emb)
    assert train_mask.shape == (20, 20)
    assert train_cent.shape == (20, 384)
    print("Text embeddings verified.")

    # C. Interaction Processor
    print("Running InteractionProcessor...")
    interaction_processor = InteractionProcessor()
    interaction_processor.fit(train_df)

    # Top-K
    train_topk = interaction_processor.get_top_k_features(train_df)
    test_topk = interaction_processor.get_top_k_features(test_df)

    # Consistency
    train_cons_t, train_cons_b = interaction_processor.compute_consistency(
        train_title_emb, train_body_emb, train_cent
    )
    test_cons_t, test_cons_b = interaction_processor.compute_consistency(
        test_title_emb, test_body_emb, test_cent
    )

    # Interactions
    train_inter = interaction_processor.get_interactions(
        train_cons_t, train_cons_b, train_num_raw, train_ratio
    )
    test_inter = interaction_processor.get_interactions(
        test_cons_t, test_cons_b, test_num_raw, test_ratio
    )

    assert train_topk.shape[1] == Config.TOP_K_SUBREDDITS
    assert train_inter.shape == (20, 2)
    print("Interaction features verified.")

    # D. TF-IDF Processor
    print("Running TfidfProcessor...")
    tfidf_processor = TfidfProcessor()
    train_tfidf = tfidf_processor.fit_transform(train_df)
    test_tfidf = tfidf_processor.transform(test_df)

    assert train_tfidf.shape == (20, Config.VOCAB_SIZE_TFIDF)
    print("TF-IDF features verified.")

    # 4. Prepare Data Dictionaries
    print("\n--- Assembling Datasets ---")

    # MLP Data
    mlp_train_data = {
        "title": train_title_emb,
        "body": train_body_emb,
        "hist": train_hist,
        "mask": train_mask,
        "cent": train_cent,
        "meta_dense": train_dense,
        "meta_skip": np.hstack([train_dense, train_ratio, train_topk]),
    }

    # RF Data
    rf_train_X = np.hstack(
        [
            train_tfidf,
            train_num_raw,
            train_ratio,
            train_topk,
            train_cons_t,
            train_cons_b,
            train_inter,
        ]
    )

    rf_test_X = np.hstack(
        [
            test_tfidf,
            test_num_raw,
            test_ratio,
            test_topk,
            test_cons_t,
            test_cons_b,
            test_inter,
        ]
    )

    y_train = train_df["requester_received_pizza"].astype(int).values

    # 5. Random Forest Demo
    print("\n--- Random Forest Demo ---")
    rf_model = InteractionProjectedRF(params=Config.RF_PARAMS)

    # Fit
    rf_model.fit(rf_train_X, y_train)
    print("RF Fit complete.")

    # Predict
    # Construct a dummy dict for the predict_rf wrapper or call model directly
    # The library function predict_rf expects a dict with 'rf_test' -> {'X': ...}
    rf_data_dict = {"rf_test": {"X": rf_test_X}}

    # We can use the wrapper method from library.tree_model or the class method
    preds_rf = rf_model.predict_proba(rf_test_X)
    assert len(preds_rf) == 10
    assert np.all((preds_rf >= 0) & (preds_rf <= 1))
    print("RF Predictions generated successfully.")

    # 6. MLP Demo
    print("\n--- MLP Demo ---")

    # Create Dataset
    train_dataset = PizzaDataset(mlp_train_data, labels=y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=True
    )

    # Instantiate Model
    # Determine dimensions dynamically
    metadata_dim = mlp_train_data["meta_dense"].shape[1]
    skip_dim = mlp_train_data["meta_skip"].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = OrthogonalSkipGatedMLP(
        metadata_dim=metadata_dim,
        skip_dim=skip_dim,
        embedding_dim=384,
        hidden_dim=64,  # Reduced for demo
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Training Step
    model.train()
    print("Executing one training epoch...")
    for batch in train_loader:
        # Move to device
        t = batch["title_emb"].to(device)
        b = batch["body_emb"].to(device)
        h = batch["history_emb"].to(device)
        m = batch["history_mask"].to(device)
        p = batch["persona_centroid"].to(device)
        md = batch["metadata_dense"].to(device)
        ms = batch["metadata_skip"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(t, b, h, m, p, md, ms)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    print("MLP training step complete.")

    # Inference Step
    model.eval()
    with torch.no_grad():
        # Use first batch for inference check
        logits = model(t, b, h, m, p, md, ms)
        probs = torch.sigmoid(logits)
        assert probs.shape == y.shape

    print("MLP inference verified.")

    # 7. Submission Demo
    print("\n--- Submission Demo ---")
    # Generate dummy IDs and predictions
    demo_ids = test_df["request_id"].values
    demo_preds = np.random.rand(len(demo_ids))

    output_path = os.path.join(Config.CACHE_DIR, "demo_submission.csv")
    save_submission(demo_ids, demo_preds, output_path=output_path)

    # Verify file existence
    assert os.path.exists(output_path)
    print(f"Submission file verified at {output_path}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
