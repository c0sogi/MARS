import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import (
    config,
    utils,
    features_text,
    features_meta,
    dataset,
    model_mlp,
    model_rf,
    trainer,
)


def run_demo():
    print("--- 1. Setup and Configuration Overrides ---")
    # Override config for speed in this demonstration
    config.SEED = 42
    config.MLP_EPOCHS = 2
    config.MLP_BATCH_SIZE = 8
    config.RF_N_ESTIMATORS = 10
    config.RF_N_JOBS = 1

    # Set seeds
    utils.set_seed(config.SEED)

    # Define debug size to process only a few samples
    DEBUG_SIZE = 50
    print(f"Debug size set to: {DEBUG_SIZE}")

    print("\n--- 2. Data Loading ---")
    # Load data using utility function
    train_df, val_df, test_df = utils.load_data(
        return_val=True,
        parse_list_cols=["requester_subreddits_at_request"],
        debug_size=DEBUG_SIZE,
    )

    # Validation
    assert (
        len(train_df) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} training samples, got {len(train_df)}"
    assert (
        len(test_df) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} test samples, got {len(test_df)}"
    print(
        f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
    )

    print("\n--- 3. Feature Generation ---")
    # Generate Text Features
    # Note: load_cached_data=False forces re-computation for demonstration purposes on the subset
    print("Generating Text Features...")
    text_feats = features_text.generate_text_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation of Text Features
    assert "train_title_emb" in text_feats
    assert text_feats["train_title_emb"].shape == (DEBUG_SIZE, config.EMBEDDING_DIM)
    print(
        f"Text features generated. Title Embedding Shape: {text_feats['train_title_emb'].shape}"
    )

    # Generate Metadata Features
    print("Generating Metadata Features...")
    meta_feats = features_meta.generate_meta_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation of Meta Features
    assert "train_meta_mlp" in meta_feats
    # Check that history sequence has correct shape (N, Max_Seq, Emb_Dim)
    # Max_Seq is defined in features_meta.HistoryProcessor (default 50)
    assert meta_feats["train_hist_seq"].shape[0] == DEBUG_SIZE
    assert meta_feats["train_hist_seq"].shape[2] == config.EMBEDDING_DIM
    print(
        f"Meta features generated. History Sequence Shape: {meta_feats['train_hist_seq'].shape}"
    )

    print("\n--- 4. Stream A: Random Forest Model ---")
    # Assemble features specifically for RF
    print("Assembling RF features...")
    # We can reuse the generated features implicitly via caching or by passing DFs again.
    # The assemble_rf_features function calls generate_* internally.
    # Since we just ran generate_* with load_cached_data=False, the cache might not be populated
    # if the library saves to disk. The library DOES save to disk.
    # However, to be explicit and avoid re-computation logic inside the library affecting the demo flow,
    # we will rely on the fact that the library saves .npz files.

    X_train_rf, X_val_rf, X_test_rf = model_rf.assemble_rf_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Prepare Labels
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    # Validation
    assert X_train_rf.shape[0] == DEBUG_SIZE
    print(f"RF Feature Matrix Shape: {X_train_rf.shape}")

    # Train RF
    print("Training Random Forest...")
    rf_model = model_rf.train_rf(X_train_rf, y_train, X_val_rf, y_val)

    # Predict RF
    rf_preds_val = model_rf.predict_rf(rf_model, X_val_rf)
    assert len(rf_preds_val) == len(y_val)
    print(f"RF Validation Predictions (first 5): {rf_preds_val[:5]}")

    print("\n--- 5. Stream B: Persona-Aware Skip-Gated MLP ---")
    # Create Datasets using the feature dictionaries we generated in Step 3
    # This demonstrates usage of the PizzaDataset class
    print("Creating PyTorch Datasets...")
    train_dataset = dataset.PizzaDataset(
        text_feats, meta_feats, split="train", labels=y_train
    )
    val_dataset = dataset.PizzaDataset(
        text_feats, meta_feats, split="val", labels=y_val
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=False
    )

    # Determine Metadata Dimension for the MLP input
    # It comes from 'dense_metadata' in the dataset item
    sample_item, _ = train_dataset[0]
    meta_dim = sample_item["dense_metadata"].shape[0]
    print(f"MLP Metadata Dimension: {meta_dim}")

    # Initialize and Train MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on {device}...")

    mlp_model = trainer.run_training(
        train_loader,
        val_loader,
        meta_dim=meta_dim,
        device=device,
        epochs=config.MLP_EPOCHS,
    )

    # Validate MLP Logic: Forward pass check
    mlp_model.eval()
    with torch.no_grad():
        # Unsqueeze to add batch dimension for single sample check
        logits = mlp_model(
            sample_item["title_emb"].unsqueeze(0).to(device),
            sample_item["body_emb"].unsqueeze(0).to(device),
            sample_item["history_seq"].unsqueeze(0).to(device),
            sample_item["history_mask"].unsqueeze(0).to(device),
            sample_item["persona_centroid"].unsqueeze(0).to(device),
            sample_item["dense_metadata"].unsqueeze(0).to(device),
        )
        assert logits.shape == (
            1,
            1,
        ), f"Expected output shape (1, 1), got {logits.shape}"
        prob = torch.sigmoid(logits).item()
        assert 0.0 <= prob <= 1.0, "Probability out of range"

    # Predict MLP on Validation
    print("Generating MLP Predictions...")
    mlp_preds_val = model_mlp.predict_mlp(mlp_model, val_loader, device)
    assert len(mlp_preds_val) == len(y_val)
    print(f"MLP Validation Predictions (first 5): {mlp_preds_val[:5]}")

    print("\n--- 6. Ensemble and Evaluation ---")
    # Simple weighted average
    final_preds = (config.WEIGHT_RF * rf_preds_val) + (
        config.WEIGHT_MLP * mlp_preds_val
    )

    # Calculate AUC
    # Handle case where debug subset might have only one class
    if len(np.unique(y_val)) > 1:
        final_auc = roc_auc_score(y_val, final_preds)
        print(f"Final Ensemble Validation AUC: {final_auc:.4f}")
    else:
        print("Skipping AUC calculation: Validation subset contains only one class.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
