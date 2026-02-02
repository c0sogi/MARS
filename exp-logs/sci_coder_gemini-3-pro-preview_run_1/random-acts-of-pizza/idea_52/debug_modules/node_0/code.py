import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Configuration Overrides (Monkey-patching for speed)
import library.config as config

print(">>> Configuring environment for rapid demonstration...")
config.MLP_EPOCHS = 2  # Reduce epochs
config.RF_N_ESTIMATORS = 10  # Reduce RF trees
config.DEBUG_SAMPLE_SIZE = 50  # Use small data subset
config.MLP_BATCH_SIZE = 8  # Small batch size for small data
config.WORKING_DIR = "./working/demo_execution"  # Separate working dir
config.SUBMISSION_DIR = "./working/demo_execution/output"
config.SUBMISSION_FILE = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

# Ensure directories exist
os.makedirs(config.WORKING_DIR, exist_ok=True)
os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

# Import library modules after config update
from library.utils import set_seed, get_device
from library.data_loader import load_data
from library.text_encoder import TextEncoder
from library.feature_engineer import FeatureEngineer
from library.dataset import PizzaDataset
from library.mlp_model import OrthogonalSkipGatedMLP
from library.rf_model import InteractionRandomForest
from library.trainer import train_models, predict_ensemble, save_submission


def run_demo():
    # Set seed for reproducibility
    set_seed(config.RANDOM_STATE)
    device = get_device()
    print(f"Running on device: {device}")

    # =========================================================================
    # 1. Data Loading
    # =========================================================================
    print("\n>>> Loading Data (Debug Mode)...")
    # load_cached_data=False ensures we run the processing logic
    train_df, val_df, test_df = load_data(load_cached_data=False, debug=True)

    # Validation
    assert len(train_df) == config.DEBUG_SAMPLE_SIZE
    assert len(val_df) == config.DEBUG_SAMPLE_SIZE
    assert len(test_df) == config.DEBUG_SAMPLE_SIZE
    print("Data loaded successfully.")

    # =========================================================================
    # 2. Text Encoding (SBERT)
    # =========================================================================
    print("\n>>> Encoding Text Features...")
    encoder = TextEncoder()

    # Encode train, val, test
    # We use unique names to avoid conflicts in cache
    sbert_train = encoder.encode(train_df, "demo_train", load_cached_data=False)
    sbert_val = encoder.encode(val_df, "demo_val", load_cached_data=False)
    sbert_test = encoder.encode(test_df, "demo_test", load_cached_data=False)

    # Validation
    expected_dim = config.EMBEDDING_DIM
    assert sbert_train["title_emb"].shape == (config.DEBUG_SAMPLE_SIZE, expected_dim)
    assert sbert_train["hist_seq"].shape == (
        config.DEBUG_SAMPLE_SIZE,
        config.MAX_HISTORY_LENGTH,
        expected_dim,
    )
    print("Text features encoded successfully.")

    # =========================================================================
    # 3. Feature Engineering (Tabular + Interactions)
    # =========================================================================
    print("\n>>> Generating Tabular Features...")
    fe = FeatureEngineer()

    # Fit on training data and transform
    tabular_train = fe.generate_features(
        train_df,
        "demo_train",
        sbert_features=sbert_train,
        train_df=train_df,
        load_cached_data=False,
    )

    # Transform val and test
    tabular_val = fe.generate_features(
        val_df, "demo_val", sbert_features=sbert_val, load_cached_data=False
    )

    tabular_test = fe.generate_features(
        test_df, "demo_test", sbert_features=sbert_test, load_cached_data=False
    )

    # Validation
    assert "mlp_metadata" in tabular_train
    assert "rf_tfidf" in tabular_train
    assert "rf_interactions" in tabular_train
    print("Tabular features generated successfully.")

    # =========================================================================
    # 4. Dataset & Model Component Verification
    # =========================================================================
    print("\n>>> Verifying Dataset and MLP Architecture...")

    # Create a dummy dataset instance
    y_train = train_df["requester_received_pizza"].values.astype(int)
    ds = PizzaDataset(sbert_train, tabular_train, labels=y_train)

    # Check __getitem__
    sample = ds[0]
    assert torch.is_tensor(sample["title_emb"])
    assert sample["label"].shape == (1,)

    # Instantiate MLP Model
    metadata_dim = tabular_train["mlp_metadata"].shape[1]
    model = OrthogonalSkipGatedMLP(metadata_dim).to(device)

    # Run a dummy forward pass
    # Add batch dimension
    title = sample["title_emb"].unsqueeze(0).to(device)
    body = sample["body_emb"].unsqueeze(0).to(device)
    hist_seq = sample["hist_seq"].unsqueeze(0).to(device)
    hist_mask = sample["hist_mask"].unsqueeze(0).to(device)
    centroid = sample["hist_centroid"].unsqueeze(0).to(device)
    meta = sample["metadata"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(title, body, hist_seq, hist_mask, centroid, meta)

    assert logits.shape == (1, 1)
    print("MLP architecture verified successfully.")

    # =========================================================================
    # 5. Full Training Pipeline
    # =========================================================================
    print("\n>>> Executing Training Pipeline (MLP + RF)...")

    mlp_model, rf_model = train_models(
        train_df,
        val_df,
        sbert_train,
        sbert_val,
        tabular_train,
        tabular_val,
        save_models=True,
    )

    print("Models trained successfully.")

    # =========================================================================
    # 6. Prediction & Submission
    # =========================================================================
    print("\n>>> Generating Predictions and Submission...")

    final_preds = predict_ensemble(
        mlp_model, rf_model, test_df, sbert_test, tabular_test
    )

    assert len(final_preds) == len(test_df)
    assert np.all((final_preds >= 0) & (final_preds <= 1))

    save_submission(test_df, final_preds)

    assert os.path.exists(config.SUBMISSION_FILE)
    print(f"Demo completed successfully. Submission saved to {config.SUBMISSION_FILE}")


if __name__ == "__main__":
    run_demo()
