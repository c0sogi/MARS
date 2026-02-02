import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
from torch.utils.data import DataLoader

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Import library modules
# We import config first to override settings before other modules use them
import library.config as config

# ==========================================
# 1. Configuration Overrides for Demo
# ==========================================
print("Setting up configuration for rapid demonstration...")
config.DEBUG = True  # Use only 50 rows
config.EPOCHS = 1
config.BATCH_SIZE = 4
config.RF_ESTIMATORS = 5
config.TFIDF_VOCAB_SIZE = 50
config.PCA_COMPONENTS = 5
config.WORKING_DIR = "./working/demo_execution"  # Isolate demo artifacts

# Clean up demo directory if it exists to ensure fresh run
if os.path.exists(config.WORKING_DIR):
    shutil.rmtree(config.WORKING_DIR)
os.makedirs(config.WORKING_DIR, exist_ok=True)

# Now import the rest of the library
from library.data_loader import load_dataset
from library.features import FeatureProcessor
from library.model_rf import RFWrapper
from library.model_mlp import PizzaDataset, MLPTrainer

if __name__ == "__main__":
    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n=== 1. Data Loading ===")
    # Load data with caching disabled to force processing logic
    train_df, val_df, test_df = load_dataset(load_cached_data=False)

    # Verification
    print(f"Train rows: {len(train_df)}")
    assert len(train_df) == 50, f"Expected 50 rows in DEBUG mode, got {len(train_df)}"
    assert len(val_df) == 50
    assert "requester_subreddits_at_request" in train_df.columns
    # Check if list parsing worked (it should be a list, not a string)
    sample_sub = train_df.iloc[0]["requester_subreddits_at_request"]
    assert isinstance(sample_sub, list), f"Expected list, got {type(sample_sub)}"
    print("Data loaded and parsed successfully.")

    # ==========================================
    # 3. Feature Processing Demonstration
    # ==========================================
    print("\n=== 2. Feature Processing ===")
    processor = FeatureProcessor()

    # Process data (this runs SBERT, TF-IDF, PCA, etc.)
    # We pass load_cached_data=False to ensure the pipeline runs
    train_data, val_data, test_data = processor.process(load_cached_data=False)

    # Verification of Data Structures
    required_keys = [
        "rf_features",
        "mlp_request_emb",
        "mlp_history_seq",
        "mlp_history_mask",
        "mlp_metadata",
        "labels",
    ]

    for key in required_keys:
        assert key in train_data, f"Missing key '{key}' in train_data"
        assert key in val_data, f"Missing key '{key}' in val_data"

    # Check shapes
    n_samples = 50
    assert train_data["rf_features"].shape[0] == n_samples
    assert train_data["mlp_request_emb"].shape[0] == n_samples
    assert train_data["mlp_metadata"].shape[0] == n_samples

    # Check sparse matrix type for RF
    import scipy.sparse

    assert scipy.sparse.issparse(
        train_data["rf_features"]
    ), "RF features should be sparse"

    print("Feature processing complete. Shapes verified.")

    # ==========================================
    # 4. Random Forest Model Demonstration
    # ==========================================
    print("\n=== 3. Random Forest Model ===")
    rf_wrapper = RFWrapper()

    # Train
    print("Training RF...")
    rf_wrapper.train(
        train_data["rf_features"],
        train_data["labels"],
        val_data["rf_features"],
        val_data["labels"],
    )

    # Predict
    print("Predicting with RF...")
    rf_preds = rf_wrapper.predict(val_data["rf_features"])

    # Verify predictions
    assert len(rf_preds) == n_samples
    assert np.all(
        (rf_preds >= 0) & (rf_preds <= 1)
    ), "RF predictions out of probability range"

    # Save/Load
    rf_path = os.path.join(config.WORKING_DIR, "rf_model.joblib")
    rf_wrapper.save(rf_path)
    assert os.path.exists(rf_path), "RF model file not created"

    # Reload to verify
    rf_wrapper.load(rf_path)
    print("Random Forest pipeline verified.")

    # ==========================================
    # 5. MLP Model Demonstration
    # ==========================================
    print("\n=== 4. MLP Model ===")

    # Create Datasets
    train_ds = PizzaDataset(train_data, mode="train")
    val_ds = PizzaDataset(val_data, mode="val")

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize Trainer
    # Determine metadata dimension from data
    meta_dim = train_data["mlp_metadata"].shape[1]
    device = torch.device(
        "cpu"
    )  # Use CPU for demo to avoid CUDA overhead/issues in small test

    trainer = MLPTrainer(metadata_dim=meta_dim, device=device)

    # Train (1 epoch as per config override)
    print("Training MLP...")
    best_auc = trainer.train(train_loader, val_loader)
    print(f"Training complete. Best Validation AUC: {best_auc:.4f}")

    # Predict
    print("Predicting with MLP...")
    mlp_preds = trainer.predict(val_loader)

    # Verify predictions
    assert len(mlp_preds) == n_samples
    assert mlp_preds.shape == (
        n_samples,
    ), f"Expected shape ({n_samples},), got {mlp_preds.shape}"
    assert np.all(
        (mlp_preds >= 0) & (mlp_preds <= 1)
    ), "MLP predictions out of probability range"

    # Save/Load
    mlp_path = os.path.join(config.WORKING_DIR, "mlp_model.pth")
    trainer.save(mlp_path)
    assert os.path.exists(mlp_path), "MLP model file not created"

    trainer.load(mlp_path)
    print("MLP pipeline verified.")

    print("\n=== Demonstration Complete Successfully ===")
