import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureEngineer
from library.dataset import create_dataloaders
from library.models import OrthogonalSkipMLP
from library.engine import MLPEngine
from library.rf_pipeline import RFPipeline


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SIZE = 60  # Small subset for demonstration
    Config.MLP_NUM_EPOCHS = 2
    Config.MLP_BATCH_SIZE = 16
    Config.RF_N_ESTIMATORS = 10
    Config.RF_N_JOBS = 1  # Avoid overhead in demo

    # Use a specific cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache"
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Update cache paths in Config to point to the new demo dir
    Config.CACHE_SBERT_EMBEDDINGS = os.path.join(
        Config.CACHE_DIR, "sbert_embeddings.npz"
    )
    Config.CACHE_TFIDF_FEATURES = os.path.join(Config.CACHE_DIR, "tfidf_features.npz")
    Config.CACHE_METADATA_FEATURES = os.path.join(
        Config.CACHE_DIR, "metadata_features.npz"
    )
    Config.CACHE_INTERACTION_FEATURES = os.path.join(
        Config.CACHE_DIR, "interaction_features.npz"
    )
    Config.CACHE_PERSONA_FEATURES = os.path.join(
        Config.CACHE_DIR, "persona_features.npz"
    )

    seed_everything(Config.RANDOM_SEED)
    print("Configuration updated. Debug mode enabled.")

    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("\n[2] Running Feature Engineering...")
    fe = FeatureEngineer()

    # Run feature engineering (force recompute to demonstrate logic)
    data_dict = fe.run(load_cached_data=False)

    # Validation
    splits = ["train", "val", "test"]
    for split in splits:
        assert split in data_dict, f"Missing split {split} in data_dict"
        assert "X_rf" in data_dict[split], f"Missing RF features for {split}"
        assert "X_mlp_title" in data_dict[split], f"Missing MLP features for {split}"

        # Check sample counts match DEBUG_SIZE (or less if source is smaller)
        n_samples = len(data_dict[split]["X_rf"])
        print(f"  - {split}: {n_samples} samples processed.")
        if n_samples > Config.DEBUG_SIZE:
            raise AssertionError(
                f"Expected <= {Config.DEBUG_SIZE} samples, got {n_samples}"
            )

    print("Feature Engineering successful.")

    # ---------------------------------------------------------
    # 3. Data Loading (MLP Stream)
    # ---------------------------------------------------------
    print("\n[3] Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict,
        batch_size=Config.MLP_BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debug execution
    )

    # Verify a batch
    sample_batch = next(iter(train_loader))
    required_keys = [
        "title",
        "body",
        "history",
        "history_mask",
        "centroid",
        "meta",
        "consistency",
        "target",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Batch missing key: {key}"

    print(f"  - Batch keys verified: {list(sample_batch.keys())}")
    print(f"  - Title shape: {sample_batch['title'].shape}")
    print(f"  - Target shape: {sample_batch['target'].shape}")

    # ---------------------------------------------------------
    # 4. Model Initialization (MLP)
    # ---------------------------------------------------------
    print("\n[4] Initializing OrthogonalSkipMLP...")
    model = OrthogonalSkipMLP()

    # Test forward pass with sample batch
    model.eval()
    with torch.no_grad():
        logits = model(sample_batch)

    assert logits.shape == (
        sample_batch["title"].shape[0],
        1,
    ), f"Output shape mismatch: {logits.shape}"
    print("Model forward pass successful.")

    # ---------------------------------------------------------
    # 5. MLP Engine Execution
    # ---------------------------------------------------------
    print("\n[5] Running MLP Engine (Train/Val/Predict)...")
    mlp_engine = MLPEngine()

    # Run the engine
    mlp_results = mlp_engine.run(train_loader, val_loader, test_loader)

    # Validate results
    assert "val_auc" in mlp_results
    assert "val_probs" in mlp_results
    assert "test_probs" in mlp_results
    assert len(mlp_results["test_probs"]) == len(data_dict["test"]["X_rf"])

    print(f"MLP Run Complete. Val AUC: {mlp_results['val_auc']:.4f}")

    # ---------------------------------------------------------
    # 6. Random Forest Pipeline Execution
    # ---------------------------------------------------------
    print("\n[6] Running Random Forest Pipeline...")
    rf_pipeline = RFPipeline()

    rf_results = rf_pipeline.run(data_dict)

    # Validate results
    assert "val_auc" in rf_results
    assert "test_probs" in rf_results
    assert len(rf_results["test_probs"]) == len(data_dict["test"]["X_rf"])

    print(f"RF Run Complete. Val AUC: {rf_results['val_auc']:.4f}")

    # ---------------------------------------------------------
    # 7. Ensemble & Submission Generation
    # ---------------------------------------------------------
    print("\n[7] Generating Ensemble Predictions...")

    # Simple weighted average
    w_rf, w_mlp = Config.ENSEMBLE_WEIGHTS
    final_test_probs = (w_rf * rf_results["test_probs"]) + (
        w_mlp * mlp_results["test_probs"]
    )

    # Verify range
    assert np.all(final_test_probs >= 0.0) and np.all(
        final_test_probs <= 1.0
    ), "Probabilities out of bounds"

    # Create submission dataframe (concept)
    # We need request_ids from the original test csv to map predictions
    df_test = pd.read_csv(Config.TEST_PATH)
    if Config.DEBUG:
        df_test = df_test.iloc[: Config.DEBUG_SIZE]

    submission = pd.DataFrame(
        {
            "request_id": df_test["request_id"],
            "requester_received_pizza": final_test_probs,
        }
    )

    print("Submission DataFrame created:")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
