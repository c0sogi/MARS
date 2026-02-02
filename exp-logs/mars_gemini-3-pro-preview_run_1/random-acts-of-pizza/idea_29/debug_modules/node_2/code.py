import os
import shutil
import warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_processing as dp
import library.model_rf as m_rf
import library.model_nn as m_nn
import library.trainer as trainer


def run_demo():
    print("=== Starting Pizza Request Prediction Demo ===")

    # 1. Setup and Configuration Patching
    # We patch parameters to ensure the demo runs quickly (within seconds/minutes)
    print("Configuring environment and patching parameters...")

    # Set Seed
    utils.set_seed(42)

    # Define a temporary cache directory for this execution
    DEMO_CACHE_DIR = "./working/demo_execution/"
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Patch CACHE_DIR in all modules that use it
    config.CACHE_DIR = DEMO_CACHE_DIR
    dp.CACHE_DIR = DEMO_CACHE_DIR
    m_rf.CACHE_DIR = DEMO_CACHE_DIR
    m_nn.CACHE_DIR = DEMO_CACHE_DIR
    trainer.CACHE_DIR = DEMO_CACHE_DIR

    # Patch Random Forest Parameters for speed
    m_rf.RF_PARAMS["n_estimators"] = 5
    m_rf.RF_PARAMS["n_jobs"] = 1  # Single thread for demo stability

    # Patch Neural Network Parameters for speed
    m_nn.NN_PARAMS["epochs"] = 2
    m_nn.NN_PARAMS["batch_size"] = 8
    m_nn.NN_PARAMS["hidden_dim"] = 32
    m_nn.NN_PARAMS["patience"] = 1

    # Patch TF-IDF Parameters
    dp.TFIDF_PARAMS["max_features"] = 100

    # 2. Data Processing Demonstration
    print("\n[Step 1] Data Processing...")

    # Generate RF Dataset (Stream A)
    # We force load_cached_data=False to execute the processing logic
    print("  -> Generating RF features (TF-IDF + Metadata + Sentiment)...")
    X_train_rf, y_train_rf = dp.get_rf_dataset(split="train", load_cached_data=False)
    X_val_rf, y_val_rf = dp.get_rf_dataset(
        split="val", load_cached_data=True
    )  # Should use cache

    print(f"     RF Train Shape: {X_train_rf.shape}")

    # Assertions for RF Data
    assert X_train_rf.shape[0] > 0
    assert "title_compound" in X_train_rf.columns, "Sentiment features missing"
    assert "tfidf_0" in X_train_rf.columns, "TF-IDF features missing"
    assert not X_train_rf.isnull().any().any(), "NaNs found in RF data"

    # Generate NN Dataset (Stream B)
    print("  -> Generating NN features (Embeddings + History)...")
    ds_train_nn = dp.get_nn_dataset(split="train", load_cached_data=False)
    ds_val_nn = dp.get_nn_dataset(split="val", load_cached_data=True)

    print(f"     NN Train Samples: {len(ds_train_nn)}")

    # Assertions for NN Data
    sample_item = ds_train_nn[0]
    assert "title_emb" in sample_item
    assert "history_emb" in sample_item
    assert isinstance(sample_item["title_emb"], torch.Tensor)

    # 3. Subsampling for Model Training
    # We use a very small subset to demonstrate training mechanics instantly
    N_SUB = 32
    print(f"\n[Step 2] Subsampling data to {N_SUB} samples for rapid training demo...")

    X_train_rf_sub = X_train_rf.iloc[:N_SUB]
    y_train_rf_sub = y_train_rf.iloc[:N_SUB]
    X_val_rf_sub = X_val_rf.iloc[:N_SUB]
    y_val_rf_sub = y_val_rf.iloc[:N_SUB]

    ds_train_nn_sub = Subset(ds_train_nn, range(N_SUB))
    ds_val_nn_sub = Subset(ds_val_nn, range(N_SUB))

    # 4. Random Forest Model Demonstration
    print("\n[Step 3] Random Forest Model (Stream A)...")
    rf_model = m_rf.PizzaRandomForest()

    # Train
    print("  -> Training RF...")
    auc_rf = rf_model.train(X_train_rf_sub, y_train_rf_sub, X_val_rf_sub, y_val_rf_sub)
    print(f"     RF Validation AUC: {auc_rf:.4f}")

    # Predict
    preds_rf = rf_model.predict_proba(X_val_rf_sub)

    # Verify
    assert len(preds_rf) == N_SUB
    assert (preds_rf >= 0).all() and (preds_rf <= 1).all(), "RF probs out of range"

    # Persistence
    rf_model.save()
    assert os.path.exists(rf_model.model_path), "RF model file not saved"
    loaded = rf_model.load()
    assert loaded, "Failed to load RF model"

    # 5. Neural Network Model Demonstration
    print("\n[Step 4] Neural Network Model (Stream B)...")

    # Determine metadata dimension from dataset
    meta_dim = sample_item["metadata"].shape[0]
    nn_model = m_nn.PizzaNeuralNet(metadata_dim=meta_dim)

    # Create DataLoaders
    train_loader = DataLoader(
        ds_train_nn_sub, batch_size=m_nn.NN_PARAMS["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        ds_val_nn_sub, batch_size=m_nn.NN_PARAMS["batch_size"], shuffle=False
    )

    # Train
    print("  -> Training NN...")
    auc_nn = nn_model.train(train_loader, val_loader)
    print(f"     NN Validation AUC: {auc_nn:.4f}")

    # Predict
    preds_nn = nn_model.predict_proba(val_loader)

    # Verify
    assert len(preds_nn) == N_SUB
    assert (preds_nn >= 0).all() and (preds_nn <= 1).all(), "NN probs out of range"

    # Persistence
    nn_model.save()
    assert os.path.exists(nn_model.model_path), "NN model file not saved"
    loaded_nn = nn_model.load()
    assert loaded_nn, "Failed to load NN model"

    # 6. Ensemble Demonstration
    print("\n[Step 5] Ensemble Logic...")
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_nn = config.ENSEMBLE_WEIGHTS["nn"]

    ensemble_preds = (w_rf * preds_rf) + (w_nn * preds_nn)
    print(f"  -> Combined Predictions (First 5): {ensemble_preds[:5]}")

    assert len(ensemble_preds) == N_SUB

    # 7. Submission Utility Demonstration
    print("\n[Step 6] Submission Generation...")

    # Create mock IDs for the subsampled data
    mock_ids = [f"t3_demo_{i}" for i in range(N_SUB)]
    submission_path = os.path.join(DEMO_CACHE_DIR, "demo_submission.csv")

    utils.save_submission(mock_ids, ensemble_preds, submission_path)

    # Verify file content
    assert os.path.exists(submission_path)
    df_sub = pd.read_csv(submission_path)
    print(f"  -> Submission file saved at: {submission_path}")
    print(f"  -> Rows: {len(df_sub)}")
    print("  -> Head:")
    print(df_sub.head(3))

    assert len(df_sub) == N_SUB
    assert list(df_sub.columns) == ["request_id", "requester_received_pizza"]

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
