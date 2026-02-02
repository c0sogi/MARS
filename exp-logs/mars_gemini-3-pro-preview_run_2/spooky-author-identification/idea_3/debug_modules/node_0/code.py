import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.utils import set_seed, get_device, compute_log_loss
from library.data_loader import load_and_process_data
from library.feature_extractor import ClassicalFeaturePipeline, NeuralFeaturePipeline
from library.model_zoo import ClassicalModelWrapper, TransformerClassifier
from library.training_engine import train_classical_model, train_neural_model
from library.stacking_manager import StackingEnsemble


def main():
    print("=== Starting Library Demonstration ===")

    # 1. Configuration optimized for speed and demonstration
    # We use 'debug=True' to subsample data and a tiny transformer model.
    config = {
        "seed": 42,
        "debug": True,  # Subsample data (1000 train, 500 test)
        "n_folds": 2,  # Minimal folds for CV
        "svd_n_components": 10,  # Low dimension for SVD
        "transformer_model": "prajjwal1/bert-tiny",  # Fast, small model
        "max_length": 32,  # Short sequence length
        "batch_size": 8,
        "epochs": 1,  # Single epoch training
        "learning_rate": 1e-4,
        "patience": 1,
        "lr_C": 0.1,
        "nb_alpha": 1.0,
        "xgb_n_estimators": 5,  # Very few trees
        "xgb_max_depth": 2,
    }

    # Clean working directory to ensure we test generation logic
    working_dir = "./working/idea_3"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # --- Step 1: Data Loading ---
    print("\n[1] Testing Data Loader...")
    # Force load from scratch by setting load_cached_data=False
    train_df, test_df, label_classes = load_and_process_data(
        config, load_cached_data=False
    )

    # Validation
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert "text" in train_df.columns
    assert "author_encoded" in train_df.columns
    assert "fold" in train_df.columns
    assert len(train_df) <= 1000, "Debug mode should limit training size"
    assert len(label_classes) == 3, "Should have 3 author classes"
    print(
        f"    Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )

    # --- Step 2: Feature Extraction ---
    print("\n[2] Testing Feature Extraction...")

    # 2a. Classical Features
    print("    Running Classical Pipeline...")
    classical_pipe = ClassicalFeaturePipeline(config)
    train_sparse, train_dense, test_sparse, test_dense = classical_pipe.execute(
        train_df["text"], test_df["text"], load_cached_data=False
    )

    assert train_sparse.shape[0] == len(train_df)
    assert train_dense.shape[0] == len(train_df)
    assert train_dense.shape[1] == config["svd_n_components"]

    # 2b. Neural Features
    print("    Running Neural Pipeline...")
    neural_pipe = NeuralFeaturePipeline(config)
    train_neural, test_neural = neural_pipe.execute(
        train_df["text"], test_df["text"], load_cached_data=False
    )

    assert "input_ids" in train_neural
    assert train_neural["input_ids"].shape == (len(train_df), config["max_length"])
    print("    Feature extraction successful.")

    # --- Step 3: Model Training (Manual Split) ---
    print("\n[3] Testing Model Training Components...")

    # Create a manual split for demonstration
    split_idx = int(len(train_df) * 0.8)

    # 3a. Classical Model (Logistic Regression)
    print("    Training Classical Model (LR)...")
    wrapper = ClassicalModelWrapper("lr", config)

    X_tr_cls = train_sparse[:split_idx]
    y_tr = train_df["author_encoded"].values[:split_idx]
    X_val_cls = train_sparse[split_idx:]
    y_val = train_df["author_encoded"].values[split_idx:]

    model_cls, val_probs_cls, val_loss_cls = train_classical_model(
        wrapper, X_tr_cls, y_tr, X_val_cls, y_val
    )

    assert val_probs_cls.shape == (len(y_val), 3)
    assert val_loss_cls < 2.0, "Loss should be reasonable"
    print(f"    LR Validation Loss: {val_loss_cls:.4f}")

    # 3b. Neural Model (Transformer)
    print("    Training Neural Model (Transformer)...")
    device = get_device()
    print(f"    Device: {device}")

    neural_model = TransformerClassifier(config["transformer_model"], num_classes=3)

    X_tr_neu = {k: v[:split_idx] for k, v in train_neural.items()}
    X_val_neu = {k: v[split_idx:] for k, v in train_neural.items()}

    trained_model_neu, val_probs_neu, val_loss_neu = train_neural_model(
        neural_model, X_tr_neu, y_tr, X_val_neu, y_val, config
    )

    assert val_probs_neu.shape == (len(y_val), 3)
    print(f"    Neural Validation Loss: {val_loss_neu:.4f}")

    # --- Step 4: Full Stacking Pipeline ---
    print("\n[4] Testing Stacking Ensemble Manager...")
    # The ensemble manager will orchestrate CV for all models (LR, NB, XGB, Transformer)
    # and train the meta-learner. It will reuse the cached features generated above
    # because the config hash will match.

    ensemble = StackingEnsemble(config)
    ensemble.run()

    # Verify Submission
    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(submission_path)

    # Validate Submission Format
    assert "id" in sub_df.columns
    assert sub_df.shape[0] == len(test_df)
    assert sub_df.shape[1] == 4  # id + 3 classes

    # Check probabilities sum (approx 1 due to rounding, but our metric rescales anyway)
    # Just check they are floats and within range
    cols = [c for c in sub_df.columns if c != "id"]
    probs = sub_df[cols].values
    assert (probs >= 0).all() and (probs <= 1).all()

    print(f"    Submission generated at {submission_path}")
    print(f"    Submission shape: {sub_df.shape}")

    print("\n=== Demonstration Complete: All tests passed successfully ===")


if __name__ == "__main__":
    main()
