import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

# Import from the provided library
from library.config import Config
from library.data_loader import load_dataset, get_common_columns
from library.feature_engineering import FeatureEngineer
from library.neural_net import NeuralNetTrainer
from library.training import run_training_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def demo_pipeline():
    print("=== Starting Library Usage Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Demo
    # ---------------------------------------------------------
    print("1. Configuring environment for fast execution...")

    # Set up a temporary working directory for this demo
    demo_working_dir = "./working/demo_execution/"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config attributes to speed up processing
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 30  # Use only 30 samples for demonstration
    Config.RF_N_ESTIMATORS = 5  # Few trees for RF
    Config.MLP_EPOCHS = 2  # Few epochs for MLP
    Config.MLP_BATCH_SIZE = 4  # Small batch size
    Config.TFIDF_MAX_FEATURES = 100  # Reduce TF-IDF features

    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print("   Configuration updated successfully.\n")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("2. Demonstrating Data Loading...")

    # Load data (this will trigger the debug sampling)
    train_df, val_df, test_df = load_dataset(load_cached_data=False)

    # Verification
    print(f"   Train shape: {train_df.shape}")
    print(f"   Val shape:   {val_df.shape}")
    print(f"   Test shape:  {test_df.shape}")

    assert (
        len(train_df) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train set size exceeds debug limit"
    assert len(val_df) <= Config.DEBUG_SAMPLE_SIZE, "Val set size exceeds debug limit"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train"

    # Demonstrate get_common_columns utility
    common_cols = get_common_columns(train_df, test_df)
    print(f"   Identified {len(common_cols)} common feature columns.")
    assert (
        "request_text_edit_aware" not in common_cols
    ), "Leaky column 'request_text_edit_aware' found in common columns"
    print("   Data Loading verification passed.\n")

    # ---------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # ---------------------------------------------------------
    print("3. Demonstrating Feature Engineering...")

    fe = FeatureEngineer()

    # A. Random Forest Inputs
    print("   A. Preparing Random Forest Inputs (Metadata + TF-IDF + Consistency)...")
    (X_train_rf, y_train_rf), (X_val_rf, y_val_rf), X_test_rf = fe.prepare_rf_inputs(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify RF inputs
    assert sparse.issparse(X_train_rf), "RF Input X_train should be a sparse matrix"
    assert X_train_rf.shape[0] == len(train_df), "RF Input rows mismatch with train_df"
    assert len(y_train_rf) == len(train_df), "RF Target rows mismatch"
    print(f"      RF Train Input Shape: {X_train_rf.shape}")
    print("      RF Input verification passed.")

    # B. MLP Inputs
    print("   B. Preparing MLP Inputs (Embeddings + Scaled Metadata)...")
    train_mlp, val_mlp, test_mlp = fe.prepare_mlp_inputs(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify MLP inputs
    required_keys = ["metadata", "request_emb", "history_emb"]
    for key in required_keys:
        assert key in train_mlp, f"Missing key '{key}' in MLP train data"

    # Check Embedding Dimensions (SBERT default is 384)
    assert train_mlp["request_emb"].shape[1] == 384, "Request embedding dim mismatch"
    assert train_mlp["history_emb"].shape[2] == 384, "History embedding dim mismatch"
    assert train_mlp["metadata"].shape[0] == len(train_df), "Metadata rows mismatch"

    print(f"      MLP Request Emb Shape: {train_mlp['request_emb'].shape}")
    print(f"      MLP History Emb Shape: {train_mlp['history_emb'].shape}")
    print("      MLP Input verification passed.\n")

    # ---------------------------------------------------------
    # 4. Model Training Demonstration (Random Forest)
    # ---------------------------------------------------------
    print("4. Demonstrating Random Forest Training...")

    rf_model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS, random_state=Config.RANDOM_SEED
    )
    rf_model.fit(X_train_rf, y_train_rf)

    # Predict
    rf_preds = rf_model.predict_proba(X_val_rf)[:, 1]

    assert len(rf_preds) == len(val_df), "RF prediction length mismatch"
    assert np.all((rf_preds >= 0) & (rf_preds <= 1)), "RF probabilities out of bounds"
    print(f"   RF validation predictions generated. Mean prob: {rf_preds.mean():.4f}")
    print("   RF Training verification passed.\n")

    # ---------------------------------------------------------
    # 5. Model Training Demonstration (Neural Net)
    # ---------------------------------------------------------
    print("5. Demonstrating Neural Net Training...")

    # Initialize Trainer
    meta_dim = train_mlp["metadata"].shape[1]
    trainer = NeuralNetTrainer(input_dims={"metadata": meta_dim})

    # Train
    best_auc = trainer.train(train_mlp, val_mlp)
    print(f"   Training complete. Best Val AUC: {best_auc:.4f}")

    # Predict
    mlp_preds = trainer.predict(test_mlp)

    assert len(mlp_preds) == len(test_df), "MLP prediction length mismatch"
    assert mlp_preds.ndim == 1, "MLP predictions should be 1D array"
    print(f"   MLP test predictions generated. Shape: {mlp_preds.shape}")
    print("   Neural Net verification passed.\n")

    # ---------------------------------------------------------
    # 6. Full Pipeline Integration Check
    # ---------------------------------------------------------
    print("6. Verifying Full Pipeline Wrapper...")

    # We call the main pipeline function to ensure it ties everything together correctly.
    # Note: This will re-run the steps above but encapsulated.
    metrics = run_training_pipeline(load_cached_data=True)  # Use cache generated above

    assert "rf_val_auc" in metrics, "Pipeline missing RF metric"
    assert "mlp_val_auc" in metrics, "Pipeline missing MLP metric"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_df) == len(test_df), "Submission row count mismatch"
    assert "request_id" in submission_df.columns, "Submission missing request_id"
    assert (
        "requester_received_pizza" in submission_df.columns
    ), "Submission missing target"

    print(f"   Pipeline execution successful. Metrics: {metrics}")
    print("   Submission file verified.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
