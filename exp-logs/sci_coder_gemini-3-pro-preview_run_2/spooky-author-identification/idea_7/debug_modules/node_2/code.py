import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import logging

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_artifact
from library.features import FeatureEngineer
from library.trainers import ModelTrainer
from library.meta_learner import StackingEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Author Identification Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # We override Config parameters to ensure the demo runs quickly (within seconds/minutes)
    # and uses a separate working directory.
    print("[1] Configuring environment for fast demonstration...")

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config paths
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Cache Paths to avoid conflicts with real experiments
    Config.CACHE_TRAIN_FEATURES = os.path.join(DEMO_DIR, "train_features.npy")
    Config.CACHE_TEST_FEATURES = os.path.join(DEMO_DIR, "test_features.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.CACHE_LABEL_ENCODER = os.path.join(DEMO_DIR, "label_encoder.npy")

    # Reduce Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.ACCUMULATION_STEPS = 1
    Config.SVD_N_COMPONENTS = 10  # Reduced from 100
    Config.MAX_LENGTH = 32  # Reduced from 85
    Config.N_FOLDS = 2  # Not strictly used in this linear demo, but good practice

    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[2] Running Feature Engineering...")
    fe = FeatureEngineer()

    # process_data checks for cache, but since we changed paths, it will compute from scratch
    data = fe.process_data(load_cached_data=True)

    # Verify Data Dictionary
    required_keys = [
        "train_df",
        "test_df",
        "X_train_sparse",
        "X_test_sparse",
        "X_train_dense",
        "X_test_dense",
        "y_train",
        "label_encoder",
    ]
    for key in required_keys:
        assert key in data, f"Missing key in data dictionary: {key}"

    print(f"    Train shape (Dense): {data['X_train_dense'].shape}")
    print(f"    Train shape (Sparse): {data['X_train_sparse'].shape}")
    print(f"    Labels shape: {data['y_train'].shape}")

    # -------------------------------------------------------------------------
    # 3. Create Subsets for Fast Training
    # -------------------------------------------------------------------------
    print("\n[3] Subsetting data for rapid model verification...")
    # We will use a tiny subset: 50 training samples, 20 validation samples
    subset_train_size = 50
    subset_val_size = 20
    total_subset = subset_train_size + subset_val_size

    # Ensure we don't exceed available data
    total_available = len(data["y_train"])
    if total_subset > total_available:
        total_subset = total_available
        subset_train_size = int(total_subset * 0.8)
        subset_val_size = total_subset - subset_train_size

    # Indices
    indices = np.random.permutation(total_available)
    train_idx = indices[:subset_train_size]
    val_idx = indices[subset_train_size:total_subset]

    # Subset Data
    # 1. Text (for Neural Net)
    train_texts = data["train_df"].iloc[train_idx]["text"].values
    val_texts = data["train_df"].iloc[val_idx]["text"].values

    # 2. Labels
    y_train_sub = data["y_train"][train_idx]
    y_val_sub = data["y_train"][val_idx]

    # 3. Sparse Features (for LR, NB)
    X_train_sparse_sub = data["X_train_sparse"][train_idx]
    X_val_sparse_sub = data["X_train_sparse"][val_idx]

    # 4. Dense Features (for XGB)
    X_train_dense_sub = data["X_train_dense"][train_idx]
    X_val_dense_sub = data["X_train_dense"][val_idx]

    print(f"    Subset Train Size: {len(train_idx)}")
    print(f"    Subset Val Size: {len(val_idx)}")

    # -------------------------------------------------------------------------
    # 4. Model Training (Neural & Classical)
    # -------------------------------------------------------------------------
    trainer = ModelTrainer()
    oof_preds = {}  # Store validation predictions

    # A. Neural Network
    print("\n[4A] Training Neural Network (DeBERTa)...")
    nn_preds, nn_state = trainer.train_neural_fold(
        train_texts, y_train_sub, val_texts, y_val_sub, fold_idx=0
    )
    assert nn_preds.shape == (subset_val_size, 3), "NN output shape mismatch"
    oof_preds["nn"] = nn_preds

    # B. Classical Models
    print("\n[4B] Training Classical Models...")

    # Logistic Regression
    lr_preds, _ = trainer.train_classical_fold(
        X_train_sparse_sub,
        y_train_sub,
        X_val_sparse_sub,
        y_val_sub,
        model_type="lr",
        fold_idx=0,
    )
    oof_preds["lr"] = lr_preds

    # Naive Bayes
    nb_preds, _ = trainer.train_classical_fold(
        X_train_sparse_sub,
        y_train_sub,
        X_val_sparse_sub,
        y_val_sub,
        model_type="nb",
        fold_idx=0,
    )
    oof_preds["nb"] = nb_preds

    # XGBoost
    xgb_preds, _ = trainer.train_classical_fold(
        X_train_dense_sub,
        y_train_sub,
        X_val_dense_sub,
        y_val_sub,
        model_type="xgb",
        fold_idx=0,
    )
    oof_preds["xgb"] = xgb_preds

    # -------------------------------------------------------------------------
    # 5. Meta-Learner (Stacking)
    # -------------------------------------------------------------------------
    print("\n[5] Training Meta-Learner (Stacking Ensemble)...")
    stacker = StackingEnsemble()

    # Fit on the validation predictions (acting as OOF predictions here)
    stacker.fit(oof_preds, y_val_sub)

    # -------------------------------------------------------------------------
    # 6. Prediction & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Test Predictions...")

    # Create dummy test predictions for the meta-learner
    # In a real scenario, we would predict on the full test set with each base model.
    # Here we simulate it using random probabilities for the demo speed.
    test_size = 10
    test_ids = [f"id{i}" for i in range(test_size)]

    test_preds_dict = {
        "nn": np.random.rand(test_size, 3),
        "lr": np.random.rand(test_size, 3),
        "nb": np.random.rand(test_size, 3),
        "xgb": np.random.rand(test_size, 3),
    }

    # Normalize dummy preds to sum to 1 (just for consistency)
    for k in test_preds_dict:
        test_preds_dict[k] /= test_preds_dict[k].sum(axis=1, keepdims=True)

    # Predict using Meta-Learner
    final_preds = stacker.predict(test_preds_dict)

    assert final_preds.shape == (test_size, 3), "Final prediction shape mismatch"

    # Create Submission
    stacker.create_submission(test_ids, final_preds, output_path=Config.SUBMISSION_PATH)

    # Verify file existence
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    SUCCESS: Submission file created at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print("    Submission Head:")
        print(df_sub.head())
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
