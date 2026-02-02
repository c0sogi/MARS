import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import set_seed, save_model, load_model
from library.data_loader import load_dataset
from library.features import FeaturePipeline, get_features
from library.models import SparseBagger, DenseBooster, StackingMetaLearner
from library.train import train_ensemble
from library.predict import generate_predictions


def run_demo():
    print("=== Starting Demonstration of Pizza Request Prediction Pipeline ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")
    DEMO_DIR = "./working/demo_pipeline"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch Config to use demo directory and lightweight parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce feature dimensionality
    Config.LEXICAL_MAX_FEATURES = 50
    Config.BEHAVIORAL_MAX_FEATURES = 20

    # Reduce Model Complexity
    Config.LEXICAL_RF_PARAMS["n_estimators"] = 5
    Config.BEHAVIORAL_RF_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_XGB_PARAMS["n_estimators"] = 5
    Config.SEMANTIC_XGB_PARAMS["early_stopping_rounds"] = 2

    # Reduce CV Folds
    Config.N_FOLDS = 2

    print(f"   Working Directory set to: {Config.WORKING_DIR}")
    print("   Model parameters reduced for speed.")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Subsampling
    # -------------------------------------------------------------------------
    print("\n2. Loading and Subsampling Data...")
    # Load raw data (ignoring cache initially to get full data)
    train_full, val_full, test_full = load_dataset(load_cached_data=False)

    # Create small subsets
    N_SAMPLES = 40
    train_small = train_full.head(N_SAMPLES).copy()
    val_small = val_full.head(N_SAMPLES).copy()
    test_small = test_full.head(N_SAMPLES).copy()

    print(f"   Subsampled Train shape: {train_small.shape}")
    print(f"   Subsampled Val shape: {val_small.shape}")
    print(f"   Subsampled Test shape: {test_small.shape}")

    # Save these small subsets as "processed" cache.
    # This tricks the pipeline into using them when load_cached_data=True
    train_small.to_parquet(
        os.path.join(DEMO_DIR, "train_processed.parquet"), index=False
    )
    val_small.to_parquet(os.path.join(DEMO_DIR, "val_processed.parquet"), index=False)
    test_small.to_parquet(os.path.join(DEMO_DIR, "test_processed.parquet"), index=False)
    print("   Cached subsampled datasets.")

    # -------------------------------------------------------------------------
    # 3. Feature Generation (on Subsets)
    # -------------------------------------------------------------------------
    print("\n3. Generating Features on Subsets...")
    # This will generate features for the small datasets and save them to DEMO_DIR
    data = get_features(train_small, val_small, test_small, load_cached_data=False)

    # Verify Feature Shapes
    assert data["X_train_lexical"].shape[0] == N_SAMPLES
    assert data["X_train_semantic"].shape[0] == N_SAMPLES
    assert data["X_test_lexical"].shape[0] == N_SAMPLES

    print("   Feature generation complete. Shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Unit Testing Model Classes
    # -------------------------------------------------------------------------
    print("\n4. Unit Testing Model Classes with Synthetic Data...")

    # Synthetic Data
    n_synth = 100
    n_feat_sparse = 50
    n_feat_dense = 10
    X_sparse = sp.random(n_synth, n_feat_sparse, density=0.1, format="csr")
    X_dense = np.random.rand(n_synth, n_feat_dense)
    y_synth = np.random.randint(0, 2, n_synth)

    # A. SparseBagger (Random Forest)
    print("   Testing SparseBagger...")
    sb = SparseBagger(params={"n_estimators": 5, "n_jobs": 1})
    sb.fit(X_sparse, y_synth)
    probs_sb = sb.predict_proba(X_sparse)
    assert probs_sb.shape == (n_synth,)
    assert 0.0 <= probs_sb.min() and probs_sb.max() <= 1.0
    print("   SparseBagger passed.")

    # B. DenseBooster (XGBoost)
    print("   Testing DenseBooster...")
    db = DenseBooster(params={"n_estimators": 5, "max_depth": 2, "n_jobs": 1})
    # Test with validation set
    db.fit(X_dense, y_synth, X_val=X_dense, y_val=y_synth)
    probs_db = db.predict_proba(X_dense)
    assert probs_db.shape == (n_synth,)
    print("   DenseBooster passed.")

    # C. StackingMetaLearner (Logistic Regression)
    print("   Testing StackingMetaLearner...")
    # Input to meta learner is predictions from base models (e.g., 3 models)
    X_meta = np.random.rand(n_synth, 3)
    ml = StackingMetaLearner(params={"C": 1.0})
    ml.fit(X_meta, y_synth)
    probs_ml = ml.predict_proba(X_meta)
    assert probs_ml.shape == (n_synth,)
    assert len(ml.model.coef_[0]) == 3
    print("   StackingMetaLearner passed.")

    # -------------------------------------------------------------------------
    # 5. Integration Test: Training Pipeline
    # -------------------------------------------------------------------------
    print("\n5. Running Full Training Pipeline (train_ensemble)...")
    # This will use the cached subsampled data and features we created earlier
    train_ensemble(load_cached_data=True)

    # Verify models were saved
    expected_models = [
        "lexical_rf.joblib",
        "behavioral_rf.joblib",
        "semantic_xgb.joblib",
        "meta_learner.joblib",
    ]
    for m in expected_models:
        path = os.path.join(DEMO_DIR, m)
        assert os.path.exists(path), f"Model file {m} was not created."
    print("   Training pipeline completed successfully. Artifacts verified.")

    # -------------------------------------------------------------------------
    # 6. Integration Test: Prediction Pipeline
    # -------------------------------------------------------------------------
    print("\n6. Running Prediction Pipeline (generate_predictions)...")
    generate_predictions(load_cached_data=True)

    # Verify Submission
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), "Submission file not found."

    df_sub = pd.read_csv(sub_path)
    print(f"   Submission loaded. Shape: {df_sub.shape}")

    # Checks
    assert df_sub.shape == (
        N_SAMPLES,
        2,
    ), f"Expected ({N_SAMPLES}, 2), got {df_sub.shape}"
    assert "request_id" in df_sub.columns
    assert "requester_received_pizza" in df_sub.columns
    assert df_sub["requester_received_pizza"].between(0, 1).all()

    print("   Prediction pipeline completed successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
