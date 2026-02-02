import os
import shutil
import pandas as pd
import numpy as np
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
import library.config as config
import library.preprocessing as pp
import library.features as feats
import library.model_definitions as md
import library.ensemble as ens


def main():
    print("============================================================")
    print("   Random Acts of Pizza - Library Usage Demonstration")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration & Patching
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch 'library.preprocessing' constants
    pp.WORKING_DIR = DEMO_DIR

    # Patch 'library.features' constants
    feats.WORKING_DIR = DEMO_DIR
    # Reduce min_df to ensure vocabulary is generated on small sample
    feats.TFIDF_PARAMS["min_df"] = 1

    # Patch 'library.model_definitions' constants for speed
    # Reduce estimators to minimal values
    md.RF_PARAMS["n_estimators"] = 5
    md.XGB_PARAMS["n_estimators"] = 5

    # Patch 'library.ensemble' constants
    ens.N_FOLDS = 2  # Use 2-fold CV instead of 5
    ens.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # We will patch ens.TEST_PATH later after creating the sample file

    print(f"    Working Directory: {DEMO_DIR}")
    print("    Patched hyperparameters: n_estimators=5, n_folds=2, min_df=1")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[2] Loading and Preprocessing Data...")

    # Load full data (skipping cache to force execution of loading logic)
    # We use the provided function which handles text cleaning and list serialization
    train_full, val_full, test_full = pp.load_dataset(load_cached_data=False)

    # Sample data for the demonstration (50 rows each)
    SAMPLE_SIZE = 50
    print(f"    Sampling top {SAMPLE_SIZE} rows from Train, Val, and Test...")

    train_sample = train_full.head(SAMPLE_SIZE).copy()
    val_sample = val_full.head(SAMPLE_SIZE).copy()
    test_sample = test_full.head(SAMPLE_SIZE).copy()

    # Create a temporary test parquet file for the ensemble.predict method to read
    # (The ensemble reads IDs directly from disk to ensure alignment)
    temp_test_path = os.path.join(DEMO_DIR, "test_sample.parquet")
    test_sample.to_parquet(temp_test_path, index=False)

    # Patch the TEST_PATH in ensemble to point to our sampled file
    ens.TEST_PATH = temp_test_path

    # Validation
    assert len(train_sample) == SAMPLE_SIZE
    assert "request_text_edit_aware" in train_sample.columns
    print("    Data loaded and sampled successfully.")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[3] Generating Features...")

    # Instantiate FeatureFactory
    # It will use the patched WORKING_DIR and TFIDF_PARAMS
    ff = feats.FeatureFactory()

    # Process data
    # load_cached_data=False forces re-computation
    print("    Computing features (TF-IDF, Embeddings, Metadata)...")
    feature_data = ff.process_data(
        train_sample, val_sample, test_sample, load_cached_data=False
    )

    # Validation of Feature Dictionary
    print("    Validating feature shapes...")

    # Check Sparse Matrices
    assert feature_data["X_train_lexical"].shape[0] == SAMPLE_SIZE
    assert feature_data["X_test_behavioral"].shape[0] == SAMPLE_SIZE

    # Check Dense Embeddings (MPNet produces 768 dim)
    assert feature_data["X_train_text_emb"].shape == (SAMPLE_SIZE, 768)

    # Check Targets
    assert feature_data["y_train"].shape == (SAMPLE_SIZE,)

    print("    Feature generation complete and validated.")

    # -------------------------------------------------------------------------
    # 4. Model Definitions Check
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Definitions...")

    # Check if our patches took effect
    rf_model = md.get_lexical_bagger()
    xgb_model = md.get_semantic_booster()

    assert rf_model.n_estimators == 5, "RF n_estimators should be 5"
    assert xgb_model.n_estimators == 5, "XGB n_estimators should be 5"
    print("    Model factories produce correctly configured instances.")

    # -------------------------------------------------------------------------
    # 5. Ensemble Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Ensemble Pipeline...")

    # Instantiate StackingEnsemble
    # It uses the patched N_FOLDS and model definitions
    ensemble = ens.StackingEnsemble()

    # A. Generate Out-of-Fold Predictions
    print("    Step A: Generating OOF Predictions...")
    oof_preds = ensemble.generate_oof_predictions(feature_data)

    assert oof_preds.shape == (SAMPLE_SIZE, len(ensemble.base_models))
    assert not oof_preds.isnull().values.any()
    print("            OOF predictions generated.")

    # B. Train Meta-Learner
    print("    Step B: Training Meta-Learner...")
    ensemble.train_meta_learner(oof_preds, feature_data["y_train"])
    # Check if model is fitted (has coef_)
    assert hasattr(ensemble.meta_learner, "coef_")
    print("            Meta-learner trained.")

    # C. Retrain Base Models
    print("    Step C: Retraining Base Models on Full (Sampled) Data...")
    ensemble.retrain_base_models(feature_data)
    print("            Base models retrained.")

    # D. Final Prediction
    print("    Step D: Predicting on Test Set...")
    ensemble.predict(feature_data)

    # -------------------------------------------------------------------------
    # 6. Final Validation
    # -------------------------------------------------------------------------
    print("\n[6] Validating Submission...")

    if not os.path.exists(ens.SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not created at {ens.SUBMISSION_PATH}")

    submission_df = pd.read_csv(ens.SUBMISSION_PATH)

    print(f"    Submission File: {ens.SUBMISSION_PATH}")
    print(f"    Rows: {len(submission_df)}")
    print(f"    Columns: {list(submission_df.columns)}")

    # Check format
    assert len(submission_df) == SAMPLE_SIZE
    assert "request_id" in submission_df.columns
    assert "requester_received_pizza" in submission_df.columns

    # Check values are probabilities
    probs = submission_df["requester_received_pizza"]
    assert probs.min() >= 0 and probs.max() <= 1

    print("\n============================================================")
    print("   Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    main()
