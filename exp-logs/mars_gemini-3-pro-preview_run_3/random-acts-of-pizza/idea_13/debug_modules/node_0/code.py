import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed, save_submission, compute_score
from library.feature_engineering import DataPreparer
from library.stacking_trainer import StackingEnsemble


def main():
    # ==========================================
    # 1. Configuration & Setup for Rapid Demo
    # ==========================================
    print("Setting up configuration for rapid execution...")

    # Enable Debug mode to use a small subset of data (e.g., 50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50

    # Reduce Stacking Folds
    Config.N_FOLDS = 2

    # Reduce Feature Dimensions for speed
    Config.TFIDF_TEXT_MAX_FEATURES = 100
    Config.TFIDF_SUBREDDIT_MAX_FEATURES = 50

    # Reduce Model Complexity (Estimators)
    Config.RF_LEXICAL_PARAMS["n_estimators"] = 5
    Config.RF_BEHAVIORAL_PARAMS["n_estimators"] = 5
    Config.XGB_SEMANTIC_PARAMS["n_estimators"] = 5

    # Ensure reproducibility
    set_seed(Config.SEED)

    # ==========================================
    # 2. Feature Engineering Pipeline
    # ==========================================
    print("\n--- Step 1: Feature Engineering ---")
    preparer = DataPreparer()

    # A. Process Training Data
    # Returns: X_lex (sparse), X_beh (sparse), X_sem (dense), y (array), ids (array)
    # We force reload=False to demonstrate computation, though caching is enabled by default in the lib
    print("Processing Training Data...")
    X_lex_train, X_beh_train, X_sem_train, y_train, train_ids = preparer.get_features(
        split="train", load_cached_data=False
    )

    # Validation of Train Shapes
    n_train = len(y_train)
    print(f"Train samples: {n_train}")

    # Check Lexical (Sparse)
    assert sparse.issparse(X_lex_train), "Lexical features should be sparse matrix"
    # Expected cols: TFIDF_TEXT (100) + Metadata (Dense features ~11)
    # Note: Metadata extractor adds cols. Let's just check > 100.
    assert X_lex_train.shape[0] == n_train
    assert X_lex_train.shape[1] >= Config.TFIDF_TEXT_MAX_FEATURES

    # Check Behavioral (Sparse)
    assert sparse.issparse(X_beh_train), "Behavioral features should be sparse matrix"
    assert X_beh_train.shape[0] == n_train

    # Check Semantic (Dense)
    assert isinstance(
        X_sem_train, np.ndarray
    ), "Semantic features should be numpy array"
    # SBERT (384) + Metadata (~11)
    assert X_sem_train.shape[1] >= 384

    # B. Process Validation Data
    print("Processing Validation Data...")
    X_lex_val, X_beh_val, X_sem_val, y_val, val_ids = preparer.get_features(
        split="val", load_cached_data=False
    )

    assert len(y_val) == X_lex_val.shape[0]

    # ==========================================
    # 3. Model Training (Stacking Ensemble)
    # ==========================================
    print("\n--- Step 2: Training Stacking Ensemble ---")
    ensemble = StackingEnsemble()

    # Fit the ensemble
    # This triggers:
    # 1. K-Fold OOF generation for Level 1 models
    # 2. Training Level 2 Meta-Learner
    # 3. Retraining Level 1 models on full training data
    ensemble.fit(X_lex_train, X_beh_train, X_sem_train, y_train)

    # Verify models are stored
    assert ensemble.lexical_model is not None, "Lexical model not retrained"
    assert ensemble.behavioral_model is not None, "Behavioral model not retrained"
    assert ensemble.semantic_model is not None, "Semantic model not retrained"
    assert ensemble.meta_learner is not None, "Meta-learner not trained"

    print("Ensemble training successful.")

    # ==========================================
    # 4. Evaluation
    # ==========================================
    print("\n--- Step 3: Evaluation on Validation Set ---")

    # Predict
    val_probs = ensemble.predict(X_lex_val, X_beh_val, X_sem_val)

    # Check Predictions
    assert len(val_probs) == len(y_val)
    assert np.all(
        (val_probs >= 0) & (val_probs <= 1)
    ), "Probabilities must be in [0, 1]"

    # Compute Metric
    auc_score = compute_score(y_val, val_probs)
    print(f"Validation AUC: {auc_score:.4f}")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n--- Step 4: Test Inference & Submission ---")

    # Get Test Features
    # Note: y_test will be None
    X_lex_test, X_beh_test, X_sem_test, y_test, test_ids = preparer.get_features(
        split="test", load_cached_data=False
    )

    assert y_test is None, "Test target should be None"

    # Predict
    test_probs = ensemble.predict(X_lex_test, X_beh_test, X_sem_test)

    # Save Submission
    save_submission(test_ids, test_probs)

    # Verify File Creation
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file verified at: {Config.SUBMISSION_PATH}")

        # Quick content check
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        assert list(df_sub.columns) == ["request_id", "requester_received_pizza"]
        assert len(df_sub) == len(test_ids)
        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
