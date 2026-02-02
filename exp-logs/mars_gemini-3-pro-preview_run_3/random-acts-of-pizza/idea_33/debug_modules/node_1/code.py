import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed, suppress_warnings, Timer
from library.data_loader import load_and_clean_data
from library.feature_engineering import create_features
from library.workflow import (
    CrossValidationEngine,
    ValidationGuidedRetrainer,
    generate_submission,
)


# 1. Setup and Configuration Override for Speed
def setup_demo_environment():
    """
    Overrides default Config settings to make the demo run fast.
    """
    print("Setting up demo environment...")

    # Set a specific working directory for this demo to avoid cache conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce CV folds
    Config.N_FOLDS = 2

    # Reduce Feature Complexity for speed
    Config.TFIDF_PARAMS["max_features"] = 500
    Config.MAX_SUBREDDIT_VOCAB = 100

    # Reduce Model Complexity (Estimators)
    Config.RF_LEXICAL_PARAMS["n_estimators"] = 10
    Config.RF_BEHAVIORAL_PARAMS["n_estimators"] = 10
    Config.RF_SEMANTIC_PARAMS["n_estimators"] = 10

    # Reduce XGBoost Complexity
    Config.XGB_SEMANTIC_PARAMS["n_estimators"] = 10
    Config.XGB_EARLY_STOPPING_ROUNDS = 2

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    suppress_warnings()
    set_seed(Config.SEED)
    print("Configuration optimized for speed.")


def verify_data_loading():
    """
    Demonstrates data loading and cleaning.
    """
    print("\n--- Step 1: Data Loading ---")
    # Force reload to demonstrate processing logic
    train_df, val_df, test_df = load_and_clean_data(load_cached_data=False)

    # Verification
    assert not train_df.empty, "Train DataFrame is empty"
    assert not val_df.empty, "Val DataFrame is empty"
    assert not test_df.empty, "Test DataFrame is empty"

    # Check leakage columns are removed
    leakage_cols = [c for c in train_df.columns if c.endswith("_at_retrieval")]
    assert len(leakage_cols) == 0, f"Leakage columns found: {leakage_cols}"

    print(
        f"Data Loaded: Train({train_df.shape}), Val({val_df.shape}), Test({test_df.shape})"
    )
    return train_df, val_df, test_df


def verify_feature_engineering(train_df, val_df, test_df):
    """
    Demonstrates feature generation pipeline.
    """
    print("\n--- Step 2: Feature Engineering ---")

    # Generate features
    # This uses the reduced vocab size from our config override
    X_train, X_val, X_test = create_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verification of dictionary structure
    expected_keys = {"lexical", "behavioral", "semantic", "metadata"}
    assert set(X_train.keys()) == expected_keys

    # Verification of shapes
    n_train = len(train_df)
    assert X_train["lexical"].shape[0] == n_train
    assert X_train["metadata"].shape[0] == n_train

    # Verification of types
    assert sp.issparse(X_train["lexical"]), "Lexical features should be sparse"
    assert isinstance(
        X_train["semantic"], np.ndarray
    ), "Semantic features should be dense numpy array"

    print("Features generated and shapes verified.")
    return X_train, X_val, X_test


def verify_workflow(X_train, train_df, X_val, val_df, X_test, test_df):
    """
    Demonstrates the full training and prediction workflow.
    """
    print("\n--- Step 3: Cross-Validation & Modeling ---")

    # Prepare targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # 1. Cross-Validation (OOF Generation)
    cv_engine = CrossValidationEngine(n_folds=Config.N_FOLDS)
    oof_preds, y_train_aligned = cv_engine.run_cv(X_train, y_train)

    # Verify OOF
    assert oof_preds.shape == (
        len(train_df),
        5,
    ), "OOF predictions shape mismatch (N_samples, 5 models)"
    assert np.allclose(y_train, y_train_aligned), "Target alignment failed"

    # 2. Final Training
    print("\n--- Step 4: Final Model Retraining ---")
    retrainer = ValidationGuidedRetrainer()

    # We pass the OOF predictions to train the meta-learner
    trained_models = retrainer.train_final_models(
        X_train, y_train, X_val, y_val, oof_X=oof_preds, oof_y=y_train
    )

    expected_models = {
        "LexicalBagger",
        "CommunityBagger",
        "SemanticBooster",
        "SemanticBagger",
        "MetadataAnchor",
        "meta",
    }
    assert set(trained_models.keys()) == expected_models, "Missing trained models"
    print("All models successfully retrained.")

    # 3. Submission Generation
    print("\n--- Step 5: Submission Generation ---")
    generate_submission(trained_models, X_test, test_df)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (len(test_df), 2), "Submission shape mismatch"
    assert Config.ID_COL in sub_df.columns
    assert Config.TARGET_COL in sub_df.columns

    # Check probability range
    probs = sub_df[Config.TARGET_COL]
    assert probs.min() >= 0 and probs.max() <= 1, "Probabilities out of range [0, 1]"

    print("Submission generated and verified.")


if __name__ == "__main__":
    with Timer("Full Demo Execution"):
        # 1. Configure
        setup_demo_environment()

        # 2. Load Data
        train_df, val_df, test_df = verify_data_loading()

        # 3. Create Features
        X_train, X_val, X_test = verify_feature_engineering(train_df, val_df, test_df)

        # 4. Run Workflow
        verify_workflow(X_train, train_df, X_val, val_df, X_test, test_df)

    print("\nDemo completed successfully!")
