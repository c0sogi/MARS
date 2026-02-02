import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp

# Import from the provided library
from library.config import Config
from library.utils import set_seed, log
from library.data_loader import get_data_splits
from library.features import FeatureEngineer
from library.model_definitions import get_base_models
from library.ensemble import StackingEnsemble


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    log("Step 1: Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples per split
    Config.N_FOLDS = 2  # Reduce folds for OOF
    Config.TFIDF_MAX_FEATURES = 500  # Reduce vocabulary size

    # Reduce model complexity for demo
    Config.LEXICAL_RF_PARAMS["n_estimators"] = 10
    Config.BEHAVIORAL_RF_PARAMS["n_estimators"] = 10
    Config.SEMANTIC_RF_PARAMS["n_estimators"] = 10
    Config.SEMANTIC_XGB_PARAMS["n_estimators"] = 10

    # Ensure directories exist (Config does this, but good to double check logic)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    log("Step 2: Loading Data...")

    # We force load_cached_data=False to demonstrate the full pipeline logic
    # (preprocessing -> cleaning) on the debug subset.
    train_df, y_train, val_df, y_val, test_df, test_ids = get_data_splits(
        load_cached_data=False, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Validation
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert len(y_train) == Config.DEBUG_SAMPLE_SIZE, "Target size mismatch"
    assert (
        "request_text" in train_df.columns
    ), "Preprocessing failed to standardize text column"
    assert (
        "requester_received_pizza" not in train_df.columns
    ), "Target leakage in train features"

    log(f"Data Loaded Successfully. Train shape: {train_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    log("Step 3: Generating Features (Lexical, Behavioral, Semantic, Contextual)...")

    fe = FeatureEngineer()

    # This generates all views and caches them
    # Note: This might take a moment due to embedding generation, but debug_size=100 makes it fast.
    feature_views = fe.generate_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validate Structure
    for split in ["train", "val", "test"]:
        assert split in feature_views, f"Missing split {split} in feature views"
        for view in ["lexical", "behavioral", "semantic", "contextual"]:
            assert view in feature_views[split], f"Missing view {view} in {split}"

    # Validate Dimensions
    # Lexical should be sparse
    assert sp.issparse(
        feature_views["train"]["lexical"]
    ), "Lexical view should be sparse"
    # Semantic should be dense
    assert isinstance(
        feature_views["train"]["semantic"], np.ndarray
    ), "Semantic view should be dense numpy array"
    # Row counts must match
    assert feature_views["train"]["lexical"].shape[0] == Config.DEBUG_SAMPLE_SIZE

    log("Feature Engineering Complete and Validated.")

    # -------------------------------------------------------------------------
    # 4. Model Definition Check
    # -------------------------------------------------------------------------
    log("Step 4: Verifying Model Definitions...")

    models = get_base_models()
    expected_models = [
        "LexicalBagger",
        "CommunityBagger",
        "SemanticBooster",
        "SemanticBagger",
        "MetadataAnchor",
    ]

    for m in expected_models:
        assert m in models, f"Missing base model definition: {m}"

    log(f"Models defined: {list(models.keys())}")

    # -------------------------------------------------------------------------
    # 5. Ensemble Training (OOF & Meta-Learner)
    # -------------------------------------------------------------------------
    log("Step 5: Training Stacking Ensemble...")

    ensemble = StackingEnsemble()

    # Fit OOF (Level 1) and train Meta-Learner (Level 2)
    # This uses the 'train' split of the data
    oof_preds = ensemble.fit_oof(feature_views["train"], y_train)

    # Validate OOF
    assert oof_preds.shape == (
        Config.DEBUG_SAMPLE_SIZE,
        len(expected_models),
    ), "OOF predictions shape mismatch"
    assert not oof_preds.isnull().values.any(), "OOF predictions contain NaNs"

    log("OOF Generation Complete.")

    # -------------------------------------------------------------------------
    # 6. Final Retraining & Prediction
    # -------------------------------------------------------------------------
    log("Step 6: Retraining on full data and generating submission...")

    # This retrains base models on Train + Val (or uses Val for early stopping)
    # and predicts on Test
    submission_df = ensemble.retrain_and_predict(
        train_features=feature_views["train"],
        y_train=y_train,
        val_features=feature_views["val"],
        y_val=y_val,
        test_features=feature_views["test"],
        test_ids=test_ids,
    )

    # Validate Submission
    assert isinstance(submission_df, pd.DataFrame), "Submission is not a DataFrame"
    assert list(submission_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns mismatch"
    assert (
        len(submission_df) == Config.DEBUG_SAMPLE_SIZE
    ), "Submission row count mismatch"

    # Check if file exists
    assert os.path.exists(
        Config.SUBMISSION_FILE_PATH
    ), "Submission file not found on disk"

    log(f"Submission generated successfully at {Config.SUBMISSION_FILE_PATH}")
    log("Demonstration completed successfully.")


if __name__ == "__main__":
    main()
