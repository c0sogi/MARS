import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Import from the provided library
from library.config import Config
from library.utils import set_seed, Timer
from library.data_loader import load_datasets
from library.features import generate_features
from library.stacking_engine import NestedStackingTrainer
from library.model_factory import (
    get_lexical_bagger,
    get_community_bagger,
    get_semantic_booster,
    get_semantic_bagger,
    get_metadata_anchor,
    get_meta_learner,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Initializing Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    # We modify the Config class attributes at runtime to make the demo run fast.
    print("Overriding Config parameters for speed optimization...")

    Config.N_FOLDS = 2  # Reduce folds from 5 to 2
    Config.INTERNAL_VAL_SIZE = 0.2

    # Reduce model complexity
    Config.RF_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_EARLY_STOPPING_ROUNDS = 5

    # Reduce feature dimensionality
    Config.TFIDF_MAX_FEATURES = 50

    # Use a separate cache directory for the demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache"

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    # We load the data using the provided loader.
    # For the demo, we will subsample the dataframes immediately after loading.
    train_df, val_df, test_df = load_datasets(load_cached_data=False)

    # Subsample for demonstration speed (e.g., 100 train, 20 val, 20 test)
    # We ensure we have enough positive samples for stratified splitting
    print("Subsampling data for rapid execution...")
    train_df = (
        train_df.groupby(Config.TARGET_COL, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 50), random_state=Config.SEED))
        .reset_index(drop=True)
    )

    val_df = val_df.head(20).reset_index(drop=True)
    test_df = test_df.head(20).reset_index(drop=True)

    print(f"Demo Train Shape: {train_df.shape}")
    print(f"Demo Test Shape: {test_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Feature Generation
    # -------------------------------------------------------------------------
    # Generate features using the PentViewFeatureGenerator via the wrapper function.
    # This handles TF-IDF vectorization, embedding generation, and metadata scaling.
    print("Generating features...")
    train_feats, val_feats, test_feats = generate_features(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify feature dictionary structure
    expected_views = ["lexical", "behavioral", "metadata", "semantic"]
    for view in expected_views:
        assert view in train_feats, f"Missing view {view} in train features"
        assert view in test_feats, f"Missing view {view} in test features"

    print("Feature generation complete and verified.")

    # -------------------------------------------------------------------------
    # 4. Model Factory Verification
    # -------------------------------------------------------------------------
    # Quickly verify that the factory functions return the expected sklearn/xgb objects
    print("Verifying model factory...")
    assert isinstance(get_lexical_bagger(), RandomForestClassifier)
    assert isinstance(get_community_bagger(), RandomForestClassifier)
    assert isinstance(get_semantic_bagger(), RandomForestClassifier)
    assert isinstance(get_metadata_anchor(), LogisticRegression)
    assert isinstance(get_meta_learner(), LogisticRegression)
    assert isinstance(get_semantic_booster(), XGBClassifier)
    print("Model factory verification passed.")

    # -------------------------------------------------------------------------
    # 5. Stacking Pipeline Execution
    # -------------------------------------------------------------------------
    # Instantiate the trainer
    y_train = train_df[Config.TARGET_COL]
    trainer = NestedStackingTrainer(train_feats, y_train)

    # Step A: Train CV (Level 1 OOF Generation)
    print("Starting Stacking CV...")
    oof_preds = trainer.train_cv()

    # Validation: Check OOF shape matches (n_samples, 5_models)
    assert oof_preds.shape == (
        len(train_df),
        5,
    ), f"OOF shape mismatch. Expected ({len(train_df)}, 5), got {oof_preds.shape}"

    # Step B: Train Meta Learner (Level 2)
    print("Training Meta-Learner...")
    trainer.train_meta_learner(oof_preds)

    # Step C: Retrain Full Level 1 Models
    print("Retraining base models on full training set...")
    trainer.retrain_full_models()

    # Step D: Predict on Test Set
    print("Generating predictions on test set...")
    test_probs = trainer.predict_ensemble(test_feats)

    # Validation: Check prediction shape and range
    assert test_probs.shape == (
        len(test_df),
    ), f"Prediction shape mismatch. Expected ({len(test_df)},), got {test_probs.shape}"
    assert np.all(
        (test_probs >= 0) & (test_probs <= 1)
    ), "Predictions contain values outside [0, 1] range."

    print("Pipeline execution successful.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("Generating submission file...")
    submission_df = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": test_probs}
    )

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Verify file existence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Wrap in a Timer to ensure we stay within limits (though demo is small)
    with Timer("Full Demo Execution"):
        run_demo()
