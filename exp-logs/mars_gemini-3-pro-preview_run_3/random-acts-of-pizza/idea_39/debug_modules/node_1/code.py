import os
import sys
import shutil
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logging, timer
from library.data_loader import DataLoader
from library.feature_engine import FeatureEngineer
from library.model_registry import get_base_models, get_meta_model
from library.trainer import StackingTrainer


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # ---------------------------------------------------------
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Hex-View Stacking Ensemble Demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config parameters to ensure fast execution for this demo
    logger.info("Overriding Config parameters for speed...")
    Config.WORKING_DIR = "./working/demo_run"

    # Reduce complexity of vectorizers and SVD
    Config.TEXT_VEC_PARAMS["max_features"] = 100
    Config.SUBREDDIT_VEC_PARAMS["max_features"] = 50
    Config.SVD_N_COMPONENTS_TEXT = 5
    Config.SVD_N_COMPONENTS_HISTORY = 5

    # Reduce model complexity (fewer trees/iterations)
    # Note: We modify the dictionaries in Config directly because model_registry uses deepcopy
    Config.MODEL_LEXICAL_RF["n_estimators"] = 5
    Config.MODEL_COMMUNITY_RF["n_estimators"] = 5
    Config.MODEL_SEMANTIC_XGB["n_estimators"] = 10
    Config.MODEL_SEMANTIC_XGB["early_stopping_rounds"] = (
        None  # Disable for tiny dataset
    )
    Config.MODEL_SEMANTIC_RF["n_estimators"] = 5
    Config.MODEL_INTERACTION_XGB["n_estimators"] = 10
    Config.MODEL_INTERACTION_XGB["early_stopping_rounds"] = None
    Config.MODEL_METADATA_LR["max_iter"] = 100
    Config.MODEL_META_LR["max_iter"] = 100

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    logger.info("--- Step 2: Data Loading ---")
    loader = DataLoader()

    # Load Train and Val
    # We force load_cached_data=False to demonstrate processing logic
    df_train = loader.load_data(split="train", load_cached_data=False)
    df_val = loader.load_data(split="val", load_cached_data=False)

    # SUBSET DATA FOR SPEED
    logger.info("Subsetting data to 100 samples for demonstration...")
    df_train = df_train.head(100).copy()
    df_val = df_val.head(50).copy()

    # Verification
    assert "text_combined" in df_train.columns, "text_combined column missing"
    assert "subreddit_text" in df_train.columns, "subreddit_text column missing"
    assert Config.TARGET_COL in df_train.columns, "Target column missing in train"
    logger.info(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}")

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    logger.info("--- Step 3: Feature Engineering ---")
    engineer = FeatureEngineer()

    # Fit on training data
    engineer.fit(df_train)

    # Transform Train
    X_train_dict = engineer.transform(df_train, split="train", load_cached_data=False)
    # Transform Val
    X_val_dict = engineer.transform(df_val, split="val", load_cached_data=False)

    # Verification of Feature Views
    logger.info("Verifying feature views...")

    # Lexical (Sparse)
    assert sp.issparse(X_train_dict["view_lexical"]), "Lexical view should be sparse"
    assert X_train_dict["view_lexical"].shape[0] == 100

    # Behavioral (Sparse)
    assert sp.issparse(
        X_train_dict["view_behavioral"]
    ), "Behavioral view should be sparse"

    # Semantic (Dense Embeddings)
    # all-MiniLM-L6-v2 produces 384-dim embeddings
    assert isinstance(
        X_train_dict["view_semantic"], np.ndarray
    ), "Semantic view should be dense"
    assert X_train_dict["view_semantic"].shape[1] == 384

    # Interaction (Dense SVD + Meta)
    # 5 (Text SVD) + 5 (Hist SVD) + N_Meta
    n_meta = X_train_dict["view_meta"].shape[1]
    expected_interaction_dim = (
        Config.SVD_N_COMPONENTS_TEXT + Config.SVD_N_COMPONENTS_HISTORY + n_meta
    )
    assert X_train_dict["view_interaction"].shape[1] == expected_interaction_dim

    # Target
    y_train = X_train_dict["y"]
    y_val = X_val_dict["y"]
    assert len(y_train) == 100

    logger.info("Feature engineering verification passed.")

    # ---------------------------------------------------------
    # 4. Model Registry Check
    # ---------------------------------------------------------
    logger.info("--- Step 4: Model Registry ---")
    base_models = get_base_models()
    meta_model = get_meta_model()

    assert "lexical_bagger" in base_models
    assert isinstance(base_models["lexical_bagger"], RandomForestClassifier)
    assert isinstance(base_models["semantic_booster"], XGBClassifier)
    assert isinstance(meta_model, LogisticRegression)

    logger.info(f"Base models loaded: {list(base_models.keys())}")

    # ---------------------------------------------------------
    # 5. Stacking Training (OOF + Meta)
    # ---------------------------------------------------------
    logger.info("--- Step 5: Stacking Training ---")
    trainer = StackingTrainer()

    # Override n_folds for speed
    trainer.n_folds = 2

    # Generate Out-Of-Fold Predictions
    oof_df = trainer.generate_oof(X_train_dict, y_train)

    # Verify OOF
    assert oof_df.shape == (
        100,
        6,
    ), f"OOF shape mismatch. Expected (100, 6), got {oof_df.shape}"
    assert not oof_df.isnull().values.any(), "OOF contains NaNs"

    # Train Meta-Learner
    trainer.train_meta(oof_df, y_train)

    # Retrain Final Base Models (Train + Val strategy)
    trainer.retrain_final_models(X_train_dict, y_train, X_val_dict, y_val)

    assert len(trainer.final_models) == 6, "Not all base models were retrained"
    logger.info("Training complete.")

    # ---------------------------------------------------------
    # 6. Prediction on Test Set
    # ---------------------------------------------------------
    logger.info("--- Step 6: Test Prediction ---")

    # Load Test Data
    df_test = loader.load_data(split="test", load_cached_data=False)
    # Subset test for speed
    df_test = df_test.head(50).copy()

    # Transform Test
    X_test_dict = engineer.transform(df_test, split="test", load_cached_data=False)

    # Predict
    final_probs = trainer.predict(X_test_dict)

    # Verify Predictions
    assert len(final_probs) == 50
    assert np.all((final_probs >= 0) & (final_probs <= 1)), "Probabilities out of range"

    logger.info(f"Generated {len(final_probs)} predictions.")
    logger.info(f"Sample predictions: {final_probs[:5]}")

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    logger.info("--- Step 7: Submission Generation ---")

    submission_df = pd.DataFrame(
        {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: final_probs}
    )

    # Save to demo location
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path)
    logger.info(f"Submission saved to {submission_path}")
    logger.info("Demonstration completed successfully!")


if __name__ == "__main__":
    main()
