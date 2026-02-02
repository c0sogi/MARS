import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy import sparse

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data_processor import DataProcessor
from library.feature_extraction import FeatureGenerator
from library.stacking_manager import StackingEngine


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print("Initializing task with configuration overrides for speed...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for rapid demonstration
    # We use a small sample size and reduced model parameters
    Config.DEBUG_SAMPLE_SIZE = 100
    Config.N_FOLDS = 2

    # Reduce Random Forest estimators
    Config.L1_RF_LEXICAL_PARAMS["n_estimators"] = 10
    Config.L1_RF_SEMANTIC_PARAMS["n_estimators"] = 10

    # Reduce XGBoost complexity and switch to CPU for small data speed
    Config.L1_XGB_SEMANTIC_PARAMS.update(
        {
            "n_estimators": 10,
            "early_stopping_rounds": 2,
            "device": "cpu",
            "tree_method": "hist",
        }
    )

    # Use a separate cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache"
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Processing
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Data Processing ---")
    processor = DataProcessor()

    # Force reprocessing (load_cached_data=False) to ensure we use the sampled subset
    train_df, val_df, test_df = processor.process_data(load_cached_data=False)

    # Validation
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train set size mismatch."
    assert Config.TEXT_COL in train_df.columns, "Text column missing."
    assert Config.TARGET_COL in train_df.columns, "Target column missing in train."
    # Check for no NaNs in numerical columns (imputation check)
    assert (
        not train_df.select_dtypes(include=np.number).isnull().any().any()
    ), "NaNs found in numerical columns."

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Feature Extraction ---")
    extractor = FeatureGenerator()

    # 3a. Lexical View (Sparse)
    # Force regeneration to match the sampled data
    X_lex_train, X_lex_val, X_lex_test = extractor.get_lexical_view(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation
    assert sparse.issparse(X_lex_train), "Lexical view should be sparse."
    assert X_lex_train.shape[0] == len(train_df), "Lexical train rows mismatch."
    # Check dimensions (TF-IDF max_features + metadata cols)
    # Metadata cols count depends on the dataframe, but should be consistent
    assert (
        X_lex_train.shape[1] == X_lex_test.shape[1]
    ), "Feature dimension mismatch between train and test."

    # 3b. Semantic View (Dense)
    X_sem_train, X_sem_val, X_sem_test = extractor.get_semantic_view(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Validation
    assert isinstance(
        X_sem_train, np.ndarray
    ), "Semantic view should be dense numpy array."
    assert X_sem_train.shape[0] == len(train_df), "Semantic train rows mismatch."

    print(f"Lexical Feature Shape: {X_lex_train.shape}")
    print(f"Semantic Feature Shape: {X_sem_train.shape}")

    # -------------------------------------------------------------------------
    # 4. Stacking Engine: Cross-Validation & Meta-Learning
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Stacking Engine (Fit CV) ---")
    engine = StackingEngine()

    y_train = train_df[Config.TARGET_COL].values

    # Fit CV (Trains Level 1 models via CV, generates OOFs, trains Level 2 Meta-Learner)
    engine.fit_cv(X_lex_train, X_sem_train, y_train)

    # Validation
    assert engine.meta_learner is not None, "Meta-learner should be trained."
    assert hasattr(engine.meta_learner, "coef_"), "Meta-learner not fitted properly."

    # -------------------------------------------------------------------------
    # 5. Stacking Engine: Retraining Base Models
    # -------------------------------------------------------------------------
    print("\n--- Step 5: Retraining Base Models ---")
    # Retrain on the full training set (in this demo context, 'train_df' is our full training set)
    engine.retrain_base_models(X_lex_train, X_sem_train, y_train)

    # Validation
    assert engine.final_lexical_rf is not None, "Final Lexical RF not retrained."
    assert engine.final_semantic_xgb is not None, "Final Semantic XGB not retrained."

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n--- Step 6: Inference on Test Set ---")
    predictions = engine.predict(X_lex_test, X_sem_test)

    # Validation
    assert len(predictions) == len(test_df), "Prediction count mismatch."
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range."

    print(f"Generated {len(predictions)} predictions.")
    print(f"Sample predictions: {predictions[:5]}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Saving Submission ---")
    submission_df = pd.DataFrame(
        {"request_id": test_df[Config.ID_COL], "requester_received_pizza": predictions}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    # Final Validation
    assert os.path.exists(submission_path), "Submission file not created."
    loaded_sub = pd.read_csv(submission_path)
    assert list(loaded_sub.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns incorrect."
    assert len(loaded_sub) == len(test_df), "Submission row count incorrect."

    print(f"Submission saved to {submission_path}")
    print("Demo completed successfully.")


if __name__ == "__main__":
    main()
