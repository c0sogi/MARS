import os
import sys
import shutil
import warnings
import pandas as pd
import numpy as np
import joblib

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import library components
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.data_factory import DataFactory
from library.features import FeaturePipeline
from library.engine import HybridEnsembleEngine
from library.model_zoo import get_hept_view_models


def main():
    print(">>> Starting Demonstration Script for Hept-View Architecture")

    # =========================================================================
    # 1. Configuration Patching for Speed & Demo Isolation
    # =========================================================================
    print(">>> Patching Configuration for fast execution...")

    # Set a separate working directory for this demo to avoid messing with real training artifacts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    ensure_dir(DEMO_DIR)

    # Patch Directory Paths in Config
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.CACHE_PROCESSED_DATA_PATH = os.path.join(DEMO_DIR, "processed_data.parquet")
    Config.CACHE_EMBEDDINGS_PATH = os.path.join(DEMO_DIR, "embeddings.npy")

    ensure_dir(Config.SUBMISSION_DIR)

    # Patch Global Params
    Config.N_FOLDS = 2  # Reduce folds from 5 to 2
    Config.N_JOBS = 2  # Limit threads for demo

    # Patch Model Hyperparameters (Drastically reduce for speed)
    # We modify the dictionaries in Config directly because the Engine uses them via get_hept_view_models()

    # RF Params (Lexical, Community, Semantic Bagger)
    rf_fast_params = {
        "n_estimators": 10,
        "max_depth": 5,
        "n_jobs": 2,
        "random_state": Config.SEED,
        "verbose": 0,
    }
    Config.LEXICAL_BAGGER_PARAMS.update(rf_fast_params)
    Config.COMMUNITY_BAGGER_PARAMS.update(rf_fast_params)
    Config.SEMANTIC_BAGGER_PARAMS.update(rf_fast_params)

    # Gradient Boosting Params (Semantic Booster/Gradient, Temporal Booster)
    # Note: We keep early_stopping_rounds but reduce estimators
    gb_fast_params = {
        "n_estimators": 10,
        "n_jobs": 2,
        "random_state": Config.SEED,
        "early_stopping_rounds": 5,  # Short patience
    }
    Config.SEMANTIC_BOOSTER_PARAMS.update(gb_fast_params)
    Config.SEMANTIC_GRADIENT_PARAMS.update(gb_fast_params)
    Config.TEMPORAL_BOOSTER_PARAMS.update(gb_fast_params)

    # Linear Model Params
    lr_fast_params = {"max_iter": 20, "n_jobs": 2, "random_state": Config.SEED}
    Config.METADATA_ANCHOR_PARAMS.update(lr_fast_params)
    Config.META_LEARNER_PARAMS.update(lr_fast_params)

    # Set Seed
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print(">>> Loading Data (Debug Mode)...")

    # Load subset of data (200 train, 50 test)
    # Note: load_union_data caches the result. Since we changed WORKING_DIR, it creates a new cache.
    df_train = DataFactory.load_union_data(load_cached_data=False, debug_size=200)
    df_test = DataFactory.load_test_data(debug_size=50)

    # Validation
    assert len(df_train) == 200, f"Expected 200 train samples, got {len(df_train)}"
    assert len(df_test) == 50, f"Expected 50 test samples, got {len(df_test)}"
    assert Config.TARGET_COL in df_train.columns, "Target column missing in train"
    print("Data loaded successfully.")

    # =========================================================================
    # 3. Feature Engineering
    # =========================================================================
    print(">>> Generating Features...")

    pipeline = FeaturePipeline(df_train, df_test)

    # 3.1 Augmented Metadata
    X_train_meta, X_test_meta = pipeline.get_augmented_metadata(load_cached_data=False)
    assert X_train_meta.shape[0] == 200
    assert X_test_meta.shape[0] == 50
    assert not np.isnan(X_train_meta).any(), "NaNs found in metadata"

    # 3.2 Granular Lexical (Sparse)
    X_train_lex, X_test_lex = pipeline.get_granular_lexical(load_cached_data=False)
    assert X_train_lex.shape[0] == 200

    # 3.3 Behavioral (Sparse)
    X_train_beh, X_test_beh = pipeline.get_behavioral_sparse(load_cached_data=False)
    assert X_train_beh.shape[0] == 200

    # 3.4 Semantic Dense (Embeddings)
    # This might take a few seconds to download the model if not cached, but usually fast
    X_train_sem, X_test_sem = pipeline.get_semantic_dense(load_cached_data=False)
    assert X_train_sem.shape[1] == Config.EMBEDDING_DIM

    print("Features generated successfully.")

    # Prepare dictionaries for the Engine
    X_train_dict = {
        "lexical": X_train_lex,
        "behavioral": X_train_beh,
        "semantic": X_train_sem,
        "metadata": X_train_meta,
    }

    X_test_dict = {
        "lexical": X_test_lex,
        "behavioral": X_test_beh,
        "semantic": X_test_sem,
        "metadata": X_test_meta,
    }

    y_train = df_train[Config.TARGET_COL]

    # =========================================================================
    # 4. Hybrid Ensemble Engine Execution
    # =========================================================================
    print(">>> Initializing Engine and Training...")

    engine = HybridEnsembleEngine(
        X_train_dict=X_train_dict,
        y_train=y_train,
        X_test_dict=X_test_dict,
        test_ids=df_test[Config.ID_COL].values,
        output_dir=Config.WORKING_DIR,
    )

    # Run the full pipeline: L1 CV -> L2 Train -> Stable Retrain -> Prediction
    engine.train_cv_and_predict()

    print("Engine execution completed.")

    # =========================================================================
    # 5. Validation of Outputs
    # =========================================================================
    print(">>> Validating Outputs...")

    # Check OOF Predictions
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError("OOF predictions file not found.")

    oof_df = pd.read_csv(oof_path)
    # Check if we have columns for all models
    expected_models = get_hept_view_models().keys()
    for model in expected_models:
        if model not in oof_df.columns:
            raise ValueError(f"Missing OOF predictions for {model}")
    print("OOF predictions validated.")

    # Check Model Artifacts
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    # Check for a stable model
    if not os.path.exists(os.path.join(models_dir, "lexical_bagger.joblib")):
        raise FileNotFoundError("Stable model artifact missing.")
    # Check for a volatile fold model
    if not os.path.exists(os.path.join(models_dir, "semantic_booster_fold_0.joblib")):
        raise FileNotFoundError("Volatile fold model artifact missing.")
    # Check for meta learner
    if not os.path.exists(os.path.join(models_dir, "meta_learner.joblib")):
        raise FileNotFoundError("Meta-learner artifact missing.")
    print("Model artifacts validated.")

    # Check Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file not found.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify Submission Shape and Content
    assert (
        len(sub_df) == 50
    ), f"Submission has {len(sub_df)} rows, expected 50 (debug size)"
    assert Config.ID_COL in sub_df.columns
    assert Config.TARGET_COL in sub_df.columns

    # Check values are probabilities
    preds = sub_df[Config.TARGET_COL]
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions out of probability range [0, 1]"

    print(f"Submission validated. Shape: {sub_df.shape}")
    print(f"Sample:\n{sub_df.head(3)}")

    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    main()
