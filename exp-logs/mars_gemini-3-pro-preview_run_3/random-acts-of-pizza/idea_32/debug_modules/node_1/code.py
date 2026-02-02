import os
import sys
import numpy as np
import pandas as pd
import warnings
import logging

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import Library Components
import library.config as config
from library.utils import set_seed, print_header, print_info
from library.features import FeatureFactory
from library.ensemble import StackingPipeline


def main():
    # =========================================================================
    # 1. SETUP & OPTIMIZATION
    # =========================================================================
    print_header("DEMO: Setup and Configuration Overrides")

    # Set deterministic seed
    set_seed(config.SEED)

    # Override hyperparameters for speed (Demo Mode)
    print_info("Overriding config parameters for rapid execution...")

    # Reduce Random Forest estimators
    config.LEXICAL_RF_PARAMS["n_estimators"] = 10
    config.BEHAVIORAL_RF_PARAMS["n_estimators"] = 10
    config.SEMANTIC_RF_PARAMS["n_estimators"] = 10

    # Reduce XGBoost estimators
    config.SEMANTIC_XGB_PARAMS["n_estimators"] = 10

    # Reduce Vocabulary sizes to speed up vectorization
    config.TEXT_VOCAB_SIZE = 500
    config.HISTORY_VOCAB_SIZE = 200

    # =========================================================================
    # 2. FEATURE ENGINEERING
    # =========================================================================
    print_header("DEMO: Feature Engineering")

    ff = FeatureFactory()

    # Load Raw Data
    train_df, val_df, test_df = ff.load_raw_data()

    # Verify Data Loading
    print_info(f"Train shape: {train_df.shape}")
    print_info(f"Val shape: {val_df.shape}")
    print_info(f"Test shape: {test_df.shape}")

    assert len(train_df) == 2302, "Train set size mismatch"
    assert len(val_df) == 576, "Val set size mismatch"
    assert len(test_df) == 1162, "Test set size mismatch"

    # Generate Views
    # Note: We set load_cached_data=False to demonstrate generation logic,
    # though the library handles caching automatically.

    # 1. Metadata View
    X_train_meta, X_val_meta, X_test_meta = ff.create_metadata_view(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert X_train_meta.shape[1] == len(
        config.METADATA_FEATURES
    ), "Metadata feature count mismatch"

    # 2. Lexical View (Text TF-IDF)
    X_train_lex, X_val_lex, X_test_lex = ff.create_lexical_view(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert X_train_lex.shape[1] <= config.TEXT_VOCAB_SIZE, "Lexical vocab size exceeded"

    # 3. Behavioral View (Subreddit History)
    X_train_beh, X_val_beh, X_test_beh = ff.create_behavioral_view(
        train_df, val_df, test_df, load_cached_data=False
    )

    # 4. Semantic View (Embeddings)
    # This might take a moment depending on CPU/GPU
    X_train_sem, X_val_sem, X_test_sem = ff.create_semantic_view(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert (
        X_train_sem.shape[1] == 384
    ), "Embedding dimension mismatch (expected 384 for MiniLM)"

    # Organize into dictionaries for the pipeline
    X_train_dict = {
        "metadata": X_train_meta,
        "lexical": X_train_lex,
        "behavioral": X_train_beh,
        "semantic": X_train_sem,
    }

    X_val_dict = {
        "metadata": X_val_meta,
        "lexical": X_val_lex,
        "behavioral": X_val_beh,
        "semantic": X_val_sem,
    }

    X_test_dict = {
        "metadata": X_test_meta,
        "lexical": X_test_lex,
        "behavioral": X_test_beh,
        "semantic": X_test_sem,
    }

    # Extract Targets
    y_train = train_df[config.TARGET_COL].values
    y_val = val_df[config.TARGET_COL].values

    # =========================================================================
    # 3. STACKING PIPELINE - CROSS VALIDATION
    # =========================================================================
    print_header("DEMO: Stacking Pipeline Execution")

    pipeline = StackingPipeline()

    # Run CV to get OOF predictions
    oof_preds = pipeline.run_cross_validation(X_train_dict, y_train)

    # Verify OOF shape: (n_samples, n_base_models)
    expected_models = 5  # Lexical, Community, SemanticXGB, SemanticRF, MetadataAnchor
    assert oof_preds.shape == (
        len(train_df),
        expected_models,
    ), f"OOF shape mismatch. Expected {(len(train_df), expected_models)}, got {oof_preds.shape}"

    # Train Meta-Learner
    pipeline.train_meta_learner(oof_preds, y_train)

    assert hasattr(pipeline.meta_learner, "coef_"), "Meta-learner not trained"

    # =========================================================================
    # 4. RETRAINING & INFERENCE
    # =========================================================================
    print_header("DEMO: Retraining and Inference")

    # Retrain base models on full data (Train + Val or Train w/ Val stopping)
    pipeline.retrain_final_models(X_train_dict, y_train, X_val_dict, y_val)

    assert (
        len(pipeline.final_models) == expected_models
    ), "Not all base models were retrained"

    # Predict on Test Set
    final_preds = pipeline.predict(X_test_dict)

    assert final_preds.shape == (
        len(test_df),
    ), f"Prediction shape mismatch. Expected {(len(test_df),)}, got {final_preds.shape}"

    # Check probability range
    assert np.all(
        (final_preds >= 0) & (final_preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # =========================================================================
    # 5. SUBMISSION
    # =========================================================================
    print_header("DEMO: Submission Generation")

    pipeline.generate_submission(test_df, final_preds)

    if os.path.exists(config.SUBMISSION_PATH):
        print_info(f"Verified submission file exists at: {config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print_header("DEMO COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
