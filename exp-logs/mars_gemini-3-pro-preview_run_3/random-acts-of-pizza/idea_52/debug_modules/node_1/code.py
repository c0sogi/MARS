import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import logging

# Ensure the library is importable
sys.path.append(os.getcwd())

# Import library components
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.feature_builder as feature_builder
import library.model_factory as model_factory
import library.hybrid_engine as hybrid_engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting End-to-End Demonstration...")

    # =========================================================================
    # 1. CONFIGURATION OVERRIDES (For Speed)
    # =========================================================================
    print("\n[Step 1] Configuring environment for fast demonstration...")

    # Define a demo working directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override paths in the config module instance
    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    config.MODEL_DIR = os.path.join(DEMO_DIR, "models")
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Ensure directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Setup logging
    utils.setup_logging(os.path.join(DEMO_DIR, "demo.log"))
    utils.set_seed(config.SEED)

    # Patch Hyperparameters in config to make training very fast
    # We modify the dictionaries in place so model_factory picks up changes
    print("Patching hyperparameters for speed...")

    # 1. Lexical Bagger (RF)
    config.LEXICAL_BAGGER_PARAMS.update(
        {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}
    )

    # 2. Community Bagger (RF)
    config.COMMUNITY_BAGGER_PARAMS.update(
        {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}
    )

    # 3. Semantic Booster (XGB)
    config.SEMANTIC_BOOSTER_PARAMS.update(
        {"n_estimators": 10, "max_depth": 2, "early_stopping_rounds": 5, "n_jobs": 1}
    )

    # 4. Semantic Bagger (RF)
    config.SEMANTIC_BAGGER_PARAMS.update(
        {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}
    )

    # 5. Metadata Anchor (LR)
    config.METADATA_ANCHOR_PARAMS.update({"max_iter": 50})

    # =========================================================================
    # 2. DATA LOADING & SAMPLING
    # =========================================================================
    print("\n[Step 2] Loading and sampling data...")

    # Load full datasets
    df_train_full = data_loader.load_dataset("full_train")  # Combines train+val
    df_test_full = data_loader.load_dataset("test")

    # Sample a small subset for demonstration (e.g., 50 samples)
    # We ensure we have both classes in the sample
    SAMPLE_SIZE = 50

    df_train_demo = (
        df_train_full.groupby(config.TARGET_COL)
        .apply(lambda x: x.sample(n=SAMPLE_SIZE // 2, random_state=config.SEED))
        .reset_index(drop=True)
    )

    df_test_demo = df_test_full.sample(n=20, random_state=config.SEED).reset_index(
        drop=True
    )

    print(f"Demo Train Shape: {df_train_demo.shape}")
    print(f"Demo Test Shape: {df_test_demo.shape}")

    # =========================================================================
    # 3. FEATURE ENGINEERING
    # =========================================================================
    print("\n[Step 3] Running Feature Pipeline...")

    # Initialize pipeline with the demo cache directory
    pipeline = feature_builder.FeaturePipeline(cache_dir=config.CACHE_DIR)

    # Fit on training data
    pipeline.fit(df_train_demo)

    # Transform Train and Test
    # We use 'demo_train' and 'demo_test' as split names for caching
    X_train_dict = pipeline.transform(df_train_demo, split_name="demo_train")
    X_test_dict = pipeline.transform(df_test_demo, split_name="demo_test")

    # Validate Feature Dictionary
    expected_keys = ["X_lexical", "X_behavioral", "X_semantic", "X_meta"]
    for key in expected_keys:
        assert key in X_train_dict, f"Missing key {key} in feature dict"
        assert X_train_dict[key].shape[0] == len(
            df_train_demo
        ), f"Shape mismatch for {key}"

    print("Feature engineering complete. Feature shapes verified.")

    # =========================================================================
    # 4. MODEL TRAINING (HYBRID ENGINE)
    # =========================================================================
    print("\n[Step 4] Training Models via Hybrid Engine...")

    trainer = hybrid_engine.HybridTrainer(model_dir=config.MODEL_DIR)

    # Define models to train
    # We will train one volatile and one stable model for demonstration
    volatile_models = [
        ("semantic_booster", "SemanticBooster"),  # XGB on Embeddings
    ]

    stable_models = [
        ("lexical_bagger", "LexicalBagger"),  # RF on TF-IDF
        ("metadata_anchor", "MetadataAnchor"),  # LR on Metadata
        ("community_bagger", "CommunityBagger"),  # RF on Subreddits
    ]

    # Get Stratified Folds (using 2 folds for speed)
    folds = list(
        data_loader.get_stratified_folds(
            df_train_demo, n_splits=2, random_state=config.SEED
        )
    )

    y_train = df_train_demo[config.TARGET_COL]

    # Container for Level 1 OOF Predictions
    oof_preds_dict = {}

    # 4a. Train Volatile Learners
    for model_name, learner_name in volatile_models:
        oof = trainer.train_volatile(
            model_name, learner_name, X_train_dict, y_train, folds
        )
        oof_preds_dict[model_name] = oof

        # Verify fold models exist
        for i in range(2):
            model_path = os.path.join(config.MODEL_DIR, f"{model_name}_fold_{i}.joblib")
            assert os.path.exists(model_path), f"Fold model {model_path} not saved."

    # 4b. Train Stable Learners
    for model_name, learner_name in stable_models:
        oof = trainer.train_stable(
            model_name, learner_name, X_train_dict, y_train, folds
        )
        oof_preds_dict[model_name] = oof

        # Verify single full model exists
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}.joblib")
        assert os.path.exists(model_path), f"Full model {model_path} not saved."

    # 4c. Train Meta Learner
    print("Training Meta-Learner on stacked OOF predictions...")

    # Ensure consistent order
    level1_model_names = [m[0] for m in volatile_models] + [m[0] for m in stable_models]
    X_oof_stacked = np.column_stack(
        [oof_preds_dict[name] for name in level1_model_names]
    )

    trainer.train_meta(X_oof_stacked, y_train)

    assert os.path.exists(
        os.path.join(config.MODEL_DIR, "meta_learner.joblib")
    ), "Meta learner not saved."

    # =========================================================================
    # 5. INFERENCE
    # =========================================================================
    print("\n[Step 5] Generating Predictions on Test Set...")

    predictor = hybrid_engine.HybridPredictor(model_dir=config.MODEL_DIR)

    # Separate names for predictor
    v_names = [m[0] for m in volatile_models]
    s_names = [m[0] for m in stable_models]

    # Predict
    # Note: HybridPredictor internally stacks predictions in the order of calls
    # We need to ensure the order matches how we stacked X_oof_stacked.
    # The predictor implementation:
    #   1. Iterates volatile_models list -> appends to list
    #   2. Iterates stable_models list -> appends to list
    #   3. Stacks columns
    # This matches our level1_model_names construction above.

    final_preds = predictor.predict(X_test_dict, v_names, s_names)

    # Validation
    assert len(final_preds) == len(df_test_demo), "Prediction length mismatch."
    assert np.all(
        (final_preds >= 0) & (final_preds <= 1)
    ), "Predictions out of probability range."

    print(f"Generated {len(final_preds)} predictions.")
    print(f"Sample predictions: {final_preds[:5]}")

    # =========================================================================
    # 6. SUBMISSION GENERATION
    # =========================================================================
    print("\n[Step 6] Creating Submission File...")

    submission_df = pd.DataFrame(
        {
            "request_id": df_test_demo["request_id"],
            "requester_received_pizza": final_preds,
        }
    )

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    # Final check
    assert os.path.exists(submission_path), "Submission file was not created."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
