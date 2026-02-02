import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.features as features
import library.model_factory as model_factory
import library.engine as engine


def main():
    # 1. Setup and Reproducibility
    print("Initializing demonstration...")
    utils.set_seed(config.SEED)

    # 2. Optimization for Speed (Monkey-Patching Hyperparameters)
    # We modify the parameters directly in the model_factory module to ensure
    # the get_base_models() function uses these lightweight settings.
    print("Patching hyperparameters for fast demonstration...")

    # Reduce estimators for Random Forests
    model_factory.LEXICAL_BAGGER_PARAMS.update({"n_estimators": 10, "n_jobs": 1})
    model_factory.COMMUNITY_BAGGER_PARAMS.update({"n_estimators": 10, "n_jobs": 1})
    model_factory.SEMANTIC_BAGGER_PARAMS.update(
        {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}
    )

    # Reduce estimators and relax learning for Boosting models
    model_factory.SEMANTIC_BOOSTER_PARAMS.update(
        {
            "n_estimators": 10,
            "early_stopping_rounds": None,  # Disable ES for very short run or keep small
            "n_jobs": 1,
        }
    )
    model_factory.SEMANTIC_GRADIENT_PARAMS.update(
        {"n_estimators": 10, "early_stopping_round": None, "n_jobs": 1}
    )
    model_factory.TEMPORAL_BOOSTER_PARAMS.update(
        {"n_estimators": 10, "early_stopping_round": None, "n_jobs": 1}
    )

    # Reduce iterations for Linear models
    model_factory.METADATA_ANCHOR_PARAMS.update({"max_iter": 20})
    model_factory.META_LEARNER_PARAMS.update({"max_iter": 20})

    # 3. Data Loading
    print("Loading datasets...")
    # Load full union dataset (Train + Val)
    full_train_df = data_loader.get_union_dataset(load_cached_data=False)
    test_df = data_loader.get_test_dataset()

    # Subsample training data for speed
    train_subset_size = 100
    if len(full_train_df) > train_subset_size:
        print(f"Subsetting training data to {train_subset_size} samples for demo.")
        # Ensure we keep the stratification somewhat by just taking head if sorted,
        # or random sample. Here we just take head for simplicity and speed.
        train_df = full_train_df.head(train_subset_size).copy()
    else:
        train_df = full_train_df.copy()

    # Subsample test data for speed (optional, but good for demo)
    test_subset_size = 50
    if len(test_df) > test_subset_size:
        print(f"Subsetting test data to {test_subset_size} samples for demo.")
        test_df = test_df.head(test_subset_size).copy()

    y_train = train_df[config.TARGET_COL]

    # 4. Feature Engineering
    print("Running Feature Pipeline...")
    pipeline = features.FeaturePipeline()

    # Fit and Transform on Train (Disable cache to force computation)
    # Note: We use a custom cache name to avoid overwriting production caches if they existed
    X_train_dict = pipeline.fit_transform(
        train_df, load_cached_data=False, cache_name="demo_train"
    )

    # Transform Test
    X_test_dict = pipeline.transform(
        test_df, load_cached_data=False, cache_name="demo_test"
    )

    # Verification of Feature Shapes
    n_train = len(train_df)
    n_test = len(test_df)

    assert X_train_dict["meta"].shape[0] == n_train, "Train Meta feature rows mismatch"
    assert X_test_dict["meta"].shape[0] == n_test, "Test Meta feature rows mismatch"
    assert (
        X_train_dict["semantic"].shape[0] == n_train
    ), "Train Semantic feature rows mismatch"

    print("Feature Engineering complete.")

    # 5. Model Training (Stacking Engine)
    print("Initializing Stacking Engine...")
    stacking_engine = engine.StackingEngine()

    # Create manual folds for the subset
    # We use 2 folds for speed instead of the default 5
    n_folds_demo = 2
    config.N_FOLDS = n_folds_demo
    skf = StratifiedKFold(n_splits=n_folds_demo, shuffle=True, random_state=config.SEED)
    folds = list(skf.split(train_df, y_train))

    print(f"Training on {n_folds_demo} folds...")
    stacking_engine.train(X_train_dict, y_train, folds)

    # 6. Inference
    print("Generating predictions...")
    preds = stacking_engine.predict(X_test_dict)

    # Verification of Predictions
    assert len(preds) == n_test, "Prediction length mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # 7. Submission Generation
    print("Creating submission file...")
    submission_df = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": preds}
    )

    # Save submission
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")

    # Final Verification
    if os.path.exists(submission_path):
        saved_df = pd.read_csv(submission_path)
        assert saved_df.shape == (n_test, 2), "Saved submission shape incorrect"
        assert list(saved_df.columns) == [
            "request_id",
            "requester_received_pizza",
        ], "Submission columns incorrect"
        print("Verification successful.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
