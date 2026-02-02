import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure the current directory is in the path for imports
sys.path.append(os.getcwd())

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import training_engine


def main():
    # 1. Setup and Configuration Override for Speed
    # =========================================================================
    utils.print_header("1. Setup and Configuration")

    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # Override config parameters to ensure the demo runs quickly (< 5 mins)
    # We modify the dictionaries in the loaded config module directly.
    print("Overriding hyperparameters for fast demonstration...")

    # Reduce Random Forest estimators
    config.LEXICAL_BAGGER_PARAMS["n_estimators"] = 10
    config.COMMUNITY_BAGGER_PARAMS["n_estimators"] = 10
    config.SEMANTIC_BAGGER_PARAMS["n_estimators"] = 10

    # Reduce XGBoost estimators and depth
    config.SEMANTIC_BOOSTER_PARAMS["n_estimators"] = 10
    config.SEMANTIC_BOOSTER_PARAMS["max_depth"] = 2

    # Reduce Logistic Regression iterations
    config.METADATA_ANCHOR_PARAMS["max_iter"] = 50
    config.META_LEARNER_PARAMS["max_iter"] = 50

    # Reduce TF-IDF features for speed
    config.TFIDF_PARAMS["max_features"] = 100
    config.TFIDF_PARAMS["min_df"] = 1
    config.TFIDF_PARAMS["stop_words"] = None

    # Define a demo working directory
    demo_dir = os.path.join(config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    # Update cache dir in config to avoid messing with real experiment cache
    config.CACHE_DIR = os.path.join(demo_dir, "cache")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 2. Data Loading
    # =========================================================================
    utils.print_header("2. Data Loading")

    # Load a small sample of data (100 rows) for debugging/demo
    SAMPLE_SIZE = 100
    train_df, val_df, test_df = data_loader.load_data(
        load_cached=False, sample_size=SAMPLE_SIZE
    )

    # Validation
    assert (
        len(train_df) == SAMPLE_SIZE
    ), f"Expected {SAMPLE_SIZE} train rows, got {len(train_df)}"
    assert (
        len(val_df) == SAMPLE_SIZE
    ), f"Expected {SAMPLE_SIZE} val rows, got {len(val_df)}"
    assert (
        len(test_df) == SAMPLE_SIZE
    ), f"Expected {SAMPLE_SIZE} test rows, got {len(test_df)}"
    assert config.TARGET_COL in train_df.columns, "Target column missing in train"
    print("Data loaded and validated successfully.")

    # 3. Feature Engineering (View Generation)
    # =========================================================================
    utils.print_header("3. Feature Engineering")

    # Instantiate ViewBuilder
    view_builder = feature_engineering.ViewBuilder()

    # Fit on training data
    view_builder.fit(train_df)

    # Transform all splits
    # We disable loading from cache to ensure the code runs fully
    X_train_views = view_builder.transform(train_df, "train", load_cached=False)
    X_val_views = view_builder.transform(val_df, "val", load_cached=False)
    X_test_views = view_builder.transform(test_df, "test", load_cached=False)

    # Extract targets
    y_train = train_df[config.TARGET_COL]
    y_val = val_df[config.TARGET_COL]

    # Validation of Views
    required_views = ["metadata", "lexical", "behavioral", "semantic"]
    for view_name in required_views:
        assert view_name in X_train_views, f"Missing view: {view_name}"

        # Check rows match
        n_rows = X_train_views[view_name].shape[0]
        assert (
            n_rows == SAMPLE_SIZE
        ), f"View {view_name} has {n_rows} rows, expected {SAMPLE_SIZE}"

    print("Views generated and validated successfully.")
    print(f"Metadata shape: {X_train_views['metadata'].shape}")
    print(f"Lexical shape: {X_train_views['lexical'].shape}")
    print(f"Semantic shape: {X_train_views['semantic'].shape}")

    # 4. Model Training (Stacking Ensemble)
    # =========================================================================
    utils.print_header("4. Model Training")

    # Instantiate Trainer
    trainer = training_engine.StackingTrainer()

    # Fit the ensemble
    # This runs the 5-fold CV OOF generation, Meta-Learner training, and base learner retraining
    trainer.fit(X_train_views, y_train, X_val_views, y_val)

    # Validation of Trainer state
    assert hasattr(trainer, "meta_learner"), "Meta-learner not found in trainer"
    assert (
        len(trainer.final_models) == 5
    ), f"Expected 5 base models, found {len(trainer.final_models)}"
    print("Ensemble training completed successfully.")

    # 5. Prediction and Submission
    # =========================================================================
    utils.print_header("5. Prediction and Submission")

    # Generate predictions
    predictions = trainer.predict(X_test_views)

    # Validation of predictions
    assert len(predictions) == SAMPLE_SIZE, "Prediction count mismatch"
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Save submission
    submission_path = os.path.join(demo_dir, "submission.csv")
    request_ids = test_df[config.ID_COL].values

    trainer.save_predictions(request_ids, predictions, submission_path)

    # Verify file creation
    if os.path.exists(submission_path):
        print(f"Submission file created at: {submission_path}")

        # Check content
        sub_df = pd.read_csv(submission_path)
        print(f"Submission head:\n{sub_df.head()}")
        assert sub_df.shape == (SAMPLE_SIZE, 2), "Submission shape mismatch"
        assert list(sub_df.columns) == [
            "request_id",
            "requester_received_pizza",
        ], "Submission columns mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    utils.print_header("Demo Completed Successfully")


if __name__ == "__main__":
    main()
