import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse

# Import library components
import library.config as config
from library.trainer import Trainer
from library.utils import set_seed


def main():
    print("Starting demonstration of Pizza Request Prediction Pipeline...")

    # -------------------------------------------------------------------------
    # 1. OPTIMIZE FOR SPEED (Monkey-Patching Config)
    # -------------------------------------------------------------------------
    print("Configuring hyperparameters for fast demonstration...")

    # Reduce Cross-Validation Folds
    config.N_FOLDS = 2

    # Reduce Random Forest Estimators (Base Learners)
    config.RF_LEXICAL_PARAMS["n_estimators"] = 5
    config.RF_BEHAVIORAL_PARAMS["n_estimators"] = 5
    config.RF_SEMANTIC_PARAMS["n_estimators"] = 5

    # Reduce XGBoost Estimators
    config.XGB_SEMANTIC_PARAMS["n_estimators"] = 10
    config.XGB_EARLY_STOPPING_ROUNDS = 5

    # Reduce Logistic Regression Iterations (Meta Learner & Anchor)
    config.LR_ANCHOR_PARAMS["max_iter"] = 50
    config.META_LEARNER_PARAMS["max_iter"] = 50

    # Reduce Feature Dimensionality for TF-IDF
    config.TFIDF_TEXT_PARAMS["max_features"] = 100
    config.TFIDF_SUBREDDIT_PARAMS["max_features"] = 50

    # Reduce Embedding Batch Size (Safe for mixed environments)
    config.EMBEDDING_BATCH_SIZE = 16

    # Ensure reproducibility
    set_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 2. INSTANTIATE AND RUN TRAINER
    # -------------------------------------------------------------------------
    print("\nInitializing Trainer...")
    trainer = Trainer()

    print("Running Training Pipeline (Feature Engineering + Stacking)...")
    # We set load_cached_data=False to demonstrate the full feature generation process
    # In a real iterative run, you would likely set this to True.
    model, data_dict = trainer.train(load_cached_data=False)

    # -------------------------------------------------------------------------
    # 3. VERIFY LOGIC AND DATA INTEGRITY
    # -------------------------------------------------------------------------
    print("\nVerifying Data Integrity...")

    # Check 1: Verify Target Variable Shapes
    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_val shape: {y_val.shape}")

    assert len(y_train) > 0, "Training targets should not be empty"
    assert len(y_val) > 0, "Validation targets should not be empty"

    # Check 2: Verify Feature Views Existence and Shapes
    views = ["lexical", "community", "semantic", "metadata"]
    splits = ["train", "val", "test"]

    for view in views:
        for split in splits:
            key = f"X_{split}_{view}"
            assert key in data_dict, f"Missing feature view: {key}"

            features = data_dict[key]
            n_samples = features.shape[0]

            # Match samples with IDs or Targets
            if split == "train":
                expected_samples = len(y_train)
            elif split == "val":
                expected_samples = len(y_val)
            else:
                expected_samples = len(data_dict["id_test"])

            assert (
                n_samples == expected_samples
            ), f"Dimension mismatch for {key}: Got {n_samples}, expected {expected_samples}"

            print(f"  {key} verified: shape={features.shape}, type={type(features)}")

    # Check 3: Verify Model Training Status
    print("\nVerifying Model Training...")
    assert hasattr(
        model, "trained_base_learners"
    ), "Model should have trained_base_learners attribute"
    assert (
        len(model.trained_base_learners) == 5
    ), f"Expected 5 base learners, found {len(model.trained_base_learners)}"

    for name, learner in model.trained_base_learners.items():
        print(f"  Base Learner '{name}' is trained.")
        # Basic check to ensure the underlying sklearn/xgboost model is fitted
        # (Attributes vary by library, but checking for known attributes helps)
        if hasattr(learner, "classes_"):  # Sklearn classifiers
            assert learner.classes_ is not None
        elif hasattr(learner, "feature_importances_"):  # Trees
            assert learner.feature_importances_ is not None

    assert hasattr(
        model.meta_learner, "coef_"
    ), "Meta-learner should be fitted (have coefficients)"
    print("  Meta-Learner is trained.")

    # -------------------------------------------------------------------------
    # 4. GENERATE SUBMISSION
    # -------------------------------------------------------------------------
    print("\nGenerating Submission...")
    trainer.generate_submission(data_dict)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated at: {submission_path}")
    print(sub_df.head())

    # Check row count (Test set size is 1162)
    expected_test_size = 1162
    assert (
        len(sub_df) == expected_test_size
    ), f"Submission row count mismatch. Expected {expected_test_size}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check probability range
    probs = sub_df["requester_received_pizza"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities must be between 0 and 1"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
