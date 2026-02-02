import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library import config, utils, data_loader, preprocessing, model

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("Starting Leaf Classification Pipeline Demo...")
    set_seed(config.SEED)

    # =========================================================================
    # 1. Data Loading
    # =========================================================================
    print("\n[Step 1] Loading Raw Data...")

    # Force reload from CSVs to demonstrate loading logic (bypass cache for demo)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data_loader.load_data(
        load_cached_data=False
    )

    # Validation: Check shapes
    # Train: 712 samples, 192 features (64 margin + 64 shape + 64 texture)
    assert X_train.shape == (
        712,
        192,
    ), f"Expected X_train shape (712, 192), got {X_train.shape}"
    assert y_train.shape == (
        712,
    ), f"Expected y_train shape (712,), got {y_train.shape}"

    # Validation: 179 samples
    assert X_val.shape == (
        179,
        192,
    ), f"Expected X_val shape (179, 192), got {X_val.shape}"

    # Test: 99 samples
    assert X_test.shape == (
        99,
        192,
    ), f"Expected X_test shape (99, 192), got {X_test.shape}"

    # Classes: 99 unique species
    assert len(classes) == 99, f"Expected 99 classes, got {len(classes)}"

    print("  -> Data loaded and shapes verified successfully.")

    # =========================================================================
    # 2. Preprocessing
    # =========================================================================
    print("\n[Step 2] Preprocessing Features (PowerTransformer + StandardScaler)...")

    # This function fits the pipeline on training data and transforms all sets
    # We use load_cached_data=False to force the computation
    X_train_trans, y_train, X_val_trans, y_val, X_test_trans, test_ids, classes = (
        preprocessing.get_preprocessed_data(load_cached_data=False)
    )

    # Validation: Check that NaNs were not introduced
    assert not np.isnan(
        X_train_trans
    ).any(), "Preprocessing introduced NaNs in training data"
    assert not np.isnan(
        X_val_trans
    ).any(), "Preprocessing introduced NaNs in validation data"

    # Validation: Check Standardization (Mean ~ 0, Std ~ 1)
    # Since the pipeline ends with StandardScaler, features should be standardized
    train_mean = np.mean(X_train_trans)
    train_std = np.std(X_train_trans)

    print(f"  -> Transformed Train Stats: Mean={train_mean:.4f}, Std={train_std:.4f}")
    assert (
        np.abs(train_mean) < 1e-2
    ), "Transformed features should have approximately zero mean"
    assert (
        np.abs(train_std - 1.0) < 1e-2
    ), "Transformed features should have approximately unit variance"

    print("  -> Preprocessing pipeline verified successfully.")

    # =========================================================================
    # 3. Model Training
    # =========================================================================
    print("\n[Step 3] Training Stratified Bagged LDA Model...")

    # For the purpose of this demo, we reduce n_estimators to 5 to speed up execution
    # The default in config is 50.
    demo_n_estimators = 5
    print(f"  -> Initializing model with n_estimators={demo_n_estimators} for speed...")

    clf = model.StratifiedBaggedLDA(
        n_estimators=demo_n_estimators,
        solver=config.LDA_SOLVER,
        shrinkage=config.LDA_SHRINKAGE,
        random_state=config.SEED,
    )

    clf.fit(X_train_trans, y_train)

    # Validation: Check if estimators were fitted
    assert (
        len(clf.estimators_) == demo_n_estimators
    ), "Model failed to fit the correct number of estimators"
    assert (
        clf.n_classes_ == 99
    ), "Model failed to identify the correct number of classes"

    print("  -> Model training complete.")

    # =========================================================================
    # 4. Evaluation
    # =========================================================================
    print("\n[Step 4] Evaluating on Validation Set...")

    # Predict probabilities
    y_pred_val = clf.predict_proba(X_val_trans)

    # Validation: Check probability properties
    assert y_pred_val.shape == (
        179,
        99,
    ), f"Prediction shape mismatch. Expected (179, 99), got {y_pred_val.shape}"

    # Check that probabilities sum to 1 (within floating point error)
    row_sums = y_pred_val.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Predicted probabilities do not sum to 1"

    # Calculate Log Loss using the utility function (handles clipping)
    val_loss = utils.calculate_log_loss(y_val, y_pred_val)
    print(f"  -> Validation Log Loss: {val_loss:.5f}")

    # Sanity check: Loss should be reasonable (e.g., < 5.0 for a decent model, < 4.6 is uniform random)
    # With LDA on this dataset, it usually gets < 0.1, but with only 5 estimators it might be slightly higher.
    assert val_loss < 4.6, "Model performance is worse than random guessing"

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    print("\n[Step 5] Generating Submission for Test Set...")

    # Predict on test set
    y_pred_test = clf.predict_proba(X_test_trans)

    # Save submission
    output_path = config.SUBMISSION_CSV
    utils.save_submission(test_ids, y_pred_test, classes, output_path=output_path)

    # Validation: Verify the output file
    assert os.path.exists(output_path), "Submission file was not created"

    df_sub = pd.read_csv(output_path)
    print(f"  -> Submission saved to {output_path}")
    print(f"  -> Submission dimensions: {df_sub.shape}")

    # Check columns: id + 99 species
    expected_cols = 1 + 99
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, found {df_sub.shape[1]}"
    assert df_sub.shape[0] == 99, f"Expected 99 rows, found {df_sub.shape[0]}"
    assert config.ID_COL in df_sub.columns, "ID column missing from submission"

    # Check that IDs match test IDs
    assert np.array_equal(
        df_sub[config.ID_COL].values, test_ids
    ), "Submission IDs do not match test IDs"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
