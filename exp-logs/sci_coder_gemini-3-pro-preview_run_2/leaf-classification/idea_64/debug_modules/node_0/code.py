import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import provided library modules
from library.utils import set_seed, clipped_log_loss
from library.data_manager import LeafDataManager
from library.transformers import Float64Wrapper, GroupedLDAReducer
from library.model_definitions import get_expert_library
from library.ensemble import GreedyForwardSelector

# Constants
WORKING_DIR = "./working"
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")


def main():
    # 1. Setup
    print("Initializing demonstration...")
    set_seed(42)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n--- Step 1: Loading Data ---")
    data_manager = LeafDataManager(
        metadata_dir="./metadata", cache_dir=os.path.join(WORKING_DIR, "cache")
    )

    # Load data (this will compute morphometrics if not cached)
    data = data_manager.load_data(load_cached_data=True)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, test_ids = data["X_test"], data["test_ids"]

    # Verify shapes
    print(f"Train Data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Val Data:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test Data:  {X_test.shape}, IDs: {test_ids.shape}")

    # Encode labels to ensure consistent ordering for log_loss calculation
    # We fit the encoder on all known species to ensure coverage
    all_species = np.unique(np.concatenate([y_train, y_val]))
    le = LabelEncoder()
    le.fit(all_species)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    print(f"Number of classes: {len(le.classes_)}")

    # 3. Component Verification
    print("\n--- Step 2: Verifying Custom Transformers ---")

    # Test Float64Wrapper
    f64_transformer = Float64Wrapper()
    dummy_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    transformed_dummy = f64_transformer.transform(dummy_data)

    if transformed_dummy.dtype != np.float64:
        raise AssertionError("Float64Wrapper failed to cast data to float64.")
    print("Float64Wrapper verification passed.")

    # Test GroupedLDAReducer
    feature_indices = data_manager.get_feature_indices()
    # Create a reducer that keeps 2 components per group
    # Groups are: margin, shape, texture, physical (4 groups) -> Total output cols should be 4 * 2 = 8
    # Note: Actual output depends on min(n_features, n_classes-1).
    # Our dummy test needs enough classes/features.

    reducer = GroupedLDAReducer(feature_indices=feature_indices, n_components=2)

    # We fit on the actual training data to ensure dimensions work
    reducer.fit(X_train, y_train_enc)
    X_val_reduced = reducer.transform(X_val)

    # Check output dimensions
    # We requested 2 components per group. There are 4 groups.
    # However, if a group has fewer discriminative directions, it might be less.
    # Given 99 classes and >2 features per group, we expect exactly 2 per group.
    expected_dim = 4 * 2
    if X_val_reduced.shape[1] != expected_dim:
        raise AssertionError(
            f"GroupedLDAReducer produced shape {X_val_reduced.shape}, expected columns={expected_dim}"
        )
    print("GroupedLDAReducer verification passed.")

    # 4. Expert Training
    print("\n--- Step 3: Training Experts ---")

    experts = get_expert_library(feature_indices)
    print(f"Loaded {len(experts)} expert definitions.")

    val_preds = {}
    test_preds = {}

    for name, pipeline in experts:
        print(f"Training {name}...")

        # Fit
        pipeline.fit(X_train, y_train)  # Pipeline handles raw labels via LDA

        # Predict Probabilities
        # Note: LDA classes_ are sorted unique labels.
        # We must ensure they match our LabelEncoder for scoring.
        if not np.array_equal(pipeline.classes_, le.classes_):
            raise ValueError(
                f"Model classes do not match LabelEncoder classes for {name}"
            )

        p_val = pipeline.predict_proba(X_val)
        p_test = pipeline.predict_proba(X_test)

        # Score
        score = clipped_log_loss(y_val_enc, p_val)
        print(f"  -> Log Loss: {score:.4f}")

        val_preds[name] = p_val
        test_preds[name] = p_test

    # 5. Ensemble Optimization
    print("\n--- Step 4: Ensemble Optimization ---")

    selector = GreedyForwardSelector(max_ensemble_size=20, verbose=True)
    selector.fit(val_preds, y_val_enc)  # y_val_enc matches the column order of p_val

    print(f"Best Ensemble Score: {selector.best_score_:.4f}")

    # 6. Submission Generation
    print("\n--- Step 5: Generating Submission ---")

    # Compute weighted average of test predictions
    final_test_probs = selector.predict(test_preds)

    # Create DataFrame
    # Columns must be the species names in alphabetical order
    submission_df = pd.DataFrame(final_test_probs, columns=le.classes_)

    # Insert 'id' column at the beginning
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    # Final Validation of Submission File
    saved_df = pd.read_csv(SUBMISSION_PATH)

    # Check 1: Dimensions
    # 99 test samples + header, 99 classes + 1 id column = 100 columns
    expected_rows = 99
    expected_cols = 100

    if saved_df.shape != (expected_rows, expected_cols):
        raise AssertionError(
            f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {saved_df.shape}"
        )

    # Check 2: Probabilities range
    # Drop ID for check
    probs = saved_df.drop(columns=["id"]).values
    if np.min(probs) < 0 or np.max(probs) > 1:
        raise AssertionError("Submission contains probabilities outside [0, 1].")

    print("Submission verification passed.")


if __name__ == "__main__":
    main()
