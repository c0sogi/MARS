import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# Import from the provided library
from library.utils import set_seed, clipped_log_loss
from library.config import RANDOM_SEED, SUBMISSION_DIR, INPUT_DIR, TRAIN_METADATA_PATH
from library.data_manager import DataManager
from library.feature_extraction import process_image
from library.expert_factory import get_expert_library
from library.ensemble_selection import GreedySelector


def main():
    # 1. Setup and Reproducibility
    print(">>> Setting up environment...")
    set_seed(RANDOM_SEED)

    # 2. Data Loading and Management
    print("\n>>> Initializing DataManager...")
    dm = DataManager()

    # Load all data (features, morphometrics, labels)
    # This handles caching automatically in ./working/idea_23/
    dm.load_all_data(load_cached_data=True)

    # Verify DataManager state
    assert dm.train_provided is not None, "Train data not loaded"
    assert dm.classes_ is not None, "Classes not loaded"
    print(f"Data Loaded. Classes: {len(dm.classes_)}")
    print(f"Train samples: {dm.train_provided.shape[0]}")
    print(f"Val samples: {dm.val_provided.shape[0]}")

    # 3. Feature Extraction Verification
    # We verify the image processing logic on a single sample from the training set
    print("\n>>> Verifying Feature Extraction logic...")
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    sample_image_path = df_train.iloc[0]["image_path"]  # e.g., "images/123.jpg"

    # process_image expects relative path from INPUT_DIR
    features = process_image(sample_image_path)

    print(f"Extracted features for {sample_image_path}: {features.shape}")
    assert features.shape == (
        11,
    ), "Feature extraction should return 11 morphometric features"
    assert features.dtype == np.float64, "Features should be float64"

    # 4. Data Preparation for Ensemble Selection
    print("\n>>> Preparing data views for Ensemble Selection...")

    # We need to prepare 'Global' and 'Combined' views.
    # The GreedySelector expects a data_map: view_name -> (X_train, y_train, X_val, y_val)
    # We must apply preprocessing (PowerTransformer) before passing to the linear models.

    data_map = {}
    transformers = {}  # Keep transformers to inspect or reuse if needed

    for view in ["Global", "Combined"]:
        # Retrieve raw data
        X_train_raw, y_train, X_val_raw, y_val, _, _, _ = dm.get_view_data(view)

        # Preprocess
        # Note: preprocess_data fits on Train and transforms Train & Val
        X_train_pt, X_val_pt, _ = dm.preprocess_data(X_train_raw, X_val_raw, X_val_raw)

        # Store in map
        data_map[view] = (X_train_pt, y_train, X_val_pt, y_val)

        # Verify shapes
        n_features = X_train_raw.shape[1]
        assert X_train_pt.shape[1] == n_features
        print(f"View '{view}' prepared: {X_train_pt.shape} features.")

    # 5. Expert Library Initialization
    print("\n>>> Initializing Expert Library...")
    experts = get_expert_library()
    print(f"Loaded {len(experts)} experts.")

    # Verify expert structure
    assert "model" in experts[0]
    assert "view" in experts[0]

    # 6. Ensemble Selection (Greedy Forward Selection)
    print("\n>>> Starting Greedy Ensemble Selection...")
    # We use a smaller number of iterations for this demo to ensure speed
    selector = GreedySelector(experts, n_iterations=20, patience=5, seed=RANDOM_SEED)

    selector.fit(data_map)

    assert len(selector.selected_indices) > 0, "No experts were selected!"
    print(f"Selected {len(selector.selected_indices)} experts.")
    print(f"Best Validation Score: {selector.best_score:.5f}")

    # 7. Refit on Full Data (Train + Val)
    print("\n>>> Refitting ensemble on full dataset (Train + Val)...")

    # We need to construct the full dataset and retrain the PowerTransformers
    full_data_map = {}
    test_data_map = {}  # We also prepare test data transformed by the FULL transformer

    for view in ["Global", "Combined"]:
        # Get raw data
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, _ = (
            dm.get_view_data(view)
        )

        # Combine Train and Val
        X_full_raw = np.vstack([X_train_raw, X_val_raw])
        y_full = np.concatenate([y_train, y_val])

        # Fit PowerTransformer on Full Data
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        X_full_pt = pt.fit_transform(X_full_raw).astype(np.float64)

        # Transform Test Data using the full-data transformer
        X_test_pt = pt.transform(X_test_raw).astype(np.float64)

        full_data_map[view] = (X_full_pt, y_full)
        test_data_map[view] = X_test_pt

    selector.refit(full_data_map)

    # 8. Generate Predictions
    print("\n>>> Generating predictions for Test set...")
    test_preds = selector.predict(test_data_map)

    # Verify predictions
    assert test_preds.shape == (
        len(test_ids),
        99,
    ), f"Prediction shape mismatch: {test_preds.shape}"
    assert np.all(test_preds >= 0) and np.all(
        test_preds <= 1
    ), "Probabilities out of bounds"

    # 9. Create Submission File
    print("\n>>> Creating submission file...")
    submission_df = pd.DataFrame(test_preds, columns=dm.classes_)
    submission_df.insert(0, "id", test_ids)

    output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")

    # Final check of the file
    saved_df = pd.read_csv(output_path)
    assert saved_df.shape == (99, 100), "Submission file has incorrect dimensions"
    print("Verification complete. Script finished successfully.")


if __name__ == "__main__":
    main()
