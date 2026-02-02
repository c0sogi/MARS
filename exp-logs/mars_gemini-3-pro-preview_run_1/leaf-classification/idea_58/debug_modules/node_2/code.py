import os
import sys
import numpy as np
import pandas as pd
import logging
import shutil
from library import utils, features, pipeline, model

# Configure logging to suppress verbose output for the demo, only showing errors/critical info
# or specific print statements we explicitly make.
logging.basicConfig(level=logging.ERROR)


def run_demo():
    print("--- Starting Library Demonstration ---")

    # 1. Setup and Utils Verification
    print("\n[1] Verifying library.utils...")
    utils.set_seed(42)

    # Test cache directory creation
    cache_dir = utils.get_cache_dir()
    assert os.path.exists(cache_dir), "Cache directory was not created."
    print(f"    Cache directory verified: {cache_dir}")

    # Test metadata loading (Debug mode)
    df_train_debug = utils.load_metadata("train", debug=True)
    assert not df_train_debug.empty, "Loaded debug training metadata is empty."
    assert "id" in df_train_debug.columns, "Metadata missing 'id' column."
    assert "species" in df_train_debug.columns, "Metadata missing 'species' column."
    assert (
        len(df_train_debug) == 20
    ), f"Debug mode should return 20 rows, got {len(df_train_debug)}"
    print("    Metadata loading verified.")

    # 2. Features Verification
    print("\n[2] Verifying library.features...")

    # Test single image feature extraction
    # Get a valid image path from the loaded metadata
    first_image_rel_path = df_train_debug.iloc[0]["file_path"]
    first_image_full_path = os.path.join("./input", first_image_rel_path)

    if os.path.exists(first_image_full_path):
        geo_feats = features.extract_geometric_features(first_image_full_path)
        expected_keys = {
            "Area",
            "Eccentricity",
            "Solidity",
            "Extent",
            "Aspect_Ratio",
            "Roundness",
            "Mean_Thickness",
        }
        assert isinstance(
            geo_feats, dict
        ), "Feature extraction should return a dictionary."
        assert expected_keys.issubset(geo_feats.keys()), "Missing geometric features."
        assert isinstance(geo_feats["Area"], float), "Area feature should be a float."
        print(f"    Geometric feature extraction verified for {first_image_rel_path}.")
    else:
        print(
            f"    Warning: Image {first_image_full_path} not found. Skipping single image test."
        )

    # Test Dataset Generation (merging metadata + features)
    # We force re-computation (load_cached_data=False) to test the logic
    print("    Generating debug dataset (this involves image processing)...")
    df_dataset = features.get_dataset("train", debug=True, load_cached_data=False)

    # Check if geometric features were merged.
    # The original metadata has ~195 cols. Geometric adds ~7. Total > 200.
    # Exact count depends on metadata, but we check for specific columns.
    assert (
        "Eccentricity" in df_dataset.columns
    ), "Geometric features not merged into dataset."
    assert "margin1" in df_dataset.columns, "Original tabular features missing."
    assert len(df_dataset) == 20, "Dataset size mismatch in debug mode."
    print("    Dataset generation verified.")

    # 3. Pipeline Verification
    print("\n[3] Verifying library.pipeline...")

    # Initialize pipeline in debug mode
    data_pipe = pipeline.DataPipeline(debug=True, seed=42)

    # Run pipeline (Force re-run to verify processing logic)
    # This scales, transforms, and splits the data
    processed_data = data_pipe.run(load_cached_data=False)

    # Verify structure of returned data
    assert "train" in processed_data
    assert "val" in processed_data
    assert "test" in processed_data
    assert "classes" in processed_data

    X_train, y_train, ids_train = processed_data["train"]
    X_val, y_val, ids_val = processed_data["val"]
    class_names = processed_data["classes"]

    # Verify shapes and types
    assert isinstance(X_train, np.ndarray), "X_train must be a numpy array."
    assert X_train.dtype == np.float64, "X_train must be float64."
    assert len(X_train) == 20, "X_train row count mismatch (debug mode)."
    assert len(y_train) == 20, "y_train row count mismatch."
    assert len(ids_train) == 20, "ids_train row count mismatch."

    # Check for NaNs (Pipeline should handle cleaning)
    assert not np.isnan(X_train).any(), "X_train contains NaNs after pipeline."

    print(
        f"    Pipeline verified. Feature matrix shape: {X_train.shape}, Classes: {len(class_names)}"
    )

    # 4. Model Verification
    print("\n[4] Verifying library.model (OASDiscriminant)...")

    clf = model.OASDiscriminant()

    # Fit model on debug data
    # Note: With only 20 samples and many classes, this is numerically unstable usually,
    # but OAS is designed to handle high-dim/low-sample settings via shrinkage.
    clf.fit(X_train, y_train)

    assert clf.W_ is not None, "Model weights (W_) not initialized."
    assert clf.b_ is not None, "Model bias (b_) not initialized."

    # Predict on validation data
    probs = clf.predict_proba(X_val)

    assert probs.shape == (len(X_val), len(class_names)), "Probability shape mismatch."

    # Verify probabilities sum to 1 (approx)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."

    print("    Model training and prediction verified.")

    # 5. End-to-End Execution Verification
    print("\n[5] Verifying End-to-End Execution (train_and_evaluate)...")

    # Define output path
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        os.remove(submission_path)

    # Run the high-level function
    model.train_and_evaluate(debug=True, load_cached_data=True)

    # Verify submission file creation
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content format
    df_sub = pd.read_csv(submission_path)
    assert "id" in df_sub.columns, "Submission missing 'id' column."
    # Check if we have class columns (should be ~99 columns + id)
    assert df_sub.shape[1] > 50, "Submission seems to lack class columns."
    assert len(df_sub) > 0, "Submission file is empty."

    print(
        f"    End-to-end execution successful. Submission generated at {submission_path}"
    )
    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    try:
        run_demo()
    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
