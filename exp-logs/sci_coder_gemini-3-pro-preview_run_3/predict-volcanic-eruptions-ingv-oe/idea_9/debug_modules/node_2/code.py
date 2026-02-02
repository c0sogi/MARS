import os
import pandas as pd
import numpy as np
from library.config import Config
from library.feature_extraction import generate_dataset
from library.model_factory import get_base_model
from library.training_pipeline import StackedEnsemblePipeline


def main():
    print("Starting Seismic Eruption Prediction Pipeline Demonstration...")

    # ==========================================
    # 1. Runtime Configuration Override
    # ==========================================
    # Patch Config parameters to ensure the demo runs in seconds rather than hours.
    print("Configuring parameters for rapid execution...")

    # Reduce CV folds to minimum valid number
    Config.N_FOLDS = 2

    # Reduce parallel workers to avoid overhead on small data
    Config.NUM_WORKERS = 2

    # Drastically reduce model iterations for demo purposes
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["verbose"] = -1

    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3

    Config.CAT_PARAMS["iterations"] = 10
    Config.CAT_PARAMS["depth"] = 3

    # Reduce early stopping rounds
    Config.EARLY_STOPPING_ROUNDS = 5

    # ==========================================
    # 2. Demonstrate Feature Extraction
    # ==========================================
    print("\n--- Demonstrating Feature Extraction ---")
    # Generate features for a tiny subset of the training data
    # We use a custom name and force load_cached_data=False to execute the extraction logic
    demo_feat_name = "demo_train_features"
    demo_size = 5

    print(f"Extracting features for {demo_size} segments...")
    df_features = generate_dataset(
        meta_path=Config.TRAIN_META_PATH,
        output_name=demo_feat_name,
        load_cached_data=False,
        debug_size=demo_size,
    )

    # Validation
    assert not df_features.empty, "Feature extraction returned empty DataFrame"
    assert (
        len(df_features) == demo_size
    ), f"Expected {demo_size} rows, got {len(df_features)}"
    assert "segment_id" in df_features.columns, "segment_id column missing"
    assert "time_to_eruption" in df_features.columns, "Target column missing"

    # Check for presence of engineered features (e.g., from kinematics or spectral)
    # Based on feature_extraction.py, columns like 'sensor_1_raw_mean' should exist
    cols = df_features.columns
    assert any("mean" in c for c in cols), "Statistical features (mean) missing"
    assert any("spec" in c for c in cols), "Spectral features missing"

    print(f"Feature extraction successful. Shape: {df_features.shape}")

    # ==========================================
    # 3. Demonstrate Model Factory
    # ==========================================
    print("\n--- Demonstrating Model Factory ---")
    # Verify we can instantiate all model types
    models = {}
    for name in ["lgbm", "xgb", "cat"]:
        model = get_base_model(name)
        models[name] = model
        print(f"Instantiated {name}: {type(model).__name__}")
        assert model is not None

    # ==========================================
    # 4. Demonstrate Full Pipeline Execution
    # ==========================================
    print("\n--- Demonstrating Full Stacked Ensemble Pipeline ---")
    pipeline = StackedEnsemblePipeline()

    # Run the full pipeline with a debug limit
    # This will:
    # 1. Load/Generate features for Train/Val/Test (limited to debug_size)
    # 2. Run Stratified K-Fold CV to train base models and generate OOF preds
    # 3. Train the Ridge Meta-Learner
    # 4. Retrain base models on full data
    # 5. Predict on Test set
    # 6. Save submission
    pipeline_debug_size = 10

    print(f"Running pipeline with debug_size={pipeline_debug_size}...")
    pipeline.run(debug_size=pipeline_debug_size, load_cached_data=False)

    # ==========================================
    # 5. Validate Submission
    # ==========================================
    print("\n--- Validating Submission Output ---")
    submission_path = Config.SUBMISSION_PATH

    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission file loaded from {submission_path}")
        print(sub_df.head())

        # Validate structure
        assert list(sub_df.columns) == [
            "segment_id",
            "time_to_eruption",
        ], f"Incorrect columns: {sub_df.columns}"

        # Validate size
        # Since we used debug_size=10 in pipeline.run(), the test set loaded was also limited to 10
        assert (
            len(sub_df) == pipeline_debug_size
        ), f"Expected {pipeline_debug_size} predictions, found {len(sub_df)}"

        # Validate content
        assert not sub_df.isnull().values.any(), "Submission contains NaN values"
        assert (
            sub_df["segment_id"].dtype == "int64"
            or sub_df["segment_id"].dtype == "int32"
        ), "segment_id should be integer"

        print("Validation Passed: Submission file is correctly formatted.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
