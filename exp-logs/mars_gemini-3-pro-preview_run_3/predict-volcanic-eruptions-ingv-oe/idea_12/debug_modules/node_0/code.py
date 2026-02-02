import os
import sys
import numpy as np
import pandas as pd
import shutil
import glob

# Import from the provided library
import library.config as config
import library.utils as utils
import library.preprocessing as preprocessing
import library.features as features
import library.feature_loader as feature_loader
import library.model_zoo as model_zoo
import library.trainer as trainer


def demo_preprocessing_and_features():
    print("\n=== Demo: Preprocessing and Feature Extraction ===")

    # 1. Identify a sample file
    train_files = glob.glob(os.path.join(config.INPUT_DIR, "train", "*.csv"))
    if not train_files:
        raise FileNotFoundError("No training files found in input directory.")
    sample_file = train_files[0]
    segment_id = int(os.path.splitext(os.path.basename(sample_file))[0])

    print(f"Processing sample file: {sample_file}")

    # 2. Test Preprocessing (Stream A and Stream B generation)
    stream_a, stream_b = preprocessing.preprocess_segment(sample_file)

    # Assertions for Preprocessing
    assert isinstance(stream_a, pd.DataFrame), "Stream A should be a DataFrame"
    assert isinstance(stream_b, pd.DataFrame), "Stream B should be a DataFrame"
    assert stream_a.shape == stream_b.shape, "Streams should have identical shapes"
    assert (
        not stream_a.isnull().values.any()
    ), "Stream A should not have NaNs after imputation"
    # Stream B is smoothed, check if values are different from A (unless signal is flat)
    if not np.allclose(stream_a.values, stream_b.values):
        print("Verified: Stream B (Smoothed) differs from Stream A (Raw).")

    print(f"Stream shapes: {stream_a.shape}")

    # 3. Test Feature Extraction on specific views
    # Take sensor_1 data
    s1_a = stream_a["sensor_1"].values
    s1_b = stream_b["sensor_1"].values

    # View 1: Intensity
    v1 = features.extract_view1_intensity(s1_a)
    assert "raw_max" in v1
    print(f"View 1 (Intensity) features: {list(v1.keys())}")

    # View 2: Kinematics (uses Stream B)
    v2 = features.extract_view2_kinematics(s1_b)
    assert "kin_vel_mean" in v2
    print(f"View 2 (Kinematics) features: {list(v2.keys())}")

    # View 3: Wavelets
    v3 = features.extract_view3_wavelets(s1_a)
    assert "wav_approx_energy" in v3
    print(f"View 3 (Wavelets) features: {list(v3.keys())}")

    # View 5: Temporal
    v5 = features.extract_view5_temporal(s1_a)
    assert f"temp_w{config.N_TEMPORAL_WINDOWS-1}_mean" in v5
    print("View 5 (Temporal) features extracted successfully.")


def demo_feature_matrix_generation():
    print("\n=== Demo: Feature Matrix Generation (Debug Mode) ===")

    # Use a small sample size for speed
    sample_size = 20

    # Build feature matrix for training set
    # load_cached_data=False forces regeneration to test the logic
    df = feature_loader.build_feature_matrix(
        split="train", debug=True, sample_size=sample_size, load_cached_data=False
    )

    # Assertions
    assert len(df) <= sample_size, f"Should have at most {sample_size} rows"
    assert "segment_id" in df.columns
    assert "time_to_eruption" in df.columns
    # Check for feature columns (should be > 2)
    assert df.shape[1] > 10, "Feature matrix should have many columns"

    print(f"Generated Feature Matrix Shape: {df.shape}")
    return df


def demo_model_zoo():
    print("\n=== Demo: Model Zoo Initialization ===")

    # Get Base Models
    models = model_zoo.get_base_models()
    print(f"Base Models Initialized: {list(models.keys())}")

    assert "lgbm" in models
    assert "xgb" in models
    # CatBoost is optional based on availability, but handled in code

    # Get Meta Learner
    meta = model_zoo.get_meta_learner()
    print(f"Meta Learner: {type(meta).__name__}")
    assert hasattr(meta, "fit"), "Meta learner must have a fit method"


def configure_fast_run():
    print("\n=== Configuration: Overriding settings for Fast Demo Run ===")

    # Override global config parameters to ensure the pipeline runs in seconds/minutes
    # instead of hours.

    # Reduce CV Folds
    config.N_FOLDS = 2

    # Reduce LightGBM estimators
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["early_stopping_rounds"] = None

    # Reduce XGBoost estimators
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["early_stopping_rounds"] = None

    # Reduce CatBoost iterations
    config.CATBOOST_PARAMS["iterations"] = 10
    config.CATBOOST_PARAMS["early_stopping_rounds"] = None

    # Reduce Workers to avoid overhead on small data
    config.NUM_WORKERS = 2

    print("Configuration updated: n_estimators=10, n_folds=2")


def demo_full_training_pipeline():
    print("\n=== Demo: Full Stacking Pipeline (Debug Mode) ===")

    # Ensure clean state for output
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        os.remove(submission_path)

    # Run the trainer
    # debug=True limits the data to 100 samples
    # load_cached_data=False ensures we test the feature generation integration
    trainer.run_stacking_cv(debug=True, load_cached_data=False)

    # Validation
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {sub_df.shape}")
    assert "segment_id" in sub_df.columns
    assert "time_to_eruption" in sub_df.columns
    assert len(sub_df) > 0, "Submission file is empty."

    # Check model persistence
    model_path = os.path.join(config.WORKING_DIR, "stacked_models.pkl")
    assert os.path.exists(model_path), "Model file was not saved."
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    utils.seed_everything(42)

    # 1. Verify Low-Level Components
    demo_preprocessing_and_features()

    # 2. Verify Data Loading and Parallelism
    demo_feature_matrix_generation()

    # 3. Verify Model instantiation
    demo_model_zoo()

    # 4. Prepare for Fast Integration Test
    configure_fast_run()

    # 5. Run High-Level Pipeline
    demo_full_training_pipeline()
