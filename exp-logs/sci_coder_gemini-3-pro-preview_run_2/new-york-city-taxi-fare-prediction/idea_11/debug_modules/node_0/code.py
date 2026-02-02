import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import library modules
from library.config import Config
from library.utils import haversine_distance, manhattan_distance, clamp_coordinates
from library.data_factory import DataFactory
from library.global_knowledge import KnowledgeBase
from library.feature_engine import MarginCalculator, FeatureEngineer
from library.model_trainer import ResidualXGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating small datasets
    and overriding Config parameters to point to them.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = "./working/demo_run"
    data_dir = os.path.join(demo_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Create Mock Data
    # We use the metadata/test.parquet as a template because it's small (~10k rows)
    # We will add a dummy 'fare_amount' to simulate training/val data.
    print("Creating mock datasets from test template...")
    test_df = pd.read_parquet("./metadata/test.parquet")

    # Create dummy targets for training simulation
    # Fare = $2.50 + $1.50 * distance_approx + noise
    # Just a rough synthetic target to ensure model can learn something
    np.random.seed(42)
    dist_approx = (
        np.sqrt(
            (test_df["pickup_latitude"] - test_df["dropoff_latitude"]) ** 2
            + (test_df["pickup_longitude"] - test_df["dropoff_longitude"]) ** 2
        )
        * 100
    )  # Rough scale

    test_df["fare_amount"] = (
        2.5 + 1.5 * dist_approx + np.random.normal(0, 2, len(test_df))
    )
    test_df["fare_amount"] = test_df["fare_amount"].clip(lower=2.5)

    # Split into train/val/test
    train_df = test_df.iloc[:5000].copy()
    val_df = test_df.iloc[5000:7000].copy()
    test_small_df = test_df.iloc[7000:].drop(columns=["fare_amount"]).copy()

    # Save to demo directory
    train_path = os.path.join(data_dir, "train_small.parquet")
    val_path = os.path.join(data_dir, "val_small.parquet")
    test_path = os.path.join(data_dir, "test_small.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_small_df.to_parquet(test_path, index=False)

    # 2. Override Config
    print("Overriding Config parameters for speed...")
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir

    # Point input paths to our small mock files
    Config.TRAIN_DATA_PATH = train_path
    Config.VAL_DATA_PATH = val_path
    Config.TEST_DATA_PATH = test_path
    Config.SAMPLE_SUBMISSION_PATH = os.path.join(
        demo_dir, "sample_submission.csv"
    )  # Not strictly used in code logic

    # Update Output/Cache paths to be inside the demo working dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.GLOBAL_STATS_CACHE_PATH = os.path.join(demo_dir, "global_stats.parquet")
    Config.PROCESSED_TRAIN_CACHE_PATH = os.path.join(
        demo_dir, "train_processed.parquet"
    )
    Config.PROCESSED_VAL_CACHE_PATH = os.path.join(demo_dir, "val_processed.parquet")
    Config.PROCESSED_TEST_CACHE_PATH = os.path.join(demo_dir, "test_processed.parquet")

    # Reduce computational load
    Config.TRAIN_SUBSAMPLE_SIZE = 5000  # Use all of our small mock train set
    Config.NUM_BOOST_ROUNDS = 10  # Very few rounds for demo
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = 0  # Silent

    # Ensure directories exist
    Config.setup()

    return train_df, val_df, test_small_df


def verify_utils():
    print("\n=== Verifying Library Utils ===")

    # 1. Haversine Distance
    # Dist between (0,0) and (0,1) deg is approx 111.19 km
    d = haversine_distance(0, 0, 0, 1)
    print(f"Haversine (0,0)->(0,1): {d:.4f} km")
    assert np.isclose(d, 111.195, atol=0.1), "Haversine calculation incorrect"

    # 2. Clamp Coordinates
    df_bad = pd.DataFrame(
        {
            "pickup_latitude": [50.0],  # Out of bounds (Max 42.0)
            "pickup_longitude": [-80.0],  # Out of bounds (Min -75.0)
        }
    )
    df_clamped = clamp_coordinates(df_bad)
    print(f"Clamped Lat: {df_clamped['pickup_latitude'].iloc[0]}")
    print(f"Clamped Lon: {df_clamped['pickup_longitude'].iloc[0]}")

    assert (
        df_clamped["pickup_latitude"].iloc[0] == Config.LAT_MAX
    ), "Latitude clamping failed"
    assert (
        df_clamped["pickup_longitude"].iloc[0] == Config.LON_MIN
    ), "Longitude clamping failed"
    print("Utils verification passed.")


def verify_data_factory():
    print("\n=== Verifying Data Factory ===")

    # Load Train Data (should use our mock file)
    # Force reload (load_cached_data=False) to test processing logic
    df = DataFactory.load_train_data(load_cached_data=False)

    print(f"Loaded training data shape: {df.shape}")
    assert len(df) <= Config.TRAIN_SUBSAMPLE_SIZE, "Subsampling logic failed"
    assert "fare_amount" in df.columns, "fare_amount missing from training data"

    # Verify physics filter (mock data shouldn't have extreme outliers, but function should run)
    # We'll just check it didn't crash and returned a dataframe
    assert isinstance(df, pd.DataFrame)
    print("Data Factory verification passed.")


def verify_global_knowledge():
    print("\n=== Verifying Global Knowledge Base ===")

    kb = KnowledgeBase()
    # Force build from scratch
    fine, coarse, rate = kb.build(load_cached_data=False)

    print(f"Global Rate: {rate:.4f} $/km")
    print(f"Fine Grid Stats Shape: {fine.shape}")
    print(f"Coarse Grid Stats Shape: {coarse.shape}")

    assert rate > 0, "Global rate should be positive"
    assert not fine.empty, "Fine stats should not be empty"
    assert not coarse.empty, "Coarse stats should not be empty"

    # Check columns
    assert "fare_sum" in fine.columns and "fare_count" in fine.columns
    print("Global Knowledge Base verification passed.")

    return fine, coarse, rate


def verify_feature_engine(fine_stats, coarse_stats, global_rate):
    print("\n=== Verifying Feature Engine ===")

    # Setup
    margin_calc = MarginCalculator(fine_stats, coarse_stats, global_rate)
    fe = FeatureEngineer(margin_calc)

    # Create a dummy row for testing
    # Use a coordinate that likely exists in the stats (from the mock data generation)
    # We take a row from the training cache
    df_sample = pd.read_parquet(Config.PROCESSED_TRAIN_CACHE_PATH).head(5).copy()

    # Process
    df_feat = fe.process(df_sample, is_training=True)

    print("Generated Features:", df_feat.columns.tolist())

    # Checks
    required_feats = ["base_margin", "residual", "dist_haversine", "pickup_rot_1"]
    for f in required_feats:
        assert f in df_feat.columns, f"Feature {f} missing"

    # Verify Residual Logic: Residual = Fare - Base_Margin
    # Note: floating point arithmetic
    diff = (df_feat["residual"] + df_feat["base_margin"]) - df_feat["fare_amount"]
    max_diff = np.max(np.abs(diff))
    print(f"Max reconstruction error (Residual + Margin - Fare): {max_diff:.6f}")
    assert max_diff < 1e-5, "Residual calculation logic incorrect"

    print("Feature Engine verification passed.")


def verify_model_training():
    print("\n=== Verifying Model Training & Submission ===")

    trainer = ResidualXGBRegressor()

    # Run the full pipeline
    # This calls train() and generate_submission()
    # We use load_cached_data=True where possible to use the files generated in previous steps
    trainer.run(load_cached_data=True)

    # Check Model
    assert trainer.model is not None, "Model was not trained"
    print(f"Model Best Score (RMSE): {trainer.model.best_score:.4f}")

    # Check Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    print("First 5 predictions:")
    print(sub_df.head())

    # Validate format
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"
    assert len(sub_df) > 0, "Submission is empty"

    # Check for reasonable values (min fare prediction logic)
    assert (
        sub_df["fare_amount"].min() >= Config.MIN_FARE_PREDICTION
    ), "Predictions below minimum fare"

    print("Model Training and Submission verification passed.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Setup Environment
    setup_demo_environment()

    # 2. Verify Utils
    verify_utils()

    # 3. Verify Data Loading
    verify_data_factory()

    # 4. Verify Global Knowledge (and get stats for next step)
    fine, coarse, rate = verify_global_knowledge()

    # 5. Verify Feature Engineering
    verify_feature_engine(fine, coarse, rate)

    # 6. Verify Model Training
    verify_model_training()

    print("\nAll demonstrations and verifications completed successfully.")
