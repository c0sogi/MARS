import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library.config import Config
from library.feature_engineering import FeatureProcessor
from library.model import PhysicsInformedLinearModel
from library.utils import calculate_rmse, format_submission


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    print("Initializing configuration and environment...")
    Config.setup()

    # Set seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # ---------------------------------------------------------
    # 2. Data Loading (Optimized for Speed)
    # ---------------------------------------------------------
    print("\nLoading data...")
    # We read the full parquet files. With 220GB RAM, this is safe.
    # We then sample a subset to ensure the feature engineering and training
    # steps complete in seconds rather than minutes/hours for this demo.

    train_full = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_full = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # Downsample for demonstration speed
    SAMPLE_SIZE_TRAIN = 100_000
    SAMPLE_SIZE_VAL = 20_000

    print(f"Sampling {SAMPLE_SIZE_TRAIN} rows from training data...")
    train_subset = train_full.sample(
        n=SAMPLE_SIZE_TRAIN, random_state=Config.RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Sampling {SAMPLE_SIZE_VAL} rows from validation data...")
    val_subset = val_full.sample(
        n=SAMPLE_SIZE_VAL, random_state=Config.RANDOM_SEED
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\nRunning Feature Engineering...")
    processor = FeatureProcessor()

    # fit_transform on training data
    # We set load_cached_data=False to ensure the code logic runs from scratch
    print("Processing training subset...")
    X_train, y_train = processor.fit_transform(train_subset, load_cached_data=False)

    # transform on validation and test data
    # Note: transform uses the scaler fitted on the training data
    print("Processing validation subset...")
    X_val = processor.transform(
        val_subset, load_cached_data=False, cache_name="val_demo"
    )

    print("Processing test set...")
    X_test = processor.transform(
        test_df, load_cached_data=False, cache_name="test_demo"
    )

    # Verify shapes
    # X_train rows might be fewer than SAMPLE_SIZE_TRAIN due to cleaning/filtering in fit_transform
    assert X_train.shape[0] <= SAMPLE_SIZE_TRAIN
    assert X_train.shape[1] == 10  # We expect 10 engineered features
    assert len(y_train) == X_train.shape[0]

    # Validation/Test transform does not filter rows
    assert X_val.shape[0] == SAMPLE_SIZE_VAL
    assert X_test.shape[0] == len(test_df)

    print("Feature engineering shapes verified.")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\nInitializing and Training Model...")
    model = PhysicsInformedLinearModel()

    # Optimize hyperparameters for this quick demonstration
    # We reduce max_iter to ensure the training loop finishes instantly
    model.params["max_iter"] = 5
    model.batch_size = 2048

    # Train the model
    # We pass the validation set to enable manual early stopping logic within the class
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=val_subset["fare_amount"].values,
        patience=2,
    )

    # ---------------------------------------------------------
    # 5. Model Persistence (Save/Load)
    # ---------------------------------------------------------
    print("\nTesting Model Persistence...")
    model_save_path = os.path.join(Config.WORKING_DIR, "demo_model.npz")
    model.save(model_save_path)

    # Create a new instance and load weights to verify persistence works
    loaded_model = PhysicsInformedLinearModel()
    loaded_model.load(model_save_path)

    # Verify weights match
    assert np.allclose(model.model.coef_, loaded_model.model.coef_)
    assert np.isclose(model.model.intercept_, loaded_model.model.intercept_)
    print("Model saved and loaded successfully.")

    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------
    print("\nEvaluating Model...")
    # Predict using the loaded model
    val_preds = loaded_model.predict(X_val)

    # Calculate RMSE
    rmse = calculate_rmse(val_subset["fare_amount"].values, val_preds)
    print(f"Validation RMSE: {rmse:.4f}")

    # Basic sanity check: RMSE should be a finite positive number
    # Given the mean fare is ~11 and std ~20, an untrained/simple model might have RMSE ~10-20
    assert np.isfinite(rmse)
    assert rmse > 0

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\nGenerating Submission...")
    test_preds = loaded_model.predict(X_test)

    # Verify predictions adhere to constraints (min fare)
    assert np.all(test_preds >= Config.MIN_FARE)

    # Format and save submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    format_submission(test_df["key"], test_preds, output_path=submission_path)

    # Verify file existence and format
    if os.path.exists(submission_path):
        print(f"Submission file created at: {submission_path}")

        # Read back to verify format
        sub_df = pd.read_csv(submission_path)
        print(f"Submission shape: {sub_df.shape}")

        assert sub_df.shape == (len(test_df), 2)
        assert list(sub_df.columns) == ["key", "fare_amount"]
        assert sub_df["fare_amount"].notna().all()
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
