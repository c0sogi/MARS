import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import log_loss

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library import config, image_features, data_loader, preprocessing, model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def demo_image_features():
    """
    Demonstrates extracting geometric features from a single image.
    """
    print("\n=== Demo: Image Feature Extraction ===")

    # Load train metadata to find a valid image path
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    sample_row = df_train.iloc[0]

    # Construct full path
    image_rel_path = sample_row["file_path"]
    full_path = os.path.join(config.INPUT_DIR, image_rel_path)

    print(f"Processing image: {image_rel_path}")

    # Extract features
    features = image_features.extract_single_image_features(full_path)
    print(f"Extracted features: {features}")

    # Validation
    assert isinstance(features, dict), "Output must be a dictionary"
    assert "Aspect_Ratio" in features, "Missing Aspect_Ratio feature"
    assert "Solidity" in features, "Missing Solidity feature"
    assert isinstance(features["Aspect_Ratio"], float), "Aspect_Ratio must be a float"
    assert features["Aspect_Ratio"] >= 0, "Aspect_Ratio must be non-negative"
    assert 0 <= features["Solidity"] <= 1.0, "Solidity must be between 0 and 1"

    print("Image feature extraction validated.")


def demo_data_loading_and_preprocessing():
    """
    Demonstrates loading data, augmenting it with image features,
    and applying the high-precision preprocessing pipeline.
    """
    print("\n=== Demo: Data Loading & Preprocessing ===")

    # 1. Load Data
    # We set load_cached_data=False to force the execution of the loading logic
    print("Loading and augmenting data (from scratch)...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, classes = (
        data_loader.load_and_augment_data(load_cached_data=False)
    )

    print(f"Raw Train Data Shape: {X_train_raw.shape}")
    print(f"Number of Classes: {len(classes)}")

    # Validation: Check dimensions
    # Original features (192) + New features (2) = 194
    assert (
        X_train_raw.shape[1] == 194
    ), f"Expected 194 features, got {X_train_raw.shape[1]}"
    assert len(X_train_raw) == len(
        y_train
    ), "Mismatch between X_train and y_train length"
    assert not X_train_raw.isnull().values.any(), "Input data contains NaNs"

    # 2. Preprocessing
    print("Initializing High Precision Pipeline...")
    pipeline = preprocessing.HighPrecisionPipeline()

    # Convert to float64 for precision
    X_train_np = X_train_raw.to_numpy(dtype=np.float64)
    X_val_np = X_val_raw.to_numpy(dtype=np.float64)

    print("Fitting pipeline and transforming data...")
    pipeline.fit(X_train_np)
    X_train_trans = pipeline.transform(X_train_np)
    X_val_trans = pipeline.transform(X_val_np)

    # Validation: Check Statistics
    # After standardization, means should be ~0 and stds ~1
    col_means = np.mean(X_train_trans, axis=0)
    col_stds = np.std(X_train_trans, axis=0)

    print(f"Transformed Data - Max Mean Abs: {np.max(np.abs(col_means)):.6f}")
    print(f"Transformed Data - Mean Std: {np.mean(col_stds):.6f}")

    assert X_train_trans.dtype == np.float64, "Transformed data must be float64"
    assert np.allclose(
        col_means, 0, atol=1e-5
    ), "Feature means should be approximately 0"
    assert np.allclose(col_stds, 1, atol=1e-5), "Feature stds should be approximately 1"

    print("Data loading and preprocessing validated.")
    return X_train_trans, y_train, X_val_trans, y_val, classes


def demo_model_training(X_train, y_train, X_val, y_val, classes):
    """
    Demonstrates training the OAS Linear Discriminant model and evaluating it.
    """
    print("\n=== Demo: Model Training (OAS LDA) ===")

    # Initialize model
    # assume_centered=True because we centered data via StandardScaler in preprocessing
    model_instance = model.OASLinearDiscriminant(assume_centered=True)

    print("Fitting model...")
    model_instance.fit(X_train, y_train)

    print("Predicting on validation set...")
    probs = model_instance.predict_proba(X_val)

    print(f"Probabilities Shape: {probs.shape}")

    # Validation
    assert probs.shape == (
        len(X_val),
        len(classes),
    ), "Probability output shape mismatch"

    # Check that probabilities sum to 1
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities must sum to 1"

    # Calculate Log Loss
    loss = log_loss(y_val, probs, labels=list(range(len(classes))))
    print(f"Validation Log Loss: {loss:.4f}")

    # Sanity check: Loss should be significantly better than random guessing
    # Random guess for 99 classes is approx -ln(1/99) ~= 4.6
    assert (
        loss < 4.0
    ), f"Model performance ({loss}) is poor compared to random guessing (~4.6)"

    print("Model training and evaluation validated.")


def demo_full_pipeline():
    """
    Runs the complete pipeline using the provided orchestrator function.
    """
    print("\n=== Demo: Full Pipeline Execution ===")

    # Run the pipeline provided in model.py
    # This handles loading, preprocessing, training, and submission generation
    model.run_training_pipeline(load_cached_data=True)

    # Verify Submission File
    submission_path = config.SUBMISSION_FILE_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file created at {submission_path}")
    print(f"Submission Shape: {df_sub.shape}")

    # Check structure
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    # 1 ID column + 99 Class columns = 100 columns
    assert df_sub.shape[1] == 100, f"Expected 100 columns, found {df_sub.shape[1]}"

    # Check probability range
    feature_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[feature_cols].values
    assert probs.min() >= 0, "Found negative probabilities in submission"
    assert probs.max() <= 1 + 1e-9, "Found probabilities > 1 in submission"

    print("Full pipeline execution validated.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Feature Extraction
    demo_image_features()

    # 2. Data Loading & Preprocessing
    X_train, y_train, X_val, y_val, classes = demo_data_loading_and_preprocessing()

    # 3. Model Training
    demo_model_training(X_train, y_train, X_val, y_val, classes)

    # 4. Full Pipeline
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
