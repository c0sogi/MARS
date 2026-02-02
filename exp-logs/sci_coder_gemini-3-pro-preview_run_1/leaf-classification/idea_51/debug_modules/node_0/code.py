import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import log_loss

# Import from the provided library files
from library.config import (
    SEED,
    INPUT_DIR,
    SUBMISSION_PATH,
    IMAGE_DERIVED_FEATURES,
    FLOAT_PRECISION,
)
from library.image_features import extract_image_features
from library.data_manager import get_train_data
from library.preprocessor import HighPrecisionPipeline, get_transformed_data
from library.oas_model import CustomOASDiscriminant, run_oas_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(SEED)


def demo_image_feature_extraction():
    print("\n--- Demo: Image Feature Extraction ---")
    # Find a valid image file to test
    images_dir = os.path.join(INPUT_DIR, "images")
    sample_image = None
    for f in os.listdir(images_dir):
        if f.endswith(".jpg"):
            sample_image = f
            break

    if sample_image:
        # The function expects a relative path like 'images/1.jpg'
        rel_path = os.path.join("images", sample_image)
        print(f"Extracting features from: {rel_path}")

        features = extract_image_features(rel_path)

        # Validation
        assert isinstance(features, dict), "Output must be a dictionary"

        # Check if all expected keys are present
        missing_keys = [k for k in IMAGE_DERIVED_FEATURES if k not in features]
        assert not missing_keys, f"Missing feature keys: {missing_keys}"

        # Check value types
        for k, v in features.items():
            assert isinstance(v, float), f"Feature {k} should be a float, got {type(v)}"

        print("Successfully extracted features:")
        print(f"  Area: {features['area']:.2f}")
        print(f"  Thickness Ratio: {features['thickness_ratio']:.4f}")
    else:
        print("No images found to test extraction.")


def demo_data_manager():
    print("\n--- Demo: Data Manager ---")
    # Load training data (forcing re-computation to test logic, ignoring cache)
    print("Loading training data (ignoring cache)...")
    X_df, y, ids = get_train_data(load_cached_data=False)

    # Validation
    assert isinstance(X_df, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert len(X_df) == len(y), "X and y must have same length"
    assert len(X_df) == len(ids), "X and ids must have same length"

    # Check for NaNs
    assert not X_df.isnull().values.any(), "Feature matrix contains NaNs"

    print(f"Loaded {len(X_df)} samples with {X_df.shape[1]} features.")
    print(f"Sample Species: {y[0]}")


def demo_preprocessor_and_model():
    print("\n--- Demo: Preprocessor and OAS Model ---")

    # 1. Get Transformed Data
    # We use the provided function which handles fitting on train and transforming splits
    print("Getting transformed data...")
    X_train, y_train, X_val, y_val, X_test, ids_test = get_transformed_data(
        load_cached_data=False
    )

    # Verify Precision
    assert X_train.dtype == FLOAT_PRECISION, "X_train must be float64"
    assert X_val.dtype == FLOAT_PRECISION, "X_val must be float64"

    # 2. Encode Labels for Model
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # 3. Instantiate and Fit Model
    print("Initializing CustomOASDiscriminant...")
    model = CustomOASDiscriminant()

    print("Fitting model...")
    model.fit(X_train, y_train_enc)

    # Validate Model Attributes
    assert model.W_ is not None, "Weights W_ should be computed"
    assert model.b_ is not None, "Biases b_ should be computed"
    assert model.precision_ is not None, "Precision matrix should be computed"

    # 4. Predict
    print("Predicting on validation set...")
    probs = model.predict_proba(X_val)

    # Validation of Probabilities
    assert probs.shape == (len(X_val), len(le.classes_)), "Probability shape mismatch"

    # Check Sum to 1 (approximate due to float precision)
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Check Bounds
    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Probabilities out of [0, 1] range"

    # Calculate Loss
    loss = log_loss(y_val_enc, probs)
    print(f"Validation Log Loss: {loss:.4f}")


def demo_full_pipeline():
    print("\n--- Demo: Full Pipeline Execution ---")
    # Run the orchestrated pipeline provided in oas_model.py
    # This generates the submission file
    run_oas_pipeline(load_cached_data=True)

    # Verify Submission
    if os.path.exists(SUBMISSION_PATH):
        print(f"Submission file found at {SUBMISSION_PATH}")
        df_sub = pd.read_csv(SUBMISSION_PATH)

        # Check columns
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        assert len(df_sub) > 0, "Submission file is empty"

        # Check values
        # All columns except 'id' should be numeric probabilities
        prob_cols = [c for c in df_sub.columns if c != "id"]
        probs = df_sub[prob_cols].values

        # Basic sanity check on probabilities
        assert np.all(probs >= 0), "Negative probabilities found"
        # Note: The provided metric description says probabilities are rescaled,
        # but our model output is softmax, so it should be normalized already.

        print("Submission format verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    print("Starting Demonstration Script...")

    try:
        demo_image_feature_extraction()
        demo_data_manager()
        demo_preprocessor_and_model()
        demo_full_pipeline()
        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nERROR OCCURRED: {e}")
        exit(1)
