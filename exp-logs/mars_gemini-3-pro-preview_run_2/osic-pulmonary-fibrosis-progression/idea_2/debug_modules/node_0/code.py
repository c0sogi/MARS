import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings and verbose output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library components
from library.config import Config
from library.feature_extractor import generate_features
from library.data_processor import DataProcessor
from library.model import DualElasticNet, generate_submission
from library.metrics import laplace_log_likelihood


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing Demonstration Script...")

    # 1. Configuration Overrides for Speed and Debugging
    # We enable DEBUG to use a tiny subset of patients (5 per split).
    # We reduce PCA components to 2 because we only have 5 samples in debug mode.
    Config.DEBUG = True
    Config.N_PCA_COMPONENTS = 2

    # Set seeds
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # Step 1: Feature Extraction
    # ---------------------------------------------------------
    print("\n[Step 1] Extracting Features (CNN + PCA)...")

    # generate_features handles CNN inference and PCA.
    # We pass load_cached_data=False to force execution of the logic.
    train_feats, val_feats, test_feats = generate_features(
        load_cached_data=False, debug=Config.DEBUG
    )

    # Verification
    print("Verifying Feature Extraction...")
    assert len(train_feats) > 0, "Training features dictionary is empty."
    assert len(val_feats) > 0, "Validation features dictionary is empty."

    # Check dimensionality of a sample feature vector
    sample_pid = list(train_feats.keys())[0]
    sample_vec = train_feats[sample_pid]
    assert sample_vec.shape == (
        Config.N_PCA_COMPONENTS,
    ), f"Expected feature shape ({Config.N_PCA_COMPONENTS},), got {sample_vec.shape}"

    print("Feature Extraction Verified.")

    # ---------------------------------------------------------
    # Step 2: Data Processing
    # ---------------------------------------------------------
    print("\n[Step 2] Processing Tabular Data and Interactions...")

    processor = DataProcessor()

    # process_data merges image features with tabular metadata
    # and creates the interaction terms [Static, Time, Static*Time]
    (X_train, y_train), (X_val, y_val), (X_test, df_test) = processor.process_data(
        train_feats, val_feats, test_feats, load_cached_data=False
    )

    # Verification
    print("Verifying Data Processing...")

    # Check basic shapes
    assert X_train.ndim == 2, "X_train must be 2D."
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length."

    # Calculate expected column count
    # Static features:
    #   Tabular: Age, Baseline_FVC, Baseline_Percent (3 numerical) + Sex(2), Smoking(3) (one-hot) = ~8 cols
    #   Image: N_PCA_COMPONENTS (2)
    #   Total Static ~ 10
    # Full Matrix: Static + Time(1) + Interactions(Static*1) = 2*Static + 1

    n_static_features = (X_train.shape[1] - 1) // 2
    assert (
        n_static_features > 0
    ), "Feature matrix construction failed (no static features found)."

    # Check that interaction terms exist (columns > static features)
    assert (
        X_train.shape[1] > n_static_features + 1
    ), "Interaction terms missing from feature matrix."

    print(f"Data Shapes Validated: X_train {X_train.shape}, y_train {y_train.shape}")

    # ---------------------------------------------------------
    # Step 3: Model Training
    # ---------------------------------------------------------
    print("\n[Step 3] Training Dual Elastic Net Model...")

    model = DualElasticNet()

    # Fit the model
    model.fit(X_train, y_train, X_val, y_val)

    # Verification
    print("Verifying Model State...")
    # Check if underlying sklearn models are fitted by accessing attributes set during fit
    assert hasattr(model.fvc_model, "coef_"), "Primary FVC model not fitted."
    assert hasattr(model.sigma_model, "coef_"), "Secondary Sigma model not fitted."

    print("Model Training Verified.")

    # ---------------------------------------------------------
    # Step 4: Inference and Metrics
    # ---------------------------------------------------------
    print("\n[Step 4] Running Inference and Evaluation...")

    # Predict on validation set
    val_fvc_pred, val_sigma_pred = model.predict(X_val)

    # Calculate Metric
    score = laplace_log_likelihood(y_val, val_fvc_pred, val_sigma_pred)

    # Verification
    assert len(val_fvc_pred) == len(y_val), "Prediction length mismatch."
    assert len(val_sigma_pred) == len(y_val), "Uncertainty prediction length mismatch."
    assert np.isfinite(
        score
    ), f"Metric calculation resulted in non-finite value: {score}"

    print(f"Validation Score: {score:.4f}")

    # ---------------------------------------------------------
    # Step 5: Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 5] Generating Submission File...")

    generate_submission(model, X_test, df_test)

    # Verification
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in required_cols
    ), "Submission missing required columns."
    assert len(sub_df) == len(df_test), "Submission row count mismatch."

    print("Submission Generation Verified.")
    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
