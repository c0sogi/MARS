import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.feature_extraction import extract_features
from library.data_manager import (
    load_and_merge_data,
    create_interaction_features,
    get_static_features,
)
from library.modeling import FVCPredictor, UncertaintyPredictor


def prepare_dataset(mode, load_cached_data=True):
    """
    Prepares the dataset for a specific mode (train/val/test).
    Loads metadata, extracts image features, merges them, and constructs
    specific feature sets for the two models.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        dict: Contains 'X_full' (for FVC), 'X_static' (for Uncertainty),
              'y' (targets), and 'patient_weeks' (IDs).
    """
    # 1. Load Metadata to get patient lists
    meta_path = os.path.join(Config.METADATA_DIR, f"{mode}_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # 2. Extract Image Features (Visual Backbone + PCA)
    # The library function handles caching of the raw image features
    img_feats = extract_features(df_meta, mode=mode, load_cached_data=load_cached_data)

    # 3. Merge Tabular + Image Features
    # The library function handles caching of the merged dataset
    data_container = load_and_merge_data(
        mode=mode, image_features=img_feats, load_cached_data=load_cached_data
    )

    # 4. Construct Model-Specific Feature Sets
    # Extract components from the container
    X_static_base = get_static_features(data_container)
    weeks = data_container["weeks"]
    y = data_container["y"]
    patient_weeks = data_container["patient_weeks"]

    # FVC Predictor uses Varying-Coefficient model: [Static, Time, Static*Time]
    X_full = create_interaction_features(X_static_base, weeks)

    # Uncertainty Predictor uses only Static features (Parsimony principle)
    X_static = X_static_base

    return {
        "X_full": X_full,
        "X_static": X_static,
        "y": y,
        "patient_weeks": patient_weeks,
    }


def train_stage_1(X_train, y_train, X_val, y_val):
    """
    Stage 1: Train the FVC Predictor (Quantile Regression).
    Objective: Minimize L1 (MAE) to predict the Median (50th percentile).

    Args:
        X_train (np.array): Training features (Full).
        y_train (np.array): Training targets.
        X_val (np.array): Validation features (Full).
        y_val (np.array): Validation targets.

    Returns:
        tuple: (Trained Model, Train Predictions, Val Predictions)
    """
    print("--- Stage 1: Training FVC Predictor (Quantile Regression) ---")

    # Initialize model with quantile=0.5 for Median regression
    model = FVCPredictor(quantile=Config.QUANTILE, alpha=0.5)

    # Fit on training data
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # Evaluate using MAE (L1 Loss)
    mae_train = np.mean(np.abs(y_train - y_pred_train))
    mae_val = np.mean(np.abs(y_val - y_pred_val))

    print(f"Stage 1 Results - Train MAE: {mae_train}")
    print(f"Stage 1 Results - Val MAE: {mae_val}")

    return model, y_pred_train, y_pred_val


def train_stage_2(X_train, y_train, y_pred_train, X_val, y_val, y_pred_val):
    """
    Stage 2: Train the Uncertainty Predictor (Elastic Net).
    Objective: Predict the Mean Absolute Deviation (MAD) of the residuals.

    Args:
        X_train (np.array): Training features (Static only).
        y_train (np.array): True FVC values.
        y_pred_train (np.array): Predicted FVC values from Stage 1.
        X_val (np.array): Validation features (Static only).
        y_val (np.array): True FVC values.
        y_pred_val (np.array): Predicted FVC values from Stage 1.

    Returns:
        object: Trained Uncertainty Model.
    """
    print("--- Stage 2: Training Uncertainty Predictor (ElasticNet) ---")

    # Calculate Residuals (Absolute Error)
    # We model the error magnitude to estimate uncertainty
    train_residuals = np.abs(y_train - y_pred_train)

    # Initialize model
    # Using ElasticNet to prevent overfitting on noise
    model = UncertaintyPredictor(alpha=0.1, l1_ratio=0.5, seed=Config.SEED)

    # Fit on static features to predict residuals
    model.fit(X_train, train_residuals)

    # Predict MAD on Validation
    mad_val = model.predict(X_val)

    # Convert MAD to Sigma (Laplace Scale Parameter)
    # For Laplace distribution: sigma = MAD * sqrt(2)
    sigma_val = mad_val * np.sqrt(2)

    # Evaluate with the official Competition Metric
    score = laplace_log_likelihood_metric(y_val, y_pred_val, sigma_val)
    print(f"Stage 2 Results - Validation Laplace Log Likelihood: {score}")

    return model


def inference(fvc_model, unc_model, X_test_full, X_test_static, test_ids):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        fvc_model: Trained FVC Predictor.
        unc_model: Trained Uncertainty Predictor.
        X_test_full (np.array): Test features for FVC (Full).
        X_test_static (np.array): Test features for Uncertainty (Static).
        test_ids (np.array): Patient_Week identifiers.
    """
    print("--- Inference & Submission ---")

    # 1. Predict Median FVC
    y_pred = fvc_model.predict(X_test_full)

    # 2. Predict Uncertainty (MAD)
    mad = unc_model.predict(X_test_static)

    # 3. Convert to Sigma
    sigma = mad * np.sqrt(2)

    # 4. Clip Confidence
    # Metric requires sigma >= 70
    sigma = np.maximum(sigma, 70)

    # 5. Create DataFrame
    submission = pd.DataFrame(
        {"Patient_Week": test_ids, "FVC": y_pred, "Confidence": sigma}
    )

    # 6. Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")


def run_workflow(load_cached_data=True):
    """
    Orchestrates the end-to-end pipeline:
    1. Data Preparation (Load, Extract, Merge)
    2. Stage 1 Training (FVC Prediction)
    3. Stage 2 Training (Uncertainty Prediction)
    4. Inference on Test Set
    """
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Starting Workflow (Caching: {load_cached_data})...")

    # --- 1. Data Preparation ---
    print("\n[Data Preparation]")
    print("Processing Training Data...")
    train_data = prepare_dataset("train", load_cached_data)

    print("Processing Validation Data...")
    val_data = prepare_dataset("val", load_cached_data)

    print("Processing Test Data...")
    test_data = prepare_dataset("test", load_cached_data)

    # --- 2. Stage 1: FVC Prediction ---
    print("\n[Model Training Stage 1]")
    # FVC model uses the full feature set (Static + Time + Interactions)
    fvc_model, y_pred_train, y_pred_val = train_stage_1(
        train_data["X_full"], train_data["y"], val_data["X_full"], val_data["y"]
    )

    # --- 3. Stage 2: Uncertainty Prediction ---
    print("\n[Model Training Stage 2]")
    # Uncertainty model uses only static features to avoid overfitting
    unc_model = train_stage_2(
        train_data["X_static"],
        train_data["y"],
        y_pred_train,
        val_data["X_static"],
        val_data["y"],
        y_pred_val,
    )

    # --- 4. Inference ---
    print("\n[Inference]")
    inference(
        fvc_model,
        unc_model,
        test_data["X_full"],
        test_data["X_static"],
        test_data["patient_weeks"],
    )

    print("\nWorkflow Completed Successfully.")
