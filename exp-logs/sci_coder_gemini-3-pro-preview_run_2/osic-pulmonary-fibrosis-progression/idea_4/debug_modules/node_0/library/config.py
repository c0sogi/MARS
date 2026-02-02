import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# --- Configuration ---
class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_4"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Hyperparameters
    SEED = 42
    PCA_COMPONENTS = 30
    QUANTILE = 0.5
    VAL_SIZE = 0.2

    # Model Settings
    IMG_SIZE = 256
    SLICES_PER_PATIENT = 5

    # Feature Config
    STATIC_COLS = [
        "Age",
        "Sex_Male",
        "Smoking_Ex",
        "Smoking_Never",
        "Baseline_FVC",
        "Baseline_Percent",
    ]


# --- Helper Functions ---


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def laplace_log_likelihood(y_true, y_pred, sigma):
    sigma_clipped = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(y_true - y_pred), 1000)
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return np.mean(metric)


# --- Data Processing ---


def load_metadata():
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))
    return train_meta, val_meta, test_meta


def process_tabular_data(df, mode="train"):
    """
    Processes tabular data to generate baseline features and relative weeks.
    Handles the difference between Train (full history) and Test (baseline only provided).
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"tabular_{mode}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 1. Generate Baseline Features if not present
    if "Baseline_FVC" not in df.columns:
        # For train/val, identify the baseline visit (min weeks)
        # We group by Patient and find the row with minimum Weeks
        # Note: This assumes the dataset contains the baseline visit.
        baseline_df = df.loc[df.groupby("Patient")["Weeks"].idxmin()]
        baseline_df = baseline_df[["Patient", "Weeks", "FVC", "Percent"]].rename(
            columns={
                "Weeks": "Baseline_Weeks",
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
            }
        )

        # Merge baseline info back to all rows
        df = df.merge(baseline_df, on="Patient", how="left")

    # 2. Calculate Relative Weeks
    # For Test, Weeks is the target week, Baseline_Weeks is the visit week.
    df["Relative_Weeks"] = df["Weeks"] - df["Baseline_Weeks"]

    # 3. Encode Categoricals
    # Sex: Male/Female -> Sex_Male (1/0)
    df["Sex_Male"] = (df["Sex"] == "Male").astype(int)

    # SmokingStatus: Ex-smoker, Never smoked, Currently smokes
    df["Smoking_Ex"] = (df["SmokingStatus"] == "Ex-smoker").astype(int)
    df["Smoking_Never"] = (df["SmokingStatus"] == "Never smoked").astype(int)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def process_image_features(df, mode="train"):
    """
    Placeholder for Image Feature Extraction.
    Since pydicom is restricted, we generate zero-embeddings to maintain
    pipeline architecture (Visual Backbone + PCA).
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"img_features_{mode}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Generate Zero Embeddings
    # Shape: (N_samples, PCA_COMPONENTS)
    n_samples = len(df)
    feats = np.zeros((n_samples, Config.PCA_COMPONENTS))

    np.save(cache_path, feats)
    return feats


def construct_features(df, img_feats):
    """
    Constructs the final feature sets for the Varying-Coefficient Model.

    Returns:
        X_full: [Static_Features, Image_PCA, Time, Interaction_Terms]
        X_static: [Static_Features, Image_PCA]
    """
    # 1. Static Features
    X_base = df[Config.STATIC_COLS].values

    # 2. Combine with Image Features
    X_static = np.hstack([X_base, img_feats])

    # 3. Time Variable
    t = df["Relative_Weeks"].values.reshape(-1, 1)

    # 4. Interaction Terms (Varying Coefficients)
    # Multiply every static feature by time t
    X_interaction = X_static * t

    # 5. Full Feature Matrix
    X_full = np.hstack([X_static, t, X_interaction])

    return X_full, X_static


# --- Pipeline Execution ---


def run_pipeline():
    set_seed(Config.SEED)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("Loading Metadata...")
    train_meta, val_meta, test_meta = load_metadata()

    # --- Data Preparation ---
    print("Processing Data...")

    # Train
    train_df = process_tabular_data(train_meta, "train")
    train_img = process_image_features(train_meta, "train")
    X_train, X_train_static = construct_features(train_df, train_img)
    y_train = train_df["FVC"].values

    # Val
    val_df = process_tabular_data(val_meta, "val")
    val_img = process_image_features(val_meta, "val")
    X_val, X_val_static = construct_features(val_df, val_img)
    y_val = val_df["FVC"].values

    # Test
    test_df = process_tabular_data(test_meta, "test")
    test_img = process_image_features(test_meta, "test")
    X_test, X_test_static = construct_features(test_df, test_img)

    # Scaling
    # Quantile Regressor (L1) requires scaling for regularization consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Scale Static Features for Uncertainty Model
    scaler_static = StandardScaler()
    X_train_static_scaled = scaler_static.fit_transform(X_train_static)
    X_val_static_scaled = scaler_static.transform(X_val_static)
    X_test_static_scaled = scaler_static.transform(X_test_static)

    # --- Stage 1: FVC Prediction (Quantile Regression) ---
    print("Training FVC Predictor (QuantileRegressor q=0.5)...")
    # Using 'highs' solver if available (sklearn >= 1.1) for speed, else default
    try:
        fvc_model = QuantileRegressor(quantile=0.5, alpha=1.0, solver="highs")
        fvc_model.fit(X_train_scaled, y_train)
    except:
        fvc_model = QuantileRegressor(quantile=0.5, alpha=1.0)
        fvc_model.fit(X_train_scaled, y_train)

    y_pred_train = fvc_model.predict(X_train_scaled)
    y_pred_val = fvc_model.predict(X_val_scaled)

    mae_val = mean_absolute_error(y_val, y_pred_val)
    print(f"FVC Validation MAE: {mae_val:.4f}")

    # --- Stage 2: Uncertainty Prediction (Elastic Net) ---
    print("Training Uncertainty Predictor (ElasticNet)...")
    # Target: Absolute Residuals
    train_residuals = np.abs(y_train - y_pred_train)

    unc_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=Config.SEED)
    unc_model.fit(X_train_static_scaled, train_residuals)

    # Predict MAD
    mad_val = unc_model.predict(X_val_static_scaled)

    # Convert MAD to Sigma (Laplace Scale Parameter)
    # For Laplace: sigma = MAD * sqrt(2)
    sigma_val = mad_val * np.sqrt(2)

    # Validation Score
    val_score = laplace_log_likelihood(y_val, y_pred_val, sigma_val)
    print(f"Validation Metric Score: {val_score:.6f}")

    # --- Inference & Submission ---
    print("Generating Submission...")

    # Predict
    y_pred_test = fvc_model.predict(X_test_scaled)
    mad_test = unc_model.predict(X_test_static_scaled)
    sigma_test = mad_test * np.sqrt(2)

    # Post-processing
    sigma_test = np.maximum(sigma_test, 70)  # Clip confidence

    # Create Submission DataFrame
    sub_df = test_df[["Patient_Week"]].copy()
    sub_df["FVC"] = y_pred_test
    sub_df["Confidence"] = sigma_test

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute Pipeline
run_pipeline()
