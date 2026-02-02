import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.covariance import OAS
from scipy.special import softmax
from sklearn.metrics import log_loss


# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
class Config:
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    CACHE_DIR = "./working/idea_52"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    SEED = 42
    FLOAT_PRECISION = np.float64

    # Feature columns provided in the dataset
    TABULAR_PREFIXES = ["margin", "shape", "texture"]
    NUM_FEATURES_PER_GROUP = 64


# Set seeds for reproducibility
np.random.seed(Config.SEED)


# ------------------------------------------------------------------------------
# Image Processing & Feature Extraction
# ------------------------------------------------------------------------------
def extract_geometric_features(image_path):
    """
    Extracts geometric and integral morphological features from a leaf image.
    Implements the Integral-Morphological strategy:
    - Polarity correction
    - Lossless contours
    - Integral internal profiling (Mean Thickness)
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Return zeros if image load fails (should not happen given metadata validation)
        return np.zeros(10, dtype=Config.FLOAT_PRECISION)

    # Invert polarity: Leaf should be White (255), Background Black (0)
    # The dataset description says "binary black leaves against white backgrounds"
    # We use THRESH_BINARY_INV to get the leaf as the foreground.
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find Contours: Use CHAIN_APPROX_NONE for lossless boundary fidelity
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros(10, dtype=Config.FLOAT_PRECISION)

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Basic Geometry
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter == 0:
        perimeter = 1e-6

    # 2. Ellipse Fit (Major/Minor Axis)
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            major_axis = max(ma, MA)
            minor_axis = min(ma, MA)
        except:
            major_axis = 0.0
            minor_axis = 0.0
    else:
        # Fallback for very small contours
        rect = cv2.minAreaRect(cnt)
        major_axis = max(rect[1])
        minor_axis = min(rect[1])

    # 3. Shape Descriptors (Ratios)
    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Aspect Ratio
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Eccentricity
    if major_axis > 0:
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    else:
        eccentricity = 0.0

    # Roundness
    roundness = (4 * np.pi * area) / (perimeter**2)

    # 4. Integral Internal Structure (Mean Thickness)
    # Euclidean Distance Transform on binary image
    dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    # Mean of non-zero distances (integral measure of "fleshiness")
    leaf_pixels = dist_transform[bin_img > 0]
    if len(leaf_pixels) > 0:
        mean_thickness = np.mean(leaf_pixels)
    else:
        mean_thickness = 0.0

    features = np.array(
        [
            area,
            perimeter,
            major_axis,
            minor_axis,
            solidity,
            extent,
            aspect_ratio,
            eccentricity,
            roundness,
            mean_thickness,
        ],
        dtype=Config.FLOAT_PRECISION,
    )

    return features


FEATURE_NAMES_GEO = [
    "area",
    "perimeter",
    "major_axis",
    "minor_axis",
    "solidity",
    "extent",
    "aspect_ratio",
    "eccentricity",
    "roundness",
    "mean_thickness",
]


# ------------------------------------------------------------------------------
# Data Management & Caching
# ------------------------------------------------------------------------------
def load_and_process_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, extracts features (or loads from cache), and returns X, y, ids.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    df_meta = pd.read_csv(metadata_path)

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        df_features = pd.read_parquet(cache_path)
    else:
        print(f"Processing images for {cache_name}...")
        # Extract geometric features
        geo_features = []
        for idx, row in df_meta.iterrows():
            # Construct full path from metadata
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
            feats = extract_geometric_features(full_path)
            geo_features.append(feats)

        df_geo = pd.DataFrame(geo_features, columns=FEATURE_NAMES_GEO)

        # Combine with provided tabular features
        tab_cols = []
        for prefix in Config.TABULAR_PREFIXES:
            for i in range(1, Config.NUM_FEATURES_PER_GROUP + 1):
                tab_cols.append(f"{prefix}{i}")

        df_tab = df_meta[tab_cols].astype(Config.FLOAT_PRECISION)

        # Merge ID, Features, and Target
        df_features = pd.concat([df_meta[["id"]], df_tab, df_geo], axis=1)

        if "species" in df_meta.columns:
            df_features["species"] = df_meta["species"]

        # Save to cache
        print(f"Saving features to {cache_path}")
        df_features.to_parquet(cache_path)

    # Prepare return values
    ids = df_features["id"].values

    # Identify feature columns (exclude id, species)
    exclude = ["id", "species"]
    feature_cols = [c for c in df_features.columns if c not in exclude]

    # Enforce alphanumeric sort for deterministic memory layout
    feature_cols.sort()

    X = df_features[feature_cols].values.astype(Config.FLOAT_PRECISION)

    y = None
    if "species" in df_features.columns:
        y = df_features["species"].values

    return X, y, ids


# ------------------------------------------------------------------------------
# Model: OAS Linear Discriminant
# ------------------------------------------------------------------------------
class OASLinearDiscriminant:
    """
    Custom Linear Discriminant Classifier using OAS Covariance Estimation.
    Implemented for high-precision float64 inference.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.precision_ = None
        self.priors_ = None
        self.W_ = None
        self.b_ = None
        self.le = LabelEncoder()

    def fit(self, X, y):
        # Encode classes
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Compute Priors and Means
        self.priors_ = np.zeros(n_classes, dtype=Config.FLOAT_PRECISION)
        self.means_ = np.zeros((n_classes, n_features), dtype=Config.FLOAT_PRECISION)

        for k in range(n_classes):
            X_k = X[y_enc == k]
            self.priors_[k] = len(X_k) / len(X)
            self.means_[k] = np.mean(X_k, axis=0)

        # Center data (Global covariance assumption)
        # R = X - mu_y
        R = np.zeros_like(X, dtype=Config.FLOAT_PRECISION)
        for k in range(n_classes):
            mask = y_enc == k
            R[mask] = X[mask] - self.means_[k]

        # Estimate Covariance using OAS
        # OAS is robust to high-dimensionality and collinearity
        oas = OAS(assume_centered=True)
        oas.fit(R)
        self.precision_ = oas.precision_.astype(Config.FLOAT_PRECISION)

        # Pre-compute Linear Decision Boundaries
        # W = P * mu.T -> Shape (n_features, n_classes)
        # We store W_ such that Z = X @ W_.T
        # W_ shape: (n_classes, n_features)
        self.W_ = self.means_ @ self.precision_

        # Bias b_k = -0.5 * (mu_k.T @ P @ mu_k) + log(prior_k)
        self.b_ = np.zeros(n_classes, dtype=Config.FLOAT_PRECISION)
        for k in range(n_classes):
            term1 = -0.5 * (self.means_[k] @ self.precision_ @ self.means_[k])
            term2 = np.log(self.priors_[k])
            self.b_[k] = term1 + term2

        return self

    def predict_proba(self, X):
        # Linear Discriminant Function: Z = X @ W.T + b
        Z = X @ self.W_.T + self.b_
        # Apply Softmax
        return softmax(Z, axis=1)


# ------------------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------------------
def run_solution():
    print("Initializing Integral-Morphological High-Precision OAS Discriminant...")

    # 1. Load Data
    train_csv = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv = os.path.join(Config.METADATA_DIR, "val.csv")
    test_csv = os.path.join(Config.METADATA_DIR, "test.csv")

    X_train_raw, y_train, train_ids = load_and_process_data(train_csv, "train_features")
    X_val_raw, y_val, val_ids = load_and_process_data(val_csv, "val_features")
    X_test_raw, _, test_ids = load_and_process_data(test_csv, "test_features")

    print(f"Feature Matrix Shape: {X_train_raw.shape}")

    # 2. Preprocessing
    # Pipeline: Yeo-Johnson Power Transform -> Standard Scaler
    # Fitted ONLY on training data to prevent leakage
    print("Preprocessing data (Yeo-Johnson + StandardScaler)...")
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    ss = StandardScaler()

    # Fit on Train
    X_train_trans = pt.fit_transform(X_train_raw)
    X_train_trans = ss.fit_transform(X_train_trans)

    # Transform Val and Test
    X_val_trans = ss.transform(pt.transform(X_val_raw))
    X_test_trans = ss.transform(pt.transform(X_test_raw))

    # 3. Training
    print("Training OAS Linear Discriminant...")
    model = OASLinearDiscriminant()
    model.fit(X_train_trans, y_train)

    # 4. Validation
    print("Validating...")
    y_val_pred = model.predict_proba(X_val_trans)

    # Encode val labels for log_loss calculation
    y_val_enc = model.le.transform(y_val)
    val_loss = log_loss(y_val_enc, y_val_pred)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")

    # 5. Submission
    print("Generating Submission...")
    y_test_pred = model.predict_proba(X_test_trans)

    # Format submission
    # Columns: id, Species1, Species2, ...
    submission_df = pd.DataFrame(y_test_pred, columns=model.classes_)
    submission_df.insert(0, "id", test_ids)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


# Execute the pipeline
if __name__ == "__main__":
    run_solution()
