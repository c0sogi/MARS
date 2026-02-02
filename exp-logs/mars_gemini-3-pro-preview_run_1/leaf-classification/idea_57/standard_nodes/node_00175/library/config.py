import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.covariance import OAS
from scipy.special import softmax


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_opt"
    IMAGES_DIR = "images"

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Parameters
    SEED = 42
    N_JOBS = 12

    # Feature Groups
    TABULAR_FEATURES_PREFIX = ["margin", "shape", "texture"]
    GEOMETRIC_FEATURES = [
        "geo_area",
        "geo_eccentricity",
        "geo_solidity",
        "geo_extent",
        "geo_aspect_ratio",
        "geo_roundness",
    ]


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_geometric_features(df, input_dir):
    """
    Extracts 7 geometric features from images referenced in the DataFrame.
    Returns a DataFrame with the extracted features.
    """
    features = []
    file_paths = df["file_path"].values

    for rel_path in file_paths:
        full_path = os.path.join(input_dir, rel_path)

        # Default values (0.0) for failures or empty images
        row_feats = [0.0] * 7

        if os.path.exists(full_path):
            # Load as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Polarity Correction: Leaf=White(255), Background=Black(0)
                # The dataset description says "binary black leaves against white backgrounds"
                # We invert so the leaf is the foreground for contour detection.
                _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

                # Find Contours (Lossless)
                contours, _ = cv2.findContours(
                    bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                if contours:
                    # Get largest contour by area
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)

                    if area > 0:
                        # 1. Absolute Scale: Area
                        feat_area = area

                        # 2. Internal Scale: Mean Thickness (Euclidean Distance Transform)
                        dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
                        if np.count_nonzero(dist_transform) > 0:
                            feat_thickness = np.mean(dist_transform[dist_transform > 0])
                        else:
                            feat_thickness = 0.0

                        # 3. Elongation: Eccentricity (via Ellipse Fit)
                        # e = sqrt(1 - (min_axis/max_axis)^2)
                        if len(cnt) >= 5:
                            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                            # Sort axes to ensure min/max
                            axes = sorted([MA, ma])
                            min_axis, max_axis = axes[0], axes[1]
                            if max_axis > 0:
                                feat_eccentricity = np.sqrt(
                                    1 - (min_axis / max_axis) ** 2
                                )
                            else:
                                feat_eccentricity = 0.0
                        else:
                            feat_eccentricity = 0.0

                        # 4. Roughness: Solidity
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        feat_solidity = area / hull_area if hull_area > 0 else 0.0

                        # 5. Rectangularity: Extent
                        x, y, w, h = cv2.boundingRect(cnt)
                        rect_area = w * h
                        feat_extent = area / rect_area if rect_area > 0 else 0.0

                        # 6. Orientation: Aspect Ratio
                        feat_aspect_ratio = float(w) / h if h > 0 else 0.0

                        # 7. Compactness: Roundness
                        perimeter = cv2.arcLength(cnt, True)
                        # 4 * pi * Area / Perimeter^2
                        feat_roundness = (
                            (4 * np.pi * area) / (perimeter**2)
                            if perimeter > 0
                            else 0.0
                        )

                        row_feats = [
                            feat_area,
                            feat_thickness,
                            feat_eccentricity,
                            feat_solidity,
                            feat_extent,
                            feat_aspect_ratio,
                            feat_roundness,
                        ]

        features.append(row_feats)

    feat_df = pd.DataFrame(features, columns=Config.GEOMETRIC_FEATURES)
    return feat_df


def load_and_process_data(subset, load_cached_data=True):
    """
    Loads metadata, extracts/loads features, and returns X (features), y (labels), ids, and feature names.
    subset: 'train', 'val', or 'test'
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{subset}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {subset} data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print(f"Processing {subset} data from scratch...")
        if subset == "train":
            meta_path = Config.TRAIN_META
        elif subset == "val":
            meta_path = Config.VAL_META
        else:
            meta_path = Config.TEST_META

        df_meta = pd.read_csv(meta_path)

        # Extract Geometric Features
        print(f"Extracting geometric features for {subset}...")
        geo_df = extract_geometric_features(df_meta, Config.INPUT_DIR)

        # Select Tabular Features
        tab_cols = [
            c
            for c in df_meta.columns
            if any(c.startswith(p) for p in Config.TABULAR_FEATURES_PREFIX)
        ]

        # Concatenate Features
        # We keep 'id' and 'species' (if available) for alignment
        cols_to_keep = ["id"]
        if "species" in df_meta.columns:
            cols_to_keep.append("species")

        df_combined = pd.concat(
            [df_meta[cols_to_keep], df_meta[tab_cols], geo_df], axis=1
        )

        # Save to cache
        df_combined.to_parquet(cache_path)
        df = df_combined

    # Prepare X, y
    exclude_cols = ["id", "species"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    # Enforce deterministic column order
    feature_cols.sort()

    # High-Precision Loading
    X = df[feature_cols].values.astype(np.float64)
    ids = df["id"].values

    y = None
    if "species" in df.columns:
        y = df["species"].values

    return X, y, ids, feature_cols


class LeafModel:
    """
    Custom Linear Discriminant Classifier using OAS Covariance Estimator.
    Implements the Sanitized Parsimonious Integral-Geometric High-Precision strategy.
    """

    def __init__(self):
        # Sanitization Barrier: Remove constant features to prevent scaler explosion
        self.sanitizer = VarianceThreshold(threshold=0.0)
        # Inductive Transformation: Stabilize variance
        self.transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        # Scaling
        self.scaler = StandardScaler()
        # Backbone: Oracle Approximating Shrinkage
        self.estimator = OAS(assume_centered=True)

        self.classes_ = None
        self.means_ = None
        self.precision_ = None
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        # 1. Pipeline Fitting
        X_clean = self.sanitizer.fit_transform(X)
        X_trans = self.transformer.fit_transform(X_clean)
        X_scaled = self.scaler.fit_transform(X_trans)

        # 2. Parameter Estimation
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X_scaled.shape[1]

        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # Compute Class Means and Priors
        for idx, cls in enumerate(self.classes_):
            X_cls = X_scaled[y == cls]
            self.means_[idx] = np.mean(X_cls, axis=0)
            priors[idx] = len(X_cls) / len(X)

        # Compute Residuals for Covariance Estimation
        X_centered = X_scaled.copy()
        for idx, cls in enumerate(self.classes_):
            X_centered[y == cls] -= self.means_[idx]

        # Fit OAS Estimator
        self.estimator.fit(X_centered)
        self.precision_ = self.estimator.precision_

        # 3. Weight Derivation (Linear Formulation)
        # W = P * mu.T -> (n_classes, n_features)
        # Weights_k = Precision @ Mean_k
        self.weights_ = self.means_ @ self.precision_

        # Bias Derivation
        # b_k = -0.5 * (W_k . mu_k) + log(pi_k)
        quad_term = -0.5 * np.sum(self.weights_ * self.means_, axis=1)
        self.bias_ = quad_term + np.log(priors)

    def predict_proba(self, X):
        # Apply Pipeline
        X_clean = self.sanitizer.transform(X)
        X_trans = self.transformer.transform(X_clean)
        X_scaled = self.scaler.transform(X_trans)

        # Linear Scoring: Z = XW^T + b
        logits = X_scaled @ self.weights_.T + self.bias_

        # Softmax in float64
        probs = softmax(logits, axis=1)
        return probs


def run_training_and_submission(load_cached_data=True):
    """
    Main execution function.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    print("Loading Train...")
    X_train, y_train, _, _ = load_and_process_data("train", load_cached_data)
    print("Loading Val...")
    X_val, y_val, _, _ = load_and_process_data("val", load_cached_data)
    print("Loading Test...")
    X_test, _, ids_test, _ = load_and_process_data("test", load_cached_data)

    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    print(f"Test Shape:  {X_test.shape}")

    # 2. Train Model
    print("Training Model...")
    model = LeafModel()
    model.fit(X_train, y_train)

    # 3. Validate
    print("Validating...")
    probs_val = model.predict_proba(X_val)
    preds_val_idx = np.argmax(probs_val, axis=1)
    preds_val = model.classes_[preds_val_idx]

    # Accuracy
    acc = np.mean(preds_val == y_val)

    # Log Loss
    class_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
    y_val_idx = np.array([class_to_idx[y] for y in y_val])

    # Clip for stability in metric calculation
    eps = 1e-15
    probs_val_clipped = np.clip(probs_val, eps, 1 - eps)
    probs_val_clipped /= probs_val_clipped.sum(axis=1, keepdims=True)
    log_loss = -np.mean(np.log(probs_val_clipped[np.arange(len(y_val)), y_val_idx]))

    print(f"Validation Accuracy: {acc:.16f}")
    print(f"Validation Log Loss: {log_loss:.16f}")

    # 4. Generate Submission
    print("Generating Submission...")
    probs_test = model.predict_proba(X_test)

    submission_df = pd.DataFrame(probs_test, columns=model.classes_)
    submission_df.insert(0, "id", ids_test)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
