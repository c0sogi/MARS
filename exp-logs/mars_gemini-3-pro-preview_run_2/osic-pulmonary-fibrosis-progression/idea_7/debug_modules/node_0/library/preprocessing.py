import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    IDEA_DIR,
    SEED,
)
from library.utils import save_numpy, load_numpy

# Constants for caching
CACHE_FILES = {
    "train": ["X_fvc_train.npy", "y_train.npy", "X_unc_train.npy"],
    "val": ["X_fvc_val.npy", "y_val.npy", "X_unc_val.npy"],
    "test": ["X_fvc_test.npy", "X_unc_test.npy"],  # No y for test
}


class TabularProcessor:
    """
    Handles scaling and encoding of tabular clinical features.
    """

    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def fit(self, df):
        """
        Fits the preprocessor on the training data.
        """
        # Numerical features to scale
        num_cols = ["Age", "Baseline_FVC", "Baseline_Percent"]
        # Categorical features to encode
        cat_cols = ["Sex", "SmokingStatus"]

        # Pipeline for numerical features
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Pipeline for categorical features
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols),
            ]
        )

        self.preprocessor.fit(df)

    def transform(self, df):
        """
        Transforms the dataframe into a numpy array.
        """
        if self.preprocessor is None:
            raise ValueError("TabularProcessor has not been fitted yet.")
        return self.preprocessor.transform(df)

    def save(self, path):
        joblib.dump(self.preprocessor, path)

    def load(self, path):
        self.preprocessor = joblib.load(path)


def get_baseline_info(df):
    """
    Derives baseline FVC, Percent, and Weeks for training/validation data.
    The baseline is defined as the visit closest to Week 0 (the CT scan week).
    """
    df_temp = df.copy()
    # Calculate distance from Week 0
    df_temp["dist_from_zero"] = df_temp["Weeks"].abs()

    # Sort by Patient and distance, then pick the first one
    df_temp = df_temp.sort_values(["Patient", "dist_from_zero"])
    baseline_df = df_temp.groupby("Patient").first().reset_index()

    # Select and rename columns
    cols = ["Patient", "Weeks", "FVC", "Percent"]
    baseline_df = baseline_df[cols]
    baseline_df.columns = [
        "Patient",
        "Baseline_Weeks",
        "Baseline_FVC",
        "Baseline_Percent",
    ]

    return baseline_df


def create_interaction_features(X_base, time_col):
    """
    Creates features for the FVC model: [X_base, t, X_base * t]
    """
    # Ensure time_col is (N, 1)
    if time_col.ndim == 1:
        time_col = time_col.reshape(-1, 1)

    # Interaction terms: Element-wise multiplication of feature vector by time
    interactions = X_base * time_col

    # Concatenate: Features, Time, Interactions
    return np.hstack([X_base, time_col, interactions])


def create_horizon_features(X_base, time_col):
    """
    Creates features for the Uncertainty model: [X_base, |t|]
    """
    if time_col.ndim == 1:
        time_col = time_col.reshape(-1, 1)

    # Absolute time horizon
    time_horizon = np.abs(time_col)

    return np.hstack([X_base, time_horizon])


def process_subset(df, image_features, tabular_processor, is_train=True):
    """
    Helper to process a single subset (train/val/test).
    """
    # 1. Prepare Tabular Data
    if is_train:
        # For Train/Val, we need to derive baseline info from history
        baseline_info = get_baseline_info(df)
        # Merge baseline info back to the original dataframe
        df_proc = df.merge(baseline_info, on="Patient", how="left")
    else:
        # For Test, baseline info is already in the metadata columns
        df_proc = df.copy()

    # 2. Transform Tabular Features
    X_tab = tabular_processor.transform(df_proc)

    # 3. Retrieve and Merge Image Features
    # We assume image_features is a dict: {PatientID: np.array}
    # We must align rows of df_proc with image features
    X_img = []
    missing_cnt = 0
    feature_dim = len(next(iter(image_features.values())))

    for pid in df_proc["Patient"]:
        if pid in image_features:
            X_img.append(image_features[pid])
        else:
            # Fallback for missing image features (should be rare)
            X_img.append(np.zeros(feature_dim))
            missing_cnt += 1

    X_img = np.array(X_img)
    if missing_cnt > 0:
        print(f"Warning: {missing_cnt} samples missing image features.")

    # 4. Concatenate Base Features (Tabular + Image)
    X_base = np.hstack([X_tab, X_img])

    # 5. Calculate Time Delta (t)
    # t = Target_Week - Baseline_Week
    # Scale time by 100.0 to keep gradients stable
    time_raw = (df_proc["Weeks"].values - df_proc["Baseline_Weeks"].values).astype(
        float
    )
    time_scaled = time_raw / 100.0

    # 6. Create Model-Specific Feature Sets
    X_fvc = create_interaction_features(X_base, time_scaled)
    X_unc = create_horizon_features(X_base, time_scaled)

    # 7. Extract Targets (if available)
    if "FVC" in df_proc.columns and is_train:
        y = df_proc["FVC"].values.astype(float)
        return X_fvc, y, X_unc
    else:
        return X_fvc, None, X_unc


def run_preprocessing(
    train_image_feats, val_image_feats, test_image_feats, load_cached_data=True
):
    """
    Main entry point for data preprocessing.
    """
    # Define Cache Paths
    cache_paths = {}
    for subset, files in CACHE_FILES.items():
        cache_paths[subset] = [os.path.join(IDEA_DIR, f) for f in files]

    # 1. Check Cache
    if load_cached_data:
        all_exist = True
        for paths in cache_paths.values():
            if not all(os.path.exists(p) for p in paths):
                all_exist = False
                break

        if all_exist:
            print("[Preprocessing] Loading data from cache...")
            results = {}
            # Load Train
            results["X_fvc_train"] = load_numpy(cache_paths["train"][0])
            results["y_train"] = load_numpy(cache_paths["train"][1])
            results["X_unc_train"] = load_numpy(cache_paths["train"][2])
            # Load Val
            results["X_fvc_val"] = load_numpy(cache_paths["val"][0])
            results["y_val"] = load_numpy(cache_paths["val"][1])
            results["X_unc_val"] = load_numpy(cache_paths["val"][2])
            # Load Test
            results["X_fvc_test"] = load_numpy(cache_paths["test"][0])
            results["X_unc_test"] = load_numpy(cache_paths["test"][1])
            return results

    print("[Preprocessing] Generating features from scratch...")

    # 2. Load Metadata
    df_train = pd.read_csv(TRAIN_META_PATH)
    df_val = pd.read_csv(VAL_META_PATH)
    df_test = pd.read_csv(TEST_META_PATH)

    # 3. Fit Tabular Processor on Training Data
    # We need to temporarily compute baseline info for training data to fit the scaler correctly
    train_baseline_info = get_baseline_info(df_train)
    df_train_fit = df_train.merge(train_baseline_info, on="Patient", how="left")

    processor = TabularProcessor()
    processor.fit(df_train_fit)

    # Save processor for potential inference use
    processor.save(os.path.join(IDEA_DIR, "tabular_processor.joblib"))

    # 4. Process Datasets
    print("Processing Training Set...")
    X_fvc_train, y_train, X_unc_train = process_subset(
        df_train, train_image_feats, processor, is_train=True
    )

    print("Processing Validation Set...")
    X_fvc_val, y_val, X_unc_val = process_subset(
        df_val, val_image_feats, processor, is_train=True
    )

    print("Processing Test Set...")
    X_fvc_test, _, X_unc_test = process_subset(
        df_test, test_image_feats, processor, is_train=False
    )

    # 5. Save to Cache
    save_numpy(cache_paths["train"][0], X_fvc_train)
    save_numpy(cache_paths["train"][1], y_train)
    save_numpy(cache_paths["train"][2], X_unc_train)

    save_numpy(cache_paths["val"][0], X_fvc_val)
    save_numpy(cache_paths["val"][1], y_val)
    save_numpy(cache_paths["val"][2], X_unc_val)

    save_numpy(cache_paths["test"][0], X_fvc_test)
    save_numpy(cache_paths["test"][1], X_unc_test)

    print(f"[Preprocessing] Data saved to {IDEA_DIR}")

    return {
        "X_fvc_train": X_fvc_train,
        "y_train": y_train,
        "X_unc_train": X_unc_train,
        "X_fvc_val": X_fvc_val,
        "y_val": y_val,
        "X_unc_val": X_unc_val,
        "X_fvc_test": X_fvc_test,
        "X_unc_test": X_unc_test,
    }
