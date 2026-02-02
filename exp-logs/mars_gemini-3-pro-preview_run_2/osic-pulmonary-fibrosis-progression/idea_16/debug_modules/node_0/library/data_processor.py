import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from library.config import (
    CACHE_DIR,
    PCA_COMPONENTS,
    SEED,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
)


class TabularPreprocessor:
    def __init__(self, pca_components=PCA_COMPONENTS, seed=SEED):
        self.pca_components = pca_components
        self.seed = seed

        # Encoders
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.pca = PCA(n_components=self.pca_components, random_state=self.seed)

        # Feature names tracking
        self.num_cols = ["Baseline_FVC", "Baseline_Percent", "Age"]
        self.cat_cols = ["Sex", "SmokingStatus"]

    def _get_baseline_info(self, df, is_test=False):
        """
        Derives Baseline_FVC, Baseline_Percent, Baseline_Weeks for the dataframe.
        For Train/Val: Finds the visit closest to Week 0.
        For Test: Assumes these columns already exist.
        """
        df = df.copy()

        if is_test:
            # Test metadata already has Baseline columns populated from test.csv
            # Just ensure types are correct
            return df

        # For Train/Val, we need to identify the baseline visit
        # We define baseline as the visit closest to Weeks=0 (CT scan)
        df["abs_weeks"] = df["Weeks"].abs()

        # Sort by patient and closeness to week 0
        df_sorted = df.sort_values(["Patient", "abs_weeks"])

        # Take the first record for each patient as baseline
        baseline_df = df_sorted.groupby("Patient").first().reset_index()

        # Select relevant columns and rename
        baseline_cols = ["Patient", "FVC", "Percent", "Weeks"]
        baseline_df = baseline_df[baseline_cols].rename(
            columns={
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Weeks": "Baseline_Weeks",
            }
        )

        # Merge back to original dataframe
        # Note: We drop the temp 'abs_weeks' before merging
        df = df.drop(columns=["abs_weeks"])
        df = pd.merge(df, baseline_df, on="Patient", how="left")

        return df

    def fit(self, train_df, train_img_feats):
        """
        Fits Scaler, OHE, and PCA on training data.
        """
        # 1. Prepare Baseline Info
        df = self._get_baseline_info(train_df, is_test=False)

        # 2. Fit Clinical Scalers
        self.scaler.fit(df[self.num_cols])
        self.ohe.fit(df[self.cat_cols])

        # 3. Fit PCA on Image Features
        # Collect all image vectors
        img_vectors = []
        for pid in df["Patient"].unique():
            if pid in train_img_feats:
                img_vectors.append(train_img_feats[pid])
            else:
                # Fallback for missing keys (should not happen if extraction worked)
                img_vectors.append(np.zeros(1280))

        if len(img_vectors) > 0:
            self.pca.fit(np.stack(img_vectors))
        else:
            # Fallback if no images provided (e.g. debugging)
            self.pca.fit(np.zeros((10, 1280)))

    def transform(self, df, img_feats, is_test=False):
        """
        Transforms data into X_fvc and X_unc feature matrices.
        """
        # 1. Prepare Baseline Info
        df = self._get_baseline_info(df, is_test=is_test)

        # 2. Calculate Time Feature (Relative Weeks)
        # t = Weeks - Baseline_Weeks
        df["RelativeWeeks"] = df["Weeks"] - df["Baseline_Weeks"]

        # 3. Transform Clinical Features
        X_num = self.scaler.transform(df[self.num_cols])
        X_cat = self.ohe.transform(df[self.cat_cols])

        # 4. Transform Image Features
        # We need to map rows to image features
        pca_vectors = []
        for pid in df["Patient"]:
            if pid in img_feats:
                vec = img_feats[pid]
            else:
                vec = np.zeros(1280)
            pca_vectors.append(vec)

        X_img = self.pca.transform(np.stack(pca_vectors))

        # 5. Construct Feature Sets

        # Common Base: [Num, Cat, PCA]
        X_base = np.hstack([X_num, X_cat, X_img])

        # Time Vector (N, 1)
        time = df["RelativeWeeks"].values.reshape(-1, 1)

        # --- FVC Model Features ---
        # Structure: [Base, Time, PCA * Time]
        # Interaction: Multiply each PCA component by Time
        X_interaction = X_img * time
        X_fvc = np.hstack([X_base, time, X_interaction])

        # --- Uncertainty Model Features ---
        # Structure: [Base, abs(Time)]
        # Horizon term
        horizon = np.abs(time)
        X_unc = np.hstack([X_base, horizon])

        # Extract Target if available
        y = None
        if "FVC" in df.columns and not is_test:
            # Note: For test_metadata, 'FVC' is a placeholder 2000
            y = df["FVC"].values

        return X_fvc, X_unc, y, df["Patient_Week"].values if is_test else None


def process_data(train_feats, val_feats, test_feats, load_cached_data=True):
    """
    Main entry point. Handles caching and processing.
    """
    # Define cache paths
    paths = {
        "X_fvc_train": os.path.join(CACHE_DIR, "X_fvc_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_unc_train": os.path.join(CACHE_DIR, "X_unc_train.npy"),
        "X_fvc_val": os.path.join(CACHE_DIR, "X_fvc_val.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_unc_val": os.path.join(CACHE_DIR, "X_unc_val.npy"),
        "X_fvc_test": os.path.join(CACHE_DIR, "X_fvc_test.npy"),
        "X_unc_test": os.path.join(CACHE_DIR, "X_unc_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check cache
    all_exist = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_exist:
        print("Loading processed tabular data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in paths.items()}
        return (
            data["X_fvc_train"],
            data["y_train"],
            data["X_unc_train"],
            data["X_fvc_val"],
            data["y_val"],
            data["X_unc_val"],
            data["X_fvc_test"],
            data["X_unc_test"],
            data["test_ids"],
        )

    print("Processing tabular data from scratch...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Initialize and Fit Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df, train_feats)

    # Transform Datasets
    X_fvc_train, X_unc_train, y_train, _ = preprocessor.transform(
        train_df, train_feats, is_test=False
    )
    X_fvc_val, X_unc_val, y_val, _ = preprocessor.transform(
        val_df, val_feats, is_test=False
    )
    X_fvc_test, X_unc_test, _, test_ids = preprocessor.transform(
        test_df, test_feats, is_test=True
    )

    # Save to Cache
    np.save(paths["X_fvc_train"], X_fvc_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_unc_train"], X_unc_train)

    np.save(paths["X_fvc_val"], X_fvc_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_unc_val"], X_unc_val)

    np.save(paths["X_fvc_test"], X_fvc_test)
    np.save(paths["X_unc_test"], X_unc_test)
    np.save(paths["test_ids"], test_ids)

    print(f"Data processing complete. Features saved to {CACHE_DIR}")

    return (
        X_fvc_train,
        y_train,
        X_unc_train,
        X_fvc_val,
        y_val,
        X_unc_val,
        X_fvc_test,
        X_unc_test,
        test_ids,
    )
