import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.feature_extractor import FeatureExtractor


class DataProcessor:
    """
    Handles data loading, feature engineering, and preprocessing for the
    Dual-Moment GLM-Quantile Pipeline.
    """

    def __init__(self):
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        self.scaler_fvc = StandardScaler()
        self.scaler_unc = StandardScaler()
        self.feature_extractor = FeatureExtractor()

    def fit_transform_pca(self, train_feats, val_feats, test_feats):
        """
        Fits PCA on training features and transforms train, val, and test sets.
        """
        # Fit on train
        train_pca = self.pca.fit_transform(train_feats)
        # Transform others
        val_pca = self.pca.transform(val_feats)
        test_pca = self.pca.transform(test_feats)
        return train_pca, val_pca, test_pca

    def _get_baseline_info(self, df):
        """
        For training/validation data, identifies the baseline visit (earliest week)
        and propagates its FVC/Percent/Weeks to all rows for that patient.
        """
        # Sort by Patient and Weeks to ensure first row is earliest
        df = df.sort_values(["Patient", "Weeks"])

        # Get first visit per patient
        baseline = df.groupby("Patient").first().reset_index()
        baseline = baseline[["Patient", "Weeks", "FVC", "Percent"]]
        baseline.columns = [
            "Patient",
            "Baseline_Weeks",
            "Baseline_FVC",
            "Baseline_Percent",
        ]

        # Merge back onto the original dataframe
        df = df.merge(baseline, on="Patient", how="left")
        return df

    def prepare_regression_matrices(self, df, pca_features, patient_ids):
        """
        Merges clinical data with PCA features, generates interaction terms,
        and constructs the final feature matrices (X) and target vectors (y).
        """
        # 1. Merge PCA features
        # Create a dataframe for PCA features
        pca_cols = [f"PCA_{i}" for i in range(Config.PCA_COMPONENTS)]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols)
        pca_df["Patient"] = patient_ids

        # Merge PCA into main dataframe (Left join ensures we keep all weeks for each patient)
        df = df.merge(pca_df, on="Patient", how="left")

        # 2. Handle Categoricals (One-Hot Encoding)
        # We manually create columns to ensure consistency across train/test splits
        df["Sex_Male"] = (df["Sex"] == "Male").astype(np.float32)
        df["Sex_Female"] = (df["Sex"] == "Female").astype(np.float32)

        df["Smoking_Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype(
            np.float32
        )
        df["Smoking_Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype(
            np.float32
        )
        df["Smoking_Currently smokes"] = (
            df["SmokingStatus"] == "Currently smokes"
        ).astype(np.float32)

        # 3. Time Feature
        # Relative Weeks: Time elapsed since baseline measurement
        df["Relative_Weeks"] = (df["Weeks"] - df["Baseline_Weeks"]).astype(np.float32)

        # 4. Construct Feature Sets

        # Define Static Features
        static_cols = [
            "Baseline_FVC",
            "Baseline_Percent",
            "Age",
            "Sex_Male",
            "Sex_Female",
            "Smoking_Ex-smoker",
            "Smoking_Never smoked",
            "Smoking_Currently smokes",
        ] + pca_cols

        # --- FVC Model Features (X_fvc) ---
        # Structure: [Static_Features, Relative_Weeks, Static_Features * Relative_Weeks]
        # This allows the model to learn patient-specific slopes (decay rates).

        X_fvc_list = []
        X_fvc_list.append(df[static_cols].values.astype(np.float32))
        X_fvc_list.append(df[["Relative_Weeks"]].values.astype(np.float32))

        # Interaction Terms
        interactions = df[static_cols].values * df[["Relative_Weeks"]].values
        X_fvc_list.append(interactions.astype(np.float32))

        X_fvc = np.hstack(X_fvc_list)

        # --- Uncertainty Model Features (X_unc) ---
        # Structure: [Static_Features, abs(Relative_Weeks)]
        # We use absolute time horizon because uncertainty grows with distance from baseline.

        X_unc_list = []
        X_unc_list.append(df[static_cols].values.astype(np.float32))
        X_unc_list.append(np.abs(df[["Relative_Weeks"]].values).astype(np.float32))

        X_unc = np.hstack(X_unc_list)

        # Targets
        # Test set has dummy FVC=2000, but we extract it anyway (ignored later for test)
        y = df["FVC"].values.astype(np.float32) if "FVC" in df.columns else None

        # IDs
        if "Patient_Week" in df.columns:
            ids = df["Patient_Week"].values
        else:
            ids = (df["Patient"] + "_" + df["Weeks"].astype(str)).values

        return X_fvc, X_unc, y, ids

    def process(self, load_cached_data=True):
        """
        Main execution method.
        Checks for cached processed data. If not found, runs the full pipeline:
        Feature Extraction -> PCA -> Tabular Processing -> Scaling -> Caching.
        """
        # Define cache filenames
        cache_files = {
            "X_train_fvc": "X_train_fvc.npy",
            "X_train_unc": "X_train_unc.npy",
            "y_train": "y_train.npy",
            "X_val_fvc": "X_val_fvc.npy",
            "X_val_unc": "X_val_unc.npy",
            "y_val": "y_val.npy",
            "X_test_fvc": "X_test_fvc.npy",
            "X_test_unc": "X_test_unc.npy",
            "test_ids": "test_ids.npy",
        }

        # Check if all cache files exist
        cache_dir = Config.CACHE_DIR
        all_exist = all(
            os.path.exists(os.path.join(cache_dir, f)) for f in cache_files.values()
        )

        if load_cached_data and all_exist:
            print(f"Loading processed data from cache: {cache_dir}")
            data = {}
            for k, v in cache_files.items():
                data[k] = np.load(os.path.join(cache_dir, v), allow_pickle=True)
            return data

        print("Processed data not found or cache disabled. Running data pipeline...")

        # 1. Load Metadata
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        # 2. Extract Deep Features
        # The FeatureExtractor handles its own caching of the raw 2560-dim vectors
        print("Extracting/Loading deep features...")
        train_feats, train_pids = self.feature_extractor.process_dataset(
            "train", load_cached_data
        )
        val_feats, val_pids = self.feature_extractor.process_dataset(
            "val", load_cached_data
        )
        test_feats, test_pids = self.feature_extractor.process_dataset(
            "test", load_cached_data
        )

        # 3. PCA Compression
        print(f"Fitting PCA (components={Config.PCA_COMPONENTS})...")
        train_pca, val_pca, test_pca = self.fit_transform_pca(
            train_feats, val_feats, test_feats
        )

        # 4. Prepare Tabular Data (Baseline Extraction)
        # Train/Val need baseline info derived from the first visit
        df_train = self._get_baseline_info(df_train)
        df_val = self._get_baseline_info(df_val)
        # Test set already has baseline info in the metadata

        # 5. Generate Regression Matrices
        print("Constructing regression matrices...")
        X_train_fvc, X_train_unc, y_train, _ = self.prepare_regression_matrices(
            df_train, train_pca, train_pids
        )
        X_val_fvc, X_val_unc, y_val, _ = self.prepare_regression_matrices(
            df_val, val_pca, val_pids
        )
        X_test_fvc, X_test_unc, _, test_ids = self.prepare_regression_matrices(
            df_test, test_pca, test_pids
        )

        # 6. Scaling
        # Standardize features. Fit on Train, transform Val/Test.
        # This is critical for regularized linear models (Lasso/ElasticNet).
        print("Scaling features...")
        X_train_fvc = self.scaler_fvc.fit_transform(X_train_fvc)
        X_val_fvc = self.scaler_fvc.transform(X_val_fvc)
        X_test_fvc = self.scaler_fvc.transform(X_test_fvc)

        X_train_unc = self.scaler_unc.fit_transform(X_train_unc)
        X_val_unc = self.scaler_unc.transform(X_val_unc)
        X_test_unc = self.scaler_unc.transform(X_test_unc)

        # 7. Save to Cache
        print(f"Saving processed data to {cache_dir}...")
        data = {
            "X_train_fvc": X_train_fvc,
            "X_train_unc": X_train_unc,
            "y_train": y_train,
            "X_val_fvc": X_val_fvc,
            "X_val_unc": X_val_unc,
            "y_val": y_val,
            "X_test_fvc": X_test_fvc,
            "X_test_unc": X_test_unc,
            "test_ids": test_ids,
        }

        os.makedirs(cache_dir, exist_ok=True)
        for k, v in data.items():
            np.save(os.path.join(cache_dir, cache_files[k]), v)

        print("Data processing complete.")
        return data
