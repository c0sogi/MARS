import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import Config
from library.feature_extraction import FeaturePipeline


class DataPreparation:
    """
    Handles the preparation of training and test datasets.
    Merges clinical metadata with extracted image features and performs
    feature engineering for the Quantile-GLM architecture.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.feature_pipeline = FeaturePipeline()

        # Cache paths
        self.train_cache = os.path.join(self.working_dir, "processed_train.npz")
        self.val_cache = os.path.join(self.working_dir, "processed_val.npz")
        self.test_cache = os.path.join(self.working_dir, "processed_test.npz")
        self.preprocessors_path = os.path.join(self.working_dir, "preprocessors.pkl")

    def _load_metadata(self, mode):
        """Loads the appropriate metadata CSV."""
        if mode == "train":
            return pd.read_csv(Config.TRAIN_METADATA)
        elif mode == "val":
            return pd.read_csv(Config.VAL_METADATA)
        elif mode == "test":
            return pd.read_csv(Config.TEST_METADATA)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _fit_scalers(self, df):
        """Fits StandardScaler and OneHotEncoder on training data."""
        # Numerical
        num_cols = ["Age", "Percent"]
        scaler = StandardScaler()
        scaler.fit(df[num_cols])

        # Categorical
        cat_cols = ["Sex", "SmokingStatus"]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(df[cat_cols])

        return scaler, encoder

    def _transform_tabular(self, df, scaler, encoder, is_test=False):
        """Applies scaling and encoding to tabular features."""
        df_trans = df.copy()

        # Align columns for test set
        if is_test:
            # Test set has 'Baseline_Percent' which corresponds to 'Percent' in training
            if "Baseline_Percent" in df_trans.columns:
                df_trans = df_trans.rename(columns={"Baseline_Percent": "Percent"})

        # Numerical
        num_cols = ["Age", "Percent"]
        num_data = scaler.transform(df_trans[num_cols])

        # Categorical
        cat_cols = ["Sex", "SmokingStatus"]
        cat_data = encoder.transform(df_trans[cat_cols])

        # Concatenate
        return np.concatenate([num_data, cat_data], axis=1)

    def _prepare_matrices(self, df_meta, df_img, tabular_feats, is_test=False):
        """
        Constructs the specific feature matrices for FVC and Uncertainty models.

        Args:
            df_meta: Metadata DataFrame (contains Weeks, FVC, etc.)
            df_img: DataFrame containing image features (Patient, feat_0...)
            tabular_feats: Preprocessed numpy array of tabular features
            is_test: Boolean flag for test set processing logic

        Returns:
            X_fvc: Features for Quantile Regressor (includes interactions)
            X_unc: Features for Uncertainty GLM (includes time horizon)
            y: Target values (FVC)
            ids: Patient IDs
            weeks: Relative weeks array
        """
        # 1. Merge Image Features with Metadata
        # Note: df_meta may have multiple rows per patient (visits), df_img has 1 per patient.
        # We merge on Patient to broadcast image features to all visits.
        img_feat_cols = [c for c in df_img.columns if c.startswith("feat_")]
        merged = pd.merge(df_meta, df_img, on="Patient", how="left")

        # Fill missing image features with 0 (safety check, though pipeline handles this)
        merged[img_feat_cols] = merged[img_feat_cols].fillna(0)

        # Extract image feature matrix aligned with metadata rows
        X_img = merged[img_feat_cols].values.astype(np.float32)

        # 2. Create Base Static Feature Vector [Tabular, Image]
        X_static = np.concatenate([tabular_feats, X_img], axis=1)

        # 3. Calculate Relative Time (Weeks)
        if is_test:
            # For test, target is 'Weeks', baseline is 'Baseline_Weeks'
            weeks_col = merged["Weeks"].values.astype(np.float32)
            baseline_col = merged["Baseline_Weeks"].values.astype(np.float32)
            rel_weeks = weeks_col - baseline_col
        else:
            # For train, 'Weeks' is already relative to baseline
            rel_weeks = merged["Weeks"].values.astype(np.float32)

        rel_weeks = rel_weeks.reshape(-1, 1)

        # 4. FVC Model Features: [X_static, t, X_static * t]
        # Interaction terms allow the model to learn patient-specific slopes
        X_interactions = X_static * rel_weeks
        X_fvc = np.concatenate([X_static, rel_weeks, X_interactions], axis=1)

        # 5. Uncertainty Model Features: [X_static, |t|]
        # Uncertainty grows with absolute distance from baseline
        X_unc = np.concatenate([X_static, np.abs(rel_weeks)], axis=1)

        # 6. Targets
        if "FVC" in merged.columns:
            y = merged["FVC"].values.astype(np.float32)
        else:
            y = np.zeros(len(merged), dtype=np.float32)

        return X_fvc, X_unc, y, merged["Patient"].values, rel_weeks.flatten()

    def get_train_val_data(self, load_cached_data=True):
        """
        Generates or loads processed training and validation datasets.

        Returns:
            X_fvc_train, X_unc_train, y_train, X_fvc_val, X_unc_val, y_val
        """
        # Check cache
        if (
            load_cached_data
            and os.path.exists(self.train_cache)
            and os.path.exists(self.val_cache)
        ):
            print("Loading processed train/val data from cache...")
            t_data = np.load(self.train_cache)
            v_data = np.load(self.val_cache)
            return (
                t_data["X_fvc"],
                t_data["X_unc"],
                t_data["y"],
                v_data["X_fvc"],
                v_data["X_unc"],
                v_data["y"],
            )

        print("Processing train/val data from scratch...")

        # 1. Load Metadata
        df_train = self._load_metadata("train")
        df_val = self._load_metadata("val")

        # Debugging limit
        if Config.DEBUG_DATA_SIZE:
            print(
                f"DEBUG: Limiting train/val data to {Config.DEBUG_DATA_SIZE} patients"
            )
            # Filter by unique patients to keep structure intact
            train_pats = df_train["Patient"].unique()[: Config.DEBUG_DATA_SIZE]
            val_pats = df_val["Patient"].unique()[: Config.DEBUG_DATA_SIZE]
            df_train = df_train[df_train["Patient"].isin(train_pats)]
            df_val = df_val[df_val["Patient"].isin(val_pats)]

        # 2. Process Image Features
        df_img_train = self.feature_pipeline.process_dataset(
            df_train, mode="train", load_cached_data=load_cached_data
        )
        df_img_val = self.feature_pipeline.process_dataset(
            df_val, mode="val", load_cached_data=load_cached_data
        )

        # 3. Fit Scalers (on Train only)
        scaler, encoder = self._fit_scalers(df_train)

        # Save preprocessors for inference consistency (Cite debug_lesson_8)
        with open(self.preprocessors_path, "wb") as f:
            pickle.dump((scaler, encoder), f)

        # 4. Transform Tabular Data
        tab_train = self._transform_tabular(df_train, scaler, encoder)
        tab_val = self._transform_tabular(df_val, scaler, encoder)

        # 5. Build Final Feature Matrices
        X_fvc_train, X_unc_train, y_train, _, _ = self._prepare_matrices(
            df_train, df_img_train, tab_train, is_test=False
        )
        X_fvc_val, X_unc_val, y_val, _, _ = self._prepare_matrices(
            df_val, df_img_val, tab_val, is_test=False
        )

        # 6. Cache Results
        os.makedirs(self.working_dir, exist_ok=True)
        np.savez(self.train_cache, X_fvc=X_fvc_train, X_unc=X_unc_train, y=y_train)
        np.savez(self.val_cache, X_fvc=X_fvc_val, X_unc=X_unc_val, y=y_val)

        return X_fvc_train, X_unc_train, y_train, X_fvc_val, X_unc_val, y_val

    def get_test_data(self, load_cached_data=True):
        """
        Generates or loads processed test dataset.

        Returns:
            X_fvc_test, X_unc_test, df_ids (DataFrame with Patient_Week for submission)
        """
        if load_cached_data and os.path.exists(self.test_cache):
            print("Loading processed test data from cache...")
            data = np.load(self.test_cache, allow_pickle=True)
            df_ids = pd.DataFrame({"Patient_Week": data["ids"]})
            return data["X_fvc"], data["X_unc"], df_ids

        print("Processing test data from scratch...")

        # 1. Load Test Metadata
        df_test = self._load_metadata("test")

        # Debugging limit
        if Config.DEBUG_DATA_SIZE:
            test_pats = df_test["Patient"].unique()[: Config.DEBUG_DATA_SIZE]
            df_test = df_test[df_test["Patient"].isin(test_pats)]

        # 2. Process Image Features
        df_img_test = self.feature_pipeline.process_dataset(
            df_test, mode="test", load_cached_data=load_cached_data
        )

        # 3. Load Scalers from Training
        # We load the pre-fitted scalers to ensure consistency with training (Cite debug_lesson_8)
        if os.path.exists(self.preprocessors_path):
            with open(self.preprocessors_path, "rb") as f:
                scaler, encoder = pickle.load(f)
        else:
            # Fallback: Re-fit, ensuring we respect the debug filter if active
            print("Warning: Preprocessors not found. Re-fitting...")
            df_train = self._load_metadata("train")
            if Config.DEBUG_DATA_SIZE:
                train_pats = df_train["Patient"].unique()[: Config.DEBUG_DATA_SIZE]
                df_train = df_train[df_train["Patient"].isin(train_pats)]
            scaler, encoder = self._fit_scalers(df_train)

        # 4. Transform Tabular Data
        tab_test = self._transform_tabular(df_test, scaler, encoder, is_test=True)

        # 5. Build Final Feature Matrices
        X_fvc_test, X_unc_test, _, _, _ = self._prepare_matrices(
            df_test, df_img_test, tab_test, is_test=True
        )

        # 6. Cache Results
        os.makedirs(self.working_dir, exist_ok=True)
        ids = df_test["Patient_Week"].values
        np.savez(self.test_cache, X_fvc=X_fvc_test, X_unc=X_unc_test, ids=ids)

        return X_fvc_test, X_unc_test, df_test[["Patient_Week"]]
