import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor, ElasticNet
from sklearn.metrics import mean_absolute_error

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    SUBMISSION_PATH,
    PCA_COMPONENTS,
    QUANTILE,
    ELASTIC_NET_ALPHAS,
    ELASTIC_NET_L1_RATIO,
    SEED,
    N_JOBS,
)
from library.utils import seed_everything, laplace_log_likelihood
from library.feature_generator import FeatureGenerator


class LungFunctionPredictor:
    def __init__(self):
        seed_everything(SEED)
        self.feature_generator = FeatureGenerator()

        # Preprocessing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)

        # Models
        # Solver 'highs' is faster for large datasets in recent sklearn versions
        self.fvc_model = QuantileRegressor(quantile=QUANTILE, alpha=0, solver="highs")
        # Using a fixed alpha for simplicity as per prompt description,
        # though CV could be used if time permitted.
        self.unc_model = ElasticNet(
            alpha=ELASTIC_NET_ALPHAS[2],  # 0.1
            l1_ratio=ELASTIC_NET_L1_RATIO,
            random_state=SEED,
        )

    def _get_feature_cache_paths(self, split_name):
        """Returns paths for cached numpy arrays."""
        base = os.path.join(CACHE_DIR, f"dataset_{split_name}")
        return {
            "X_fvc": f"{base}_X_fvc.npy",
            "X_unc": f"{base}_X_unc.npy",
            "y": f"{base}_y.npy",
            "meta": f"{base}_meta.pkl",  # Store dataframe for ID mapping
        }

    def _build_matrices(self, df, patient_features, is_train=True, fit_pca=False):
        """
        Constructs the specific feature matrices for FVC and Uncertainty models.
        """
        # 1. Aggregate Raw Features
        raw_feats = []
        weeks = []
        horizons = []
        y = []

        # Pre-fetch patient features to avoid repeated lookups
        # Filter df to only patients we have features for (safety check)
        valid_indices = []

        for idx, row in df.iterrows():
            pid = row["Patient"]
            if pid in patient_features:
                raw_feats.append(patient_features[pid])
                w = row["Weeks"]
                weeks.append(w)

                # Determine Baseline Week for Horizon Calculation
                # For Train/Val: Baseline is effectively 0 (CT scan time)
                # For Test: Baseline_Weeks is in metadata
                if "Baseline_Weeks" in row:
                    base_w = row["Baseline_Weeks"]
                else:
                    base_w = 0

                horizons.append(abs(w - base_w))

                if "FVC" in row:
                    y.append(row["FVC"])
                else:
                    y.append(0)  # Placeholder for test

                valid_indices.append(idx)

        # Convert to numpy
        X_raw = np.array(raw_feats, dtype=np.float32)
        weeks = np.array(weeks, dtype=np.float32).reshape(-1, 1)
        horizons = np.array(horizons, dtype=np.float32).reshape(-1, 1)
        y = np.array(y, dtype=np.float32)

        # 2. PCA Transformation
        if fit_pca:
            # Scale then PCA
            X_scaled = self.scaler.fit_transform(X_raw)
            X_pca = self.pca.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler.transform(X_raw)
            X_pca = self.pca.transform(X_scaled)

        # 3. Construct FVC Features: PCA + Weeks + Interactions
        # Interactions: PCA_i * Weeks
        interactions = X_pca * weeks
        X_fvc = np.hstack([X_pca, weeks, interactions])

        # 4. Construct Uncertainty Features: PCA + Horizon
        X_unc = np.hstack([X_pca, horizons])

        return X_fvc, X_unc, y, df.loc[valid_indices].copy()

    def prepare_data(self, load_cached_data=True):
        """
        Loads metadata, generates features, and prepares matrices for Train/Val/Test.
        Handles caching.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load Metadata
        df_train = pd.read_csv(TRAIN_META_PATH)
        df_val = pd.read_csv(VAL_META_PATH)
        df_test = pd.read_csv(TEST_META_PATH)

        # Check if all cache files exist
        splits = ["train", "val", "test"]
        all_cached = True
        if load_cached_data:
            for split in splits:
                paths = self._get_feature_cache_paths(split)
                if not all(os.path.exists(p) for p in paths.values()):
                    all_cached = False
                    break

            # Check if PCA/Scaler models are cached (using pickle for these sklearn objects)
            preprocessor_path = os.path.join(CACHE_DIR, "tabular_preprocessor.pkl")
            if not os.path.exists(preprocessor_path):
                all_cached = False
        else:
            all_cached = False

        if all_cached:
            print("Loading datasets from cache...")
            data = {}
            for split in splits:
                paths = self._get_feature_cache_paths(split)
                data[f"X_fvc_{split}"] = np.load(paths["X_fvc"])
                data[f"X_unc_{split}"] = np.load(paths["X_unc"])
                data[f"y_{split}"] = np.load(paths["y"])
                with open(paths["meta"], "rb") as f:
                    data[f"df_{split}"] = pickle.load(f)

            # Load preprocessors
            with open(os.path.join(CACHE_DIR, "tabular_preprocessor.pkl"), "rb") as f:
                state = pickle.load(f)
                self.scaler = state["scaler"]
                self.pca = state["pca"]

            return data

        print("Processing data from scratch...")

        # 1. Generate Deep Features for all unique patients
        # Combine all dfs to get unique patients
        all_dfs = pd.concat([df_train, df_val, df_test], ignore_index=True)
        # Process patients
        # Note: FeatureGenerator handles its own caching of raw vectors
        patient_features = self.feature_generator.process_dataset(
            all_dfs, load_cached_data=load_cached_data
        )

        # 2. Build Matrices
        # Train (Fit PCA)
        X_fvc_train, X_unc_train, y_train, df_train_proc = self._build_matrices(
            df_train, patient_features, is_train=True, fit_pca=True
        )

        # Val (Transform PCA)
        X_fvc_val, X_unc_val, y_val, df_val_proc = self._build_matrices(
            df_val, patient_features, is_train=False, fit_pca=False
        )

        # Test (Transform PCA)
        X_fvc_test, X_unc_test, y_test, df_test_proc = self._build_matrices(
            df_test, patient_features, is_train=False, fit_pca=False
        )

        # 3. Save to Cache
        datasets = {
            "train": (X_fvc_train, X_unc_train, y_train, df_train_proc),
            "val": (X_fvc_val, X_unc_val, y_val, df_val_proc),
            "test": (X_fvc_test, X_unc_test, y_test, df_test_proc),
        }

        for split, (xf, xu, y, df_proc) in datasets.items():
            paths = self._get_feature_cache_paths(split)
            np.save(paths["X_fvc"], xf)
            np.save(paths["X_unc"], xu)
            np.save(paths["y"], y)
            with open(paths["meta"], "wb") as f:
                pickle.dump(df_proc, f)

        # Save preprocessors
        with open(os.path.join(CACHE_DIR, "tabular_preprocessor.pkl"), "wb") as f:
            pickle.dump({"scaler": self.scaler, "pca": self.pca}, f)

        return {
            "X_fvc_train": X_fvc_train,
            "X_unc_train": X_unc_train,
            "y_train": y_train,
            "df_train": df_train_proc,
            "X_fvc_val": X_fvc_val,
            "X_unc_val": X_unc_val,
            "y_val": y_val,
            "df_val": df_val_proc,
            "X_fvc_test": X_fvc_test,
            "X_unc_test": X_unc_test,
            "y_test": y_test,
            "df_test": df_test_proc,
        }

    def fit(self, data):
        """
        Trains the FVC Quantile Regressor and Uncertainty Elastic Net.
        """
        X_fvc_train = data["X_fvc_train"]
        y_train = data["y_train"]
        X_unc_train = data["X_unc_train"]

        X_fvc_val = data["X_fvc_val"]
        y_val = data["y_val"]
        X_unc_val = data["X_unc_val"]

        print(f"Training FVC Linear Quantile Regressor (q={QUANTILE})...")
        self.fvc_model.fit(X_fvc_train, y_train)

        # Evaluate FVC Model
        train_preds = self.fvc_model.predict(X_fvc_train)
        val_preds = self.fvc_model.predict(X_fvc_val)

        train_mae = mean_absolute_error(y_train, train_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        print(f"FVC MAE - Train: {train_mae}, Val: {val_mae}")

        # Prepare Targets for Uncertainty Model
        # Target is absolute residual: |y_true - y_pred|
        residuals_train = np.abs(y_train - train_preds)

        print("Training Uncertainty Elastic Net...")
        self.unc_model.fit(X_unc_train, residuals_train)

        # Evaluate Uncertainty Model (on predicting the error magnitude)
        unc_preds_train = self.unc_model.predict(X_unc_train)
        unc_preds_val = self.unc_model.predict(X_unc_val)

        # Note: unc_preds are MAD predictions.
        # We can check how well they correlate with actual residuals
        residuals_val = np.abs(y_val - val_preds)
        unc_mae_train = mean_absolute_error(residuals_train, unc_preds_train)
        unc_mae_val = mean_absolute_error(residuals_val, unc_preds_val)
        print(
            f"Uncertainty MAE (Predicting Residuals) - Train: {unc_mae_train}, Val: {unc_mae_val}"
        )

        # Compute Laplace Metric on Validation Set
        # Convert MAD to Sigma: sigma = MAD * sqrt(2)
        sigma_val = unc_preds_val * np.sqrt(2)
        val_score = laplace_log_likelihood(y_val, val_preds, sigma_val)
        print(f"Validation Laplace Log Likelihood: {val_score}")

    def predict(self, data):
        """
        Generates predictions for the test set and saves submission.
        """
        X_fvc_test = data["X_fvc_test"]
        X_unc_test = data["X_unc_test"]
        df_test = data["df_test"]

        print("Generating Test Predictions...")

        # 1. Predict Median FVC
        fvc_pred = self.fvc_model.predict(X_fvc_test)

        # 2. Predict Uncertainty (MAD)
        mad_pred = self.unc_model.predict(X_unc_test)

        # 3. Convert to Confidence (Sigma)
        # Analytical scaling: sigma = MAD * sqrt(2)
        confidence_pred = mad_pred * np.sqrt(2)

        # 4. Create Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "Patient_Week": df_test["Patient_Week"],
                "FVC": fvc_pred,
                "Confidence": confidence_pred,
            }
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        sub_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

        return sub_df

    def run(self, load_cached_data=True):
        data = self.prepare_data(load_cached_data=load_cached_data)
        self.fit(data)
        self.predict(data)
