import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from library.config import Config


class DataProcessor:
    """
    Handles tabular data preprocessing, feature engineering, and dimensionality reduction.
    """

    def __init__(self):
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        self.scaler_clinical = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fitted = False

    def _augment_baseline_data(self, df):
        """
        For training/val data, identifies the baseline visit (closest to Week 0)
        and propagates Baseline_FVC, Baseline_Percent, Baseline_Weeks to all rows.
        """
        # If already has baseline columns (e.g. test set), return as is
        if "Baseline_FVC" in df.columns:
            return df.copy()

        augmented_rows = []

        # Group by patient to find their specific baseline
        for patient_id, group in df.groupby("Patient"):
            # Find the row with Weeks closest to 0
            # We assume the CT scan is at Week 0.
            # The visit closest to 0 is the baseline measurement.
            group = group.copy()

            # Sort by absolute distance to 0
            group["abs_week"] = group["Weeks"].abs()
            baseline_idx = group["abs_week"].idxmin()
            baseline_row = group.loc[baseline_idx]

            # Propagate baseline values
            group["Baseline_Weeks"] = baseline_row["Weeks"]
            group["Baseline_FVC"] = baseline_row["FVC"]
            group["Baseline_Percent"] = baseline_row["Percent"]

            # Drop helper
            group = group.drop(columns=["abs_week"])
            augmented_rows.append(group)

        return pd.concat(augmented_rows, ignore_index=True)

    def _merge_image_features(self, df, feature_dict):
        """
        Merges CNN embeddings and Radiomics into the dataframe.
        feature_dict: {'patient_ids': [], 'embeddings': (N, D), 'radiomics': (N, 4)}
        """
        # Create a lookup dictionary
        pat_to_idx = {pid: i for i, pid in enumerate(feature_dict["patient_ids"])}

        embeddings = []
        radiomics = []

        # We process row by row to maintain order
        # (A bit slow but safe for alignment)
        for _, row in df.iterrows():
            pid = row["Patient"]
            if pid in pat_to_idx:
                idx = pat_to_idx[pid]
                embeddings.append(feature_dict["embeddings"][idx])
                radiomics.append(feature_dict["radiomics"][idx])
            else:
                # Fallback for missing patients (should not happen in valid split)
                # Use zeros matching dimensions
                emb_dim = feature_dict["embeddings"].shape[1]
                rad_dim = feature_dict["radiomics"].shape[1]
                embeddings.append(np.zeros(emb_dim))
                radiomics.append(np.zeros(rad_dim))

        return np.array(embeddings), np.array(radiomics)

    def process_data(
        self,
        train_meta_path,
        val_meta_path,
        test_meta_path,
        train_feats,
        val_feats,
        test_feats,
        load_cached_data=True,
    ):
        """
        Main pipeline to load, process, and return feature matrices.
        """
        # 1. Check Cache
        cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")

        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading processed data from {cache_path}...")
                data = np.load(cache_path)
                return {
                    "train": {
                        "X_fvc": data["train_X_fvc"],
                        "y": data["train_y"],
                        "X_unc": data["train_X_unc"],
                    },
                    "val": {
                        "X_fvc": data["val_X_fvc"],
                        "y": data["val_y"],
                        "X_unc": data["val_X_unc"],
                    },
                    "test": {
                        "X_fvc": data["test_X_fvc"],
                        "X_unc": data["test_X_unc"],
                        "patient_weeks": data["test_patient_weeks"],
                    },
                }
            except Exception as e:
                print(f"Cache load failed: {e}. Reprocessing...")

        # 2. Load Metadata
        df_train = pd.read_csv(train_meta_path)
        df_val = pd.read_csv(val_meta_path)
        df_test = pd.read_csv(test_meta_path)

        # 3. Augment with Baseline Info
        print("Augmenting tabular data with baseline info...")
        df_train = self._augment_baseline_data(df_train)
        df_val = self._augment_baseline_data(df_val)
        # df_test already has baseline columns from metadata generation

        # 4. Extract Image Features aligned with DataFrames
        print("Merging image features...")
        train_emb, train_rad = self._merge_image_features(df_train, train_feats)
        val_emb, val_rad = self._merge_image_features(df_val, val_feats)
        test_emb, test_rad = self._merge_image_features(df_test, test_feats)

        # 5. Fit Preprocessors (PCA, Scaler, OHE) on Training Data
        print("Fitting preprocessors...")

        # PCA
        self.pca.fit(train_emb)
        train_pca = self.pca.transform(train_emb)
        val_pca = self.pca.transform(val_emb)
        test_pca = self.pca.transform(test_emb)

        # Clinical Features to Scale
        scale_cols = ["Age", "Baseline_FVC", "Baseline_Percent"]
        self.scaler_clinical.fit(df_train[scale_cols])

        train_clin_scaled = self.scaler_clinical.transform(df_train[scale_cols])
        val_clin_scaled = self.scaler_clinical.transform(df_val[scale_cols])
        test_clin_scaled = self.scaler_clinical.transform(df_test[scale_cols])

        # Categorical Features
        cat_cols = ["Sex", "SmokingStatus"]
        self.ohe.fit(df_train[cat_cols])

        train_cat = self.ohe.transform(df_train[cat_cols])
        val_cat = self.ohe.transform(df_val[cat_cols])
        test_cat = self.ohe.transform(df_test[cat_cols])

        self.is_fitted = True

        # 6. Construct Feature Sets
        def construct_matrices(
            df, pca_feats, rad_feats, clin_scaled, cat_feats, is_test=False
        ):
            # Static Features: [Clinical_Scaled, Categorical, PCA, Radiomics]
            X_static = np.hstack([clin_scaled, cat_feats, pca_feats, rad_feats])

            # Time Variable: Weeks
            weeks = df["Weeks"].values.reshape(-1, 1)

            # --- FVC Model Features ---
            # Linear Interaction: [X_static, Weeks, X_static * Weeks]
            # This models: FVC = Intercept + Beta*X + Gamma*t + Delta*(X*t)
            # i.e. Slope depends on X
            interaction = X_static * weeks
            X_fvc = np.hstack([X_static, weeks, interaction])

            # --- Uncertainty Model Features ---
            # Horizon: |Weeks - Baseline_Weeks|
            horizon = np.abs(df["Weeks"] - df["Baseline_Weeks"]).values.reshape(-1, 1)
            # Simple concatenation for uncertainty (Parsimony)
            X_unc = np.hstack([X_static, horizon])

            if is_test:
                return X_fvc, X_unc, df["Patient_Week"].values
            else:
                y = df["FVC"].values
                return X_fvc, y, X_unc

        print("Constructing final feature matrices...")
        train_X_fvc, train_y, train_X_unc = construct_matrices(
            df_train, train_pca, train_rad, train_clin_scaled, train_cat
        )
        val_X_fvc, val_y, val_X_unc = construct_matrices(
            df_val, val_pca, val_rad, val_clin_scaled, val_cat
        )
        test_X_fvc, test_X_unc, test_pw = construct_matrices(
            df_test, test_pca, test_rad, test_clin_scaled, test_cat, is_test=True
        )

        # 7. Save to Cache
        print(f"Saving processed data to {cache_path}...")
        try:
            np.savez(
                cache_path,
                train_X_fvc=train_X_fvc,
                train_y=train_y,
                train_X_unc=train_X_unc,
                val_X_fvc=val_X_fvc,
                val_y=val_y,
                val_X_unc=val_X_unc,
                test_X_fvc=test_X_fvc,
                test_X_unc=test_X_unc,
                test_patient_weeks=test_pw,
            )
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return {
            "train": {"X_fvc": train_X_fvc, "y": train_y, "X_unc": train_X_unc},
            "val": {"X_fvc": val_X_fvc, "y": val_y, "X_unc": val_X_unc},
            "test": {
                "X_fvc": test_X_fvc,
                "X_unc": test_X_unc,
                "patient_weeks": test_pw,
            },
        }
