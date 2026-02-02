import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import Config
from library.feature_extractor import generate_embeddings


class DataProcessor:
    """
    Manages data loading, preprocessing, feature engineering, and caching.
    Implements the pipeline to transform raw metadata and images into
    feature matrices for the Quantile-Elastic pipeline.
    """

    def __init__(self):
        # Initialize transformers with fixed seeds for reproducibility
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        self.scaler_num = StandardScaler()
        # Handle unknown categories gracefully (e.g., if test set has a new category, though unlikely here)
        self.ohe_sex = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe_smoke = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.fitted = False

    def _get_cache_paths(self):
        """Returns a dictionary of file paths for cached data."""
        files = [
            "X_static_train",
            "X_inter_train",
            "y_train",
            "X_static_val",
            "X_inter_val",
            "y_val",
            "X_static_test",
            "X_inter_test",
            "test_ids",
        ]
        return {name: os.path.join(Config.WORKING_DIR, f"{name}.npy") for name in files}

    def _augment_baseline_data(self, df):
        """
        For training/validation data, identifies the baseline measurement (earliest week)
        for each patient and propagates those features to all rows.
        This aligns the training data structure with the test data structure.
        """
        # Sort by Patient and Weeks to ensure first row is the earliest visit
        df = df.sort_values(["Patient", "Weeks"])

        # Columns that define the baseline state
        baseline_cols = ["FVC", "Percent", "Weeks"]

        # Group by patient and take the first row (baseline visit)
        baseline_df = df.groupby("Patient")[baseline_cols].first().reset_index()

        # Rename columns to indicate they are baseline features
        baseline_df = baseline_df.rename(
            columns={
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Weeks": "Baseline_Weeks",
            }
        )

        # Merge baseline info back into the original dataframe
        merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

        return merged_df

    def _process_tabular_features(self, df, is_train=False):
        """
        Encodes categorical features and normalizes numerical features.
        Returns the dense static feature matrix (excluding PCA).
        """
        # 1. Categorical Features: Sex and SmokingStatus
        if is_train:
            sex_encoded = self.ohe_sex.fit_transform(df[["Sex"]])
            smoke_encoded = self.ohe_smoke.fit_transform(df[["SmokingStatus"]])
        else:
            sex_encoded = self.ohe_sex.transform(df[["Sex"]])
            smoke_encoded = self.ohe_smoke.transform(df[["SmokingStatus"]])

        # 2. Numerical Features: Age, Baseline FVC, Baseline Percent
        # Note: We do NOT scale the target 'Weeks' here; it is handled in interaction generation
        num_data = df[["Age", "Baseline_FVC", "Baseline_Percent"]].values

        if is_train:
            num_scaled = self.scaler_num.fit_transform(num_data)
        else:
            num_scaled = self.scaler_num.transform(num_data)

        # Concatenate numerical and categorical features
        static_feats = np.hstack([num_scaled, sex_encoded, smoke_encoded])
        return static_feats

    def _create_interaction_features(self, X_static, weeks, baseline_weeks):
        """
        Creates the interaction feature set for the Varying-Coefficient model.
        Features: [X_static, t, X_static * t]
        where t (time) = weeks - baseline_weeks
        """
        # Calculate relative time from baseline
        t = (weeks - baseline_weeks).reshape(-1, 1)

        # Compute interaction terms (element-wise multiplication)
        # This allows the slope of decline to depend on static characteristics
        X_interaction = X_static * t

        # Concatenate: Static features, Time variable, and Interaction terms
        return np.hstack([X_static, t, X_interaction])

    def process_data(self, load_cached_data=Config.LOAD_CACHED_DATA):
        """
        Main pipeline execution method.
        1. Checks cache.
        2. Loads metadata.
        3. Generates/Loads visual embeddings.
        4. Performs PCA and Feature Engineering.
        5. Caches and returns results.

        Returns:
            dict: Dictionary containing numpy arrays for X_static, X_inter, y, etc.
        """
        cache_paths = self._get_cache_paths()
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # --- 1. Load Metadata ---
        print("Loading metadata...")
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        if Config.DEBUG:
            print(f"DEBUG MODE: Truncating data to {Config.DEBUG_SAMPLE_SIZE} samples.")
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
            # We generally keep test intact or sample it if strictly debugging pipeline flow
            # test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # --- 2. Try Loading Cache ---
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in cache_paths.values())
            if all_exist:
                print("Checking cached data...")
                try:
                    data = {
                        name: np.load(path, allow_pickle=True)
                        for name, path in cache_paths.items()
                    }

                    # Verify consistency with current metadata
                    if len(data["y_train"]) == len(train_df) and len(
                        data["y_val"]
                    ) == len(val_df):
                        print(
                            "Cache matches current configuration. Loading processed data..."
                        )
                        return data
                    else:
                        print(
                            f"Cache size mismatch (Train: {len(data['y_train'])} vs {len(train_df)}). Recomputing..."
                        )
                except Exception as e:
                    print(f"Error loading cache: {e}. Recomputing from scratch.")
            else:
                print("Cache incomplete or missing. Recomputing from scratch.")

        # --- 3. Augment Train/Val with Baseline Info ---
        # Test data already has Baseline_FVC/_Weeks/_Percent columns from the metadata generation step.
        # We need to create these for Train/Val to ensure feature consistency.
        train_df = self._augment_baseline_data(train_df)
        val_df = self._augment_baseline_data(val_df)

        # --- 4. Generate Visual Embeddings ---
        # This step uses the feature_extractor library which handles its own caching of the raw embeddings.
        print("Generating/Loading visual embeddings...")
        train_emb = generate_embeddings(
            train_df, "train_features.npy", load_cached_data
        )
        val_emb = generate_embeddings(val_df, "val_features.npy", load_cached_data)
        test_emb = generate_embeddings(test_df, "test_features.npy", load_cached_data)

        # --- 5. Dimensionality Reduction (PCA) ---
        print(f"Fitting PCA with {Config.PCA_COMPONENTS} components...")
        # Fit only on training data to prevent data leakage
        train_pca = self.pca.fit_transform(train_emb)
        val_pca = self.pca.transform(val_emb)
        test_pca = self.pca.transform(test_emb)

        # --- 6. Process Tabular Features (Static) ---
        print("Processing tabular features...")
        # Fit scalers/encoders on training data only
        train_tabular = self._process_tabular_features(train_df, is_train=True)
        val_tabular = self._process_tabular_features(val_df, is_train=False)
        test_tabular = self._process_tabular_features(test_df, is_train=False)

        # Combine PCA features with Tabular features to create the Static Vector
        X_static_train = np.hstack([train_pca, train_tabular])
        X_static_val = np.hstack([val_pca, val_tabular])
        X_static_test = np.hstack([test_pca, test_tabular])

        # --- 7. Create Interaction Features ---
        print("Creating interaction features for FVC model...")
        X_inter_train = self._create_interaction_features(
            X_static_train, train_df["Weeks"].values, train_df["Baseline_Weeks"].values
        )
        X_inter_val = self._create_interaction_features(
            X_static_val, val_df["Weeks"].values, val_df["Baseline_Weeks"].values
        )
        X_inter_test = self._create_interaction_features(
            X_static_test, test_df["Weeks"].values, test_df["Baseline_Weeks"].values
        )

        # --- 8. Extract Targets and IDs ---
        y_train = train_df["FVC"].values
        y_val = val_df["FVC"].values
        test_ids = test_df["Patient_Week"].values

        # --- 9. Save to Cache ---
        data = {
            "X_static_train": X_static_train,
            "X_inter_train": X_inter_train,
            "y_train": y_train,
            "X_static_val": X_static_val,
            "X_inter_val": X_inter_val,
            "y_val": y_val,
            "X_static_test": X_static_test,
            "X_inter_test": X_inter_test,
            "test_ids": test_ids,
        }

        print("Saving processed data to cache...")
        for name, array in data.items():
            np.save(cache_paths[name], array)

        return data
