import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from library.config import Config


class DataProcessor:
    """
    Handles tabular data transformation, feature engineering, and dataset creation
    for the Deep-Feature Varying-Coefficient Elastic Net.
    """

    def __init__(self):
        # Define transformers for clinical features
        # We scale continuous variables and one-hot encode categorical ones
        self.numeric_features = ["Age", "Baseline_FVC", "Baseline_Percent"]
        self.categorical_features = ["Sex", "SmokingStatus"]

        # Main preprocessor for tabular data
        self.tabular_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

        # Scaler for Image PCA features (to ensure they have unit variance for ElasticNet)
        self.img_scaler = StandardScaler()

        # Scaler for the Time variable (Weeks)
        self.time_scaler = StandardScaler()

    def _add_baseline_features(self, df):
        """
        Derives baseline features for training/validation data.
        Assumes the earliest visit (min Weeks) is the baseline.
        """
        # Sort to ensure we can pick the first occurrence easily
        df = df.sort_values(["Patient", "Weeks"])

        # Identify baseline rows (first visit per patient)
        baseline_df = df.groupby("Patient").first().reset_index()

        # Extract relevant baseline columns
        cols = ["Patient", "Weeks", "FVC", "Percent"]
        baseline_df = baseline_df[cols]
        baseline_df.columns = [
            "Patient",
            "Baseline_Weeks",
            "Baseline_FVC",
            "Baseline_Percent",
        ]

        # Merge baseline info back to the full history
        df_merged = pd.merge(df, baseline_df, on="Patient", how="left")

        # Compute relative time from baseline
        df_merged["Time_Delta"] = df_merged["Weeks"] - df_merged["Baseline_Weeks"]

        return df_merged

    def _add_time_delta_test(self, df):
        """
        Computes Time_Delta for test data which already has Baseline columns.
        """
        df["Time_Delta"] = df["Weeks"] - df["Baseline_Weeks"]
        return df

    def _prepare_static_features(self, df, img_feat_dict, fit=False):
        """
        Combines processed tabular features with image features.
        """
        # 1. Process Tabular Features
        if fit:
            tab_features = self.tabular_transformer.fit_transform(df)
        else:
            tab_features = self.tabular_transformer.transform(df)

        # 2. Retrieve Image Features aligned with Dataframe rows
        # img_feat_dict is {PatientID: np.array}
        patient_ids = df["Patient"].values

        # Stack image features into a matrix (N_samples, N_PCA_Components)
        # Handle potential missing keys safely (though pipeline ensures they exist)
        img_features_list = [
            img_feat_dict.get(pid, np.zeros(Config.N_PCA_COMPONENTS))
            for pid in patient_ids
        ]
        img_features = np.vstack(img_features_list)

        # 3. Scale Image Features
        if fit:
            img_features = self.img_scaler.fit_transform(img_features)
        else:
            img_features = self.img_scaler.transform(img_features)

        # 4. Concatenate
        X_static = np.hstack([tab_features, img_features])

        return X_static

    def _create_interactions(self, X_static, t):
        """
        Creates the full feature matrix including interaction terms.
        Formula: [X_static, t, X_static * t]

        This enables the 'Varying-Coefficient' property where the slope w.r.t time
        depends on the static patient characteristics.
        """
        # Ensure t is (N, 1)
        if t.ndim == 1:
            t = t.reshape(-1, 1)

        # Interaction terms: Element-wise multiplication of static features with time
        # Broadcasts t across columns of X_static
        X_interactions = X_static * t

        # Concatenate all components
        X_full = np.hstack([X_static, t, X_interactions])

        return X_full.astype(np.float32)

    def process_data(
        self, train_img_feats, val_img_feats, test_img_feats, load_cached_data=True
    ):
        """
        Main pipeline execution method.

        Args:
            train_img_feats (dict): Mapping of PatientID -> PCA Features for training set.
            val_img_feats (dict): Mapping for validation set.
            test_img_feats (dict): Mapping for test set.
            load_cached_data (bool): Whether to load pre-computed arrays from disk.

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val), (X_test, df_test))
        """
        # Define cache paths
        cache_files = {
            "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
            "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
            "X_val": os.path.join(Config.CACHE_DIR, "X_val.npy"),
            "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
            "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        }

        # Attempt to load from cache
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            print("Loading processed tabular data from cache...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"])
            X_test = np.load(cache_files["X_test"])

            # We still need df_test for submission mapping
            df_test = pd.read_csv(Config.TEST_META_PATH)
            return (X_train, y_train), (X_val, y_val), (X_test, df_test)

        print("Processing tabular data and generating interaction features...")

        # 1. Load Metadata
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        # 2. Augment with Baseline & Time Delta
        df_train = self._add_baseline_features(df_train)
        df_val = self._add_baseline_features(df_val)
        df_test = self._add_time_delta_test(df_test)

        # 3. Extract Targets
        y_train = df_train["FVC"].values.astype(np.float32)
        y_val = df_val["FVC"].values.astype(np.float32)

        # 4. Prepare Static Features (Clinical + Image)
        # Fit scalers only on training data
        X_train_static = self._prepare_static_features(
            df_train, train_img_feats, fit=True
        )
        X_val_static = self._prepare_static_features(df_val, val_img_feats, fit=False)
        X_test_static = self._prepare_static_features(
            df_test, test_img_feats, fit=False
        )

        # 5. Prepare Time Feature
        t_train = df_train[["Time_Delta"]].values.astype(np.float32)
        t_val = df_val[["Time_Delta"]].values.astype(np.float32)
        t_test = df_test[["Time_Delta"]].values.astype(np.float32)

        # Scale Time variable
        self.time_scaler.fit(t_train)
        t_train = self.time_scaler.transform(t_train)
        t_val = self.time_scaler.transform(t_val)
        t_test = self.time_scaler.transform(t_test)

        # 6. Create Interaction Features
        X_train = self._create_interactions(X_train_static, t_train)
        X_val = self._create_interactions(X_val_static, t_val)
        X_test = self._create_interactions(X_test_static, t_test)

        # 7. Save to Cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["X_test"], X_test)

        print(f"Data processing complete. Train shape: {X_train.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, df_test)
