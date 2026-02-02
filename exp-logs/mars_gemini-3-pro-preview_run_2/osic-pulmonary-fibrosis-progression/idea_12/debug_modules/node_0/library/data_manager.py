import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from library.config import Config
from library.image_processing import process_patient
from library.feature_extractor import EfficientNetExtractor


class DataManager:
    """
    Manages data preparation for the MIP-Enhanced Zonal-Quantile Pipeline.
    Handles image feature extraction, tabular preprocessing, and dataset assembly.
    """

    def __init__(self):
        self.extractor = EfficientNetExtractor()
        self.cache_dir = Config.CACHE_DIR
        self.scaler_path = os.path.join(self.cache_dir, "tabular_preprocessor.pkl")

        # Define tabular preprocessor
        # Numerical: Age, Percent
        # Categorical: Sex, SmokingStatus
        # We use a ColumnTransformer to handle mixed types
        self.tabular_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["Age", "Percent"]),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["Sex", "SmokingStatus"],
                ),
            ],
            verbose_feature_names_out=False,
        )

        self.is_fitted = False

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_patient_embedding(self, patient_id, dcm_path):
        """
        Retrieves the combined feature vector (Image Embedding + Histogram) for a patient.

        Args:
            patient_id (str): Unique patient ID.
            dcm_path (str): Path to DICOM directory.

        Returns:
            np.ndarray: Concatenated vector of shape (5124,).
        """
        # 1. Get processed images and histogram
        # process_patient handles caching of the intermediate image arrays (MIP, Axials)
        data_dict = process_patient(patient_id, dcm_path, load_cached_data=True)

        # 2. Extract Visual Features using EfficientNet (5120 dim)
        visual_feats = self.extractor.extract_features(data_dict)

        # 3. Get Density Histogram (4 dim)
        hist_feats = data_dict["histogram"]

        # Concatenate
        return np.concatenate([visual_feats, hist_feats])

    def _prepare_tabular(self, df, is_train=True):
        """
        Fits (if train) and transforms tabular data.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            is_train (bool): Whether this is the training set (to fit scalers).

        Returns:
            np.ndarray: Transformed tabular features.
        """
        # Handle column naming consistency for Test set
        # Test metadata has 'Baseline_Percent' and 'Baseline_FVC'
        df_proc = df.copy()
        if "Baseline_Percent" in df_proc.columns:
            df_proc = df_proc.rename(columns={"Baseline_Percent": "Percent"})

        # Select relevant columns
        # Note: 'FVC' and 'Weeks' are handled separately as target/time variables
        cols_to_transform = ["Age", "Percent", "Sex", "SmokingStatus"]
        df_subset = df_proc[cols_to_transform]

        if is_train:
            print("Fitting tabular transformer on training data...")
            self.tabular_transformer.fit(df_subset)
            self.is_fitted = True
            # Save transformer
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.tabular_transformer, f)
        else:
            # Load transformer if not in memory
            if not self.is_fitted:
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, "rb") as f:
                        self.tabular_transformer = pickle.load(f)
                    self.is_fitted = True
                else:
                    raise ValueError(
                        "Tabular transformer not fitted and no cache found. Run prepare_dataset('train') first."
                    )

        # Transform
        features = self.tabular_transformer.transform(df_subset)
        return features.astype(np.float32)

    def prepare_dataset(self, split_type="train", load_cached_data=True):
        """
        Generates the full feature matrix X and target y for a specific split.

        Args:
            split_type (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load completed dataset from disk.

        Returns:
            dict: Contains 'X_static', 'weeks', 'y', 'patient_ids', 'base_weeks'.
        """
        save_path = os.path.join(self.cache_dir, f"dataset_{split_type}.npz")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(save_path):
            print(f"Loading cached {split_type} dataset from {save_path}...")
            loaded = np.load(save_path, allow_pickle=True)
            return {
                "X_static": loaded["X_static"],
                "weeks": loaded["weeks"],
                "y": loaded["y"],
                "patient_ids": loaded["patient_ids"],
                "base_weeks": loaded["base_weeks"],
            }

        print(f"Generating {split_type} dataset from scratch...")

        # 2. Load Metadata
        if split_type == "train":
            df = pd.read_csv(Config.TRAIN_META_PATH)
        elif split_type == "val":
            df = pd.read_csv(Config.VAL_META_PATH)
        elif split_type == "test":
            df = pd.read_csv(Config.TEST_META_PATH)
        else:
            raise ValueError(f"Unknown split type: {split_type}")

        # Debugging hook
        if Config.DEBUG:
            df = df.iloc[: Config.DEBUG_SIZE].copy()
            print(f"DEBUG MODE: Reduced {split_type} size to {len(df)}")

        # 3. Process Tabular Features
        # Only fit scalers if processing the training set
        is_train_split = split_type == "train"
        tab_features = self._prepare_tabular(df, is_train=is_train_split)

        # 4. Process Image Features
        # We iterate over unique patients first to avoid redundant inference
        unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()
        patient_emb_map = {}

        print(
            f"Extracting image features for {len(unique_patients)} unique patients..."
        )
        for _, row in unique_patients.iterrows():
            pid = row["Patient"]
            path = row["dcm_path"]
            # Get 5120 (Visual) + 4 (Hist) vector
            emb = self._get_patient_embedding(pid, path)
            patient_emb_map[pid] = emb

        # 5. Assemble Final Matrices
        X_static_list = []
        weeks_list = []
        y_list = []
        pids_list = []
        base_weeks_list = []

        # Iterate through all observations (visits)
        for idx, row in df.iterrows():
            pid = row["Patient"]

            # Retrieve cached image embedding
            img_vec = patient_emb_map.get(pid)
            if img_vec is None:
                # Fallback for safety (should not happen)
                img_vec = np.zeros(5124, dtype=np.float32)

            # Retrieve transformed tabular features
            tab_vec = tab_features[idx]

            # Concatenate Static Features: [Image(5120) | Hist(4) | Tabular(~7)]
            static_vec = np.concatenate([img_vec, tab_vec])
            X_static_list.append(static_vec)

            # Time Variable
            weeks_list.append(row["Weeks"])

            # Identifiers
            pids_list.append(pid)

            # Target Variable
            if "FVC" in row:
                y_list.append(row["FVC"])
            else:
                y_list.append(0.0)  # Placeholder for test if FVC column missing

            # Baseline Week (Auxiliary)
            if "Baseline_Weeks" in row:
                base_weeks_list.append(row["Baseline_Weeks"])
            else:
                # For train/val, Weeks is relative to baseline, so baseline is effectively 0
                base_weeks_list.append(0)

        # Convert lists to numpy arrays
        X_static = np.array(X_static_list, dtype=np.float32)
        weeks = np.array(weeks_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        base_weeks = np.array(base_weeks_list, dtype=np.float32)
        pids = np.array(pids_list)

        # 6. Save to Cache
        np.savez_compressed(
            save_path,
            X_static=X_static,
            weeks=weeks,
            y=y,
            patient_ids=pids,
            base_weeks=base_weeks,
        )

        print(f"Dataset {split_type} generated. Shape: {X_static.shape}")

        return {
            "X_static": X_static,
            "weeks": weeks,
            "y": y,
            "patient_ids": pids,
            "base_weeks": base_weeks,
        }
