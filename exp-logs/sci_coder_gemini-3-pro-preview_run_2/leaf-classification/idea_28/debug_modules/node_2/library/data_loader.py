import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    ALL_PROVIDED_FEATURES,
    ID_COL,
    TARGET_COL,
)
from library.feature_engineering import extract_geometric_features


class DataManager:
    """
    Manages data ingestion, feature extraction, view construction, and preprocessing
    for the Multi-Resolution Precision-Generative Ensemble.
    """

    def __init__(self):
        self.working_dir = WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        self.cache_path = os.path.join(self.working_dir, "processed_data.npz")

    def load_data(self, load_cached_data=True):
        """
        Loads and preprocesses data for all resolution views.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed data from disk.

        Returns:
            dict: A dictionary containing:
                - '{view}_X_train', '{view}_X_val', '{view}_X_test': Preprocessed feature matrices (float64).
                - 'y_train', 'y_val': Encoded target vectors.
                - 'test_ids': Array of image IDs for the test set.
                - 'classes': List of class names corresponding to encoded targets.

                Where {view} is one of ['macro', 'micro', 'synergistic'].
        """
        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading processed data from {self.cache_path}")
            try:
                with np.load(self.cache_path, allow_pickle=True) as data:
                    return {key: data[key] for key in data.files}
            except Exception as e:
                print(f"Cache load failed ({e}). Recomputing from scratch...")

        print("Processing data from scratch...")

        # 2. Load Metadata
        if not all(os.path.exists(p) for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]):
            raise FileNotFoundError(
                "Metadata files not found. Ensure ./metadata/ exists."
            )

        df_train = pd.read_csv(TRAIN_PATH)
        df_val = pd.read_csv(VAL_PATH)
        df_test = pd.read_csv(TEST_PATH)

        # 3. Construct Views

        # --- View A: Macro (Geometric Features) ---
        print("Generating Macro-Resolution View (Geometric)...")
        macro_train = extract_geometric_features(
            df_train, load_cached_data=load_cached_data, cache_name="macro_train"
        )
        macro_val = extract_geometric_features(
            df_val, load_cached_data=load_cached_data, cache_name="macro_val"
        )
        macro_test = extract_geometric_features(
            df_test, load_cached_data=load_cached_data, cache_name="macro_test"
        )

        # --- View B: Micro (Provided Features) ---
        print("Generating Micro-Resolution View (Texture/Margin/Shape)...")
        micro_train = df_train[ALL_PROVIDED_FEATURES]
        micro_val = df_val[ALL_PROVIDED_FEATURES]
        micro_test = df_test[ALL_PROVIDED_FEATURES]

        # --- View C: Synergistic (Concatenated) ---
        print("Generating Synergistic View...")
        # Concatenate along columns; indices are aligned by virtue of source dataframes
        syn_train = pd.concat([macro_train, micro_train], axis=1)
        syn_val = pd.concat([macro_val, micro_val], axis=1)
        syn_test = pd.concat([macro_test, micro_test], axis=1)

        # 4. Target Encoding
        print("Encoding Targets...")
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[TARGET_COL])
        y_val = le.transform(df_val[TARGET_COL])
        classes = le.classes_
        test_ids = df_test[ID_COL].values

        # 5. Preprocessing (Precision & Normalization)
        views = {
            "macro": (macro_train, macro_val, macro_test),
            "micro": (micro_train, micro_val, micro_test),
            "synergistic": (syn_train, syn_val, syn_test),
        }

        final_data = {
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
            "classes": classes,
        }

        for view_name, (train_df, val_df, test_df) in views.items():
            print(f"Preprocessing View: {view_name}")

            # Cast to float64 for numerical stability in LDA/QDA
            X_train = train_df.values.astype(np.float64)
            X_val = val_df.values.astype(np.float64)
            X_test = test_df.values.astype(np.float64)

            # Handle potential NaNs (though feature engineering should have filled them)
            if np.isnan(X_train).any():
                print(f"  Warning: NaNs detected in {view_name}. Imputing with mean.")
                col_mean = np.nanmean(X_train, axis=0)
                # Fill NaNs
                inds = np.where(np.isnan(X_train))
                X_train[inds] = np.take(col_mean, inds[1])
                inds_val = np.where(np.isnan(X_val))
                X_val[inds_val] = np.take(col_mean, inds_val[1])
                inds_test = np.where(np.isnan(X_test))
                X_test[inds_test] = np.take(col_mean, inds_test[1])

            # Apply PowerTransformer (Yeo-Johnson)
            # This handles skewness and forces Gaussian-like distribution
            pt = PowerTransformer(method="yeo-johnson", standardize=True)

            # Fit ONLY on training data to prevent leakage
            X_train_pt = pt.fit_transform(X_train)
            X_val_pt = pt.transform(X_val)
            X_test_pt = pt.transform(X_test)

            # Store results
            final_data[f"{view_name}_X_train"] = X_train_pt
            final_data[f"{view_name}_X_val"] = X_val_pt
            final_data[f"{view_name}_X_test"] = X_test_pt

        # 6. Save to Cache
        print(f"Saving processed data to {self.cache_path}")
        np.savez_compressed(self.cache_path, **final_data)

        return final_data
