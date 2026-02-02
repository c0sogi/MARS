import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SAMPLE_SUBMISSION_PATH,
    WORKING_DIR,
    ID_COL,
    TARGET_COL,
)
from library.feature_extraction import extract_morphometrics


class DataManager:
    def __init__(self):
        self.working_dir = WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        self.classes_ = None

        # Placeholders for data
        self.train_provided = None
        self.train_morph = None
        self.y_train = None

        self.val_provided = None
        self.val_morph = None
        self.y_val = None

        self.test_provided = None
        self.test_morph = None
        self.test_ids = None

    def load_all_data(self, load_cached_data=True):
        """
        Loads all data components: provided features, morphometrics, and labels.
        Uses caching to speed up reloading.
        """
        cache_path = os.path.join(self.working_dir, "data_arrays.npz")

        # 1. Load Classes from Sample Submission to ensure correct order
        sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        # Columns are id, Class1, Class2...
        self.classes_ = sample_sub.columns[1:].values

        # 2. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                self.train_provided = data["train_provided"]
                self.train_morph = data["train_morph"]
                self.y_train = data["y_train"]

                self.val_provided = data["val_provided"]
                self.val_morph = data["val_morph"]
                self.y_val = data["y_val"]

                self.test_provided = data["test_provided"]
                self.test_morph = data["test_morph"]
                self.test_ids = data["test_ids"]

                # Verify classes match
                cached_classes = data["classes"]
                if np.array_equal(self.classes_, cached_classes):
                    return
                else:
                    print("Cached classes do not match sample submission. Reloading...")
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source...")

        # 3. Load Metadata
        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        df_val = pd.read_csv(VAL_METADATA_PATH)
        df_test = pd.read_csv(TEST_METADATA_PATH)

        # 4. Extract Provided Features (Margin, Shape, Texture)
        # Columns that are not id, species, image_path
        # We assume the columns are consistent across files.
        exclude_cols = {ID_COL, TARGET_COL, "image_path"}
        feat_cols = [c for c in df_train.columns if c not in exclude_cols]

        self.train_provided = df_train[feat_cols].values.astype(np.float64)
        self.val_provided = df_val[feat_cols].values.astype(np.float64)
        self.test_provided = df_test[feat_cols].values.astype(np.float64)

        # 5. Extract Morphometrics (using library function with its own cache)
        self.train_morph = extract_morphometrics(df_train, "train", load_cached_data)
        self.val_morph = extract_morphometrics(df_val, "val", load_cached_data)
        self.test_morph = extract_morphometrics(df_test, "test", load_cached_data)

        # 6. Process Targets
        # We enforce the order from sample_submission using LabelEncoder with fixed classes
        le = LabelEncoder()
        le.fit(self.classes_)

        self.y_train = le.transform(df_train[TARGET_COL])
        self.y_val = le.transform(df_val[TARGET_COL])

        # 7. Process IDs
        self.test_ids = df_test[ID_COL].values

        # 8. Save to Cache
        np.savez(
            cache_path,
            train_provided=self.train_provided,
            train_morph=self.train_morph,
            y_train=self.y_train,
            val_provided=self.val_provided,
            val_morph=self.val_morph,
            y_val=self.y_val,
            test_provided=self.test_provided,
            test_morph=self.test_morph,
            test_ids=self.test_ids,
            classes=self.classes_,
        )

    def get_view_data(self, view_name):
        """
        Returns the feature matrices for the specified view.

        Args:
            view_name (str): 'Global' (provided features) or 'Combined' (provided + morphometrics).

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
        """
        if self.train_provided is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")

        if view_name == "Global":
            X_train = self.train_provided
            X_val = self.val_provided
            X_test = self.test_provided
        elif view_name == "Combined":
            X_train = np.hstack([self.train_provided, self.train_morph])
            X_val = np.hstack([self.val_provided, self.val_morph])
            X_test = np.hstack([self.test_provided, self.test_morph])
        else:
            raise ValueError(f"Unknown view_name: {view_name}")

        return (
            X_train.copy(),
            self.y_train.copy(),
            X_val.copy(),
            self.y_val.copy(),
            X_test.copy(),
            self.test_ids.copy(),
            self.classes_.copy(),
        )

    def preprocess_data(self, X_train, X_val, X_test):
        """
        Applies PowerTransformer (Yeo-Johnson) and casts to float64.
        Fits on Train, transforms Train, Val, Test.

        Args:
            X_train, X_val, X_test (np.ndarray): Feature matrices.

        Returns:
            tuple: (X_train_pt, X_val_pt, X_test_pt)
        """
        # Initialize PowerTransformer
        # standardize=True ensures zero mean unit variance after transformation
        pt = PowerTransformer(method="yeo-johnson", standardize=True)

        # Fit on training data only to prevent leakage
        X_train_pt = pt.fit_transform(X_train)
        X_val_pt = pt.transform(X_val)
        X_test_pt = pt.transform(X_test)

        # Enforce float64 precision
        X_train_pt = X_train_pt.astype(np.float64)
        X_val_pt = X_val_pt.astype(np.float64)
        X_test_pt = X_test_pt.astype(np.float64)

        return X_train_pt, X_val_pt, X_test_pt
