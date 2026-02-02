import os
from library.config import Config
from library.features import ImageFeatureExtractor
from library.preprocessing import preprocess_data


class DataManager:
    """
    Orchestrates the data loading, feature extraction, and preprocessing pipeline.
    Integrates the ImageFeatureExtractor and HighPrecisionTransformer to provide
    ready-to-use data for the OAS Linear Discriminant model.
    """

    def __init__(self, metadata_dir=Config.METADATA_DIR):
        """
        Initialize the DataManager.

        Args:
            metadata_dir (str): Directory containing the train/val/test metadata CSVs.
        """
        self.metadata_dir = metadata_dir
        self.extractor = ImageFeatureExtractor()

    def load_all_data(self, load_cached_data=True, max_samples=None):
        """
        Loads Train, Validation, and Test datasets.
        Performs feature extraction (merging tabular + geometric) and preprocessing
        (Yeo-Johnson + StandardScaler).

        Args:
            load_cached_data (bool): Whether to use cached intermediate files.
            max_samples (int, optional): Limit number of samples for debugging.

        Returns:
            tuple: ((X_train, y_train, ids_train),
                    (X_val, y_val, ids_val),
                    (X_test, ids_test))
        """
        # Define metadata paths
        train_meta = os.path.join(self.metadata_dir, "train.csv")
        val_meta = os.path.join(self.metadata_dir, "val.csv")
        test_meta = os.path.join(self.metadata_dir, "test.csv")

        # Determine cache names based on whether we are debugging (max_samples set)
        # This prevents overwriting full dataset caches with partial data
        suffix = ""
        if max_samples is not None:
            suffix = "_debug"

        train_name = f"train_features{suffix}"
        val_name = f"val_features{suffix}"
        test_name = f"test_features{suffix}"
        preprocess_name = f"preprocessed_data{suffix}"

        # 1. Extract/Load Raw Features (Tabular + Geometric)
        # The extractor handles caching internally based on the dataset_name provided
        print(f"Loading/Extracting features (max_samples={max_samples})...")

        X_train_raw, y_train, ids_train = self.extractor.process_data(
            train_meta,
            dataset_name=train_name,
            load_cached_data=load_cached_data,
            max_samples=max_samples,
        )

        X_val_raw, y_val, ids_val = self.extractor.process_data(
            val_meta,
            dataset_name=val_name,
            load_cached_data=load_cached_data,
            max_samples=max_samples,
        )

        X_test_raw, _, ids_test = self.extractor.process_data(
            test_meta,
            dataset_name=test_name,
            load_cached_data=load_cached_data,
            max_samples=max_samples,
        )

        # 2. Preprocess Features (Yeo-Johnson + Standard Scaling)
        # The preprocessor fits on Train and transforms Val/Test
        # It also handles caching of the transformed numpy arrays
        print("Preprocessing features...")

        X_train, X_val, X_test = preprocess_data(
            X_train_raw,
            X_val_raw,
            X_test_raw,
            cache_name=preprocess_name,
            load_cached_data=load_cached_data,
        )

        # Return structured data
        return (
            (X_train, y_train, ids_train),
            (X_val, y_val, ids_val),
            (X_test, ids_test),
        )
