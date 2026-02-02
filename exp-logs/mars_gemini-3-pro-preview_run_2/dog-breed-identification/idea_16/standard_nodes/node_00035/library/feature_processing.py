import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import library.config as config


class FeaturePipeline:
    """
    Manages the statistical normalization and fusion of multi-view features.
    Ensures Stage 3 features are conditioned (Z-score normalized) while Stage 4
    features (LayerNormed) are preserved.
    """

    def __init__(self):
        # Dictionary to store fitted StandardScalers for each view's Stage 3 features
        self.scalers = {}
        # strict order for fusion as per strategy
        self.view_order = ["global", "standard", "local"]

    def _align_data(self, data_dict):
        """
        Aligns features from different views based on image IDs.
        Crucial if data loaders were shuffled differently during extraction.

        Args:
            data_dict: Dict {view_name: (s3, s4, ids, labels)}

        Returns:
            aligned_dict: {
                'views': {view: {'s3': arr, 's4': arr}},
                'ids': sorted_ids_arr,
                'labels': sorted_labels_arr
            }
        """
        views = list(data_dict.keys())

        # Find intersection of IDs across all views
        common_ids = None
        for v in views:
            curr_ids = data_dict[v][2]  # Index 2 is ids
            if common_ids is None:
                common_ids = set(curr_ids)
            else:
                common_ids = common_ids.intersection(curr_ids)

        if not common_ids:
            raise ValueError("No common IDs found across views.")

        # Sort IDs to ensure deterministic order
        sorted_ids = np.array(sorted(list(common_ids)))

        aligned_views = {}
        labels = None

        # Reorder arrays
        for v in views:
            s3, s4, ids, y = data_dict[v]

            # Map ID to index for fast lookup
            id_to_idx = {id_: i for i, id_ in enumerate(ids)}

            # Get indices in correct order
            indices = [id_to_idx[uid] for uid in sorted_ids]

            aligned_views[v] = {"s3": s3[indices], "s4": s4[indices]}

            # Capture labels from the first view processed
            if labels is None:
                labels = y[indices]

        return {"views": aligned_views, "ids": sorted_ids, "labels": labels}

    def _fit_transform_s3(self, s3_dict):
        """Fits scalers on Stage 3 features and transforms them."""
        norm_dict = {}
        for view, features in s3_dict.items():
            scaler = StandardScaler()
            norm_dict[view] = scaler.fit_transform(features)
            self.scalers[view] = scaler
        return norm_dict

    def _transform_s3(self, s3_dict):
        """Transforms Stage 3 features using existing scalers."""
        norm_dict = {}
        for view, features in s3_dict.items():
            if view not in self.scalers:
                raise ValueError(f"Scaler for view '{view}' has not been fitted.")
            norm_dict[view] = self.scalers[view].transform(features)
        return norm_dict

    def _fuse(self, s4_dict, s3_norm_dict):
        """
        Concatenates features in the specific order:
        [Global_S4, Global_S3_Norm, Standard_S4, Standard_S3_Norm, Local_S4, Local_S3_Norm]
        """
        parts = []
        for view in self.view_order:
            # Ensure view exists in both dictionaries
            if view in s4_dict and view in s3_norm_dict:
                parts.append(s4_dict[view])
                parts.append(s3_norm_dict[view])
            else:
                # This might happen if a view was missing from input
                print(f"Warning: View '{view}' missing during fusion.")

        return np.hstack(parts)

    def normalize_and_fuse(self, raw_data_dict, is_train=False):
        """
        Main pipeline method for a single dataset split.

        Args:
            raw_data_dict: Dict {view_name: (s3, s4, ids, labels)}
            is_train: Bool, whether to fit scalers (True) or just transform (False)

        Returns:
            X_fused, ids, labels
        """
        # 1. Align
        aligned = self._align_data(raw_data_dict)

        # 2. Extract S3 and Normalize
        s3_features = {v: d["s3"] for v, d in aligned["views"].items()}

        if is_train:
            s3_norm = self._fit_transform_s3(s3_features)
        else:
            s3_norm = self._transform_s3(s3_features)

        # 3. Extract S4 and Fuse
        s4_features = {v: d["s4"] for v, d in aligned["views"].items()}
        X_fused = self._fuse(s4_features, s3_norm)

        return X_fused, aligned["ids"], aligned["labels"]


def process_features(raw_train, raw_val, raw_test, load_cached_data=True):
    """
    Orchestrates the feature processing pipeline with caching.

    Args:
        raw_train, raw_val, raw_test: Dictionaries containing raw features for each view.
            Format: { 'global': (s3, s4, ids, labels), 'standard': ..., 'local': ... }
        load_cached_data: Boolean to enable/disable loading from disk.

    Returns:
        (train_X, train_ids, train_y), (val_X, val_ids, val_y), (test_X, test_ids, test_y)
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    splits = ["train", "val", "test"]
    cache_files = {}
    for split in splits:
        cache_files[split] = {
            "X": os.path.join(cache_dir, f"{split}_fused_X.npy"),
            "ids": os.path.join(cache_dir, f"{split}_fused_ids.npy"),
            "y": os.path.join(cache_dir, f"{split}_fused_y.npy"),
        }

    # 1. Check Cache
    if load_cached_data:
        all_exist = True
        for split in splits:
            for key in ["X", "ids", "y"]:
                if not os.path.exists(cache_files[split][key]):
                    all_exist = False
                    break

        if all_exist:
            print("Loading fused features from cache...")
            results = {}
            for split in splits:
                X = np.load(cache_files[split]["X"])
                ids = np.load(cache_files[split]["ids"])
                y = np.load(cache_files[split]["y"])
                results[split] = (X, ids, y)
            return results["train"], results["val"], results["test"]

    # 2. Process from Scratch
    print("Processing and fusing features (Normalization + Concatenation)...")
    pipeline = FeaturePipeline()

    # Train (Fits scalers)
    print("  Processing Train set...")
    train_X, train_ids, train_y = pipeline.normalize_and_fuse(raw_train, is_train=True)

    # Val (Transforms)
    print("  Processing Validation set...")
    val_X, val_ids, val_y = pipeline.normalize_and_fuse(raw_val, is_train=False)

    # Test (Transforms)
    print("  Processing Test set...")
    test_X, test_ids, test_y = pipeline.normalize_and_fuse(raw_test, is_train=False)

    # 3. Save to Cache
    print(f"Saving fused features to {cache_dir}...")

    # Helper to save
    def save_split(split_name, X, ids, y):
        np.save(cache_files[split_name]["X"], X)
        np.save(cache_files[split_name]["ids"], ids)
        np.save(cache_files[split_name]["y"], y)

    save_split("train", train_X, train_ids, train_y)
    save_split("val", val_X, val_ids, val_y)
    save_split("test", test_X, test_ids, test_y)

    return (
        (train_X, train_ids, train_y),
        (val_X, val_ids, val_y),
        (test_X, test_ids, test_y),
    )
