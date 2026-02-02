import os
import numpy as np
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import save_cache, load_cache, set_seed


class StreamProcessor:
    """
    Handles Independent Component Compression (PCA) and Metadata Amplification.
    """

    def __init__(self):
        self.pca_variance = Config.PCA_VARIANCE
        self.metadata_scale = Config.METADATA_SCALE
        self.working_dir = Config.WORKING_DIR
        set_seed(Config.SEED)

    def process_features(self, features_dict, meta_dict, load_cached_data=True):
        """
        Applies PCA to each image stream, scales metadata, and concatenates them.

        Args:
            features_dict (dict): Dictionary containing raw features per split and backbone.
                                  Structure: {'train': {'swin': np.array, ...}, ...}
            meta_dict (dict): Dictionary containing metadata features per split.
                              Structure: {'train': np.array, ...}
            load_cached_data (bool): If True, attempts to load processed matrices from disk.

        Returns:
            tuple: (X_train, X_val, X_test) - Processed and concatenated feature matrices.
        """
        # Define cache filenames for the final processed matrices
        cache_paths = {
            "train": os.path.join(self.working_dir, "final_X_train.npy"),
            "val": os.path.join(self.working_dir, "final_X_val.npy"),
            "test": os.path.join(self.working_dir, "final_X_test.npy"),
        }

        # 1. Attempt to load from cache
        if load_cached_data:
            loaded_data = {}
            all_found = True
            for split, path in cache_paths.items():
                data = load_cache(path)
                if data is None:
                    all_found = False
                    break
                loaded_data[split] = data

            if all_found:
                print("StreamProcessor: Loaded processed features from cache.")
                return loaded_data["train"], loaded_data["val"], loaded_data["test"]

        print("StreamProcessor: Computing features from scratch...")

        # Initialize containers for processed feature parts
        # Each list will hold [pca_feat_1, pca_feat_2, ..., pca_feat_4, scaled_meta]
        processed_parts = {"train": [], "val": [], "test": []}

        # 2. Process Image Backbones (Independent PCA)
        # Iterate in the order defined in Config to ensure consistent concatenation
        for backbone_cfg in Config.BACKBONES:
            name = backbone_cfg["name"]
            print(f"  Processing stream: {name}")

            # Retrieve raw features
            # Note: If cache missed, we assume features_dict is populated
            train_raw = features_dict["train"][name]
            val_raw = features_dict["val"][name]
            test_raw = features_dict["test"][name]

            # Initialize PCA
            # n_components < 1.0 implies fraction of variance to retain
            pca = PCA(n_components=self.pca_variance, random_state=Config.SEED)

            # Fit on Train ONLY
            pca.fit(train_raw)

            # Transform all splits
            train_pca = pca.transform(train_raw)
            val_pca = pca.transform(val_raw)
            test_pca = pca.transform(test_raw)

            # Log reduction
            orig_dim = train_raw.shape[1]
            new_dim = train_pca.shape[1]
            print(
                f"    - Reduced dimensions from {orig_dim} to {new_dim} ({self.pca_variance*100}% variance)"
            )

            # Append to parts
            processed_parts["train"].append(train_pca)
            processed_parts["val"].append(val_pca)
            processed_parts["test"].append(test_pca)

        # 3. Process Metadata (Amplification)
        print(f"  Amplifying metadata (Scale Factor: {self.metadata_scale})")
        for split in ["train", "val", "test"]:
            meta_raw = meta_dict[split]
            # Scale binary features
            meta_scaled = meta_raw * self.metadata_scale
            processed_parts[split].append(meta_scaled)

        # 4. Concatenate and Save
        final_X = {}
        for split in ["train", "val", "test"]:
            # Concatenate along feature axis (axis 1)
            final_X[split] = np.concatenate(processed_parts[split], axis=1)

            print(f"  Final {split} shape: {final_X[split].shape}")

            # Save to cache
            save_cache(final_X[split], cache_paths[split])

        return final_X["train"], final_X["val"], final_X["test"]
