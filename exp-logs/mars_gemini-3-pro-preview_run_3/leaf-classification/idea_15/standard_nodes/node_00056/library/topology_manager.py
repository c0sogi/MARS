import os
import numpy as np
from library.config import Config


class TopologyTransformer:
    """
    Manages the topological transformations of the feature space.
    Implements Hyper-Densification for training and Canonical Centroid generation for inference.
    """

    def __init__(self):
        """
        Initialize the transformer with the working directory from Config.
        """
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def densify_training_data(
        self,
        dino_feats,
        conv_feats,
        tab_feats,
        labels,
        ids,
        load_cached_data=True,
        cache_prefix="train_densified",
    ):
        """
        Applies Hyper-Densification: Generates 9 orthogonal centroids per image.
        Increases dataset size by 9x to provide a dense estimate for LDA covariance.

        Args:
            dino_feats (np.ndarray): Raw DINOv2 features (N, 36, D_dino).
            conv_feats (np.ndarray): Raw ConvNeXt features (N, 36, D_conv).
            tab_feats (np.ndarray): Tabular features (N, D_tab).
            labels (np.ndarray): Target labels (N,).
            ids (np.ndarray): Image IDs (N,).
            load_cached_data (bool): Whether to attempt loading from cache.
            cache_prefix (str): Prefix for generated cache files.

        Returns:
            dict: Dictionary containing densified arrays with keys:
                  'dino', 'convnext', 'tabular', 'labels', 'ids'.
                  Visual features will have shape (9*N, D).
        """
        # Define cache paths
        cache_paths = {
            "dino": os.path.join(self.working_dir, f"{cache_prefix}_dino.npy"),
            "convnext": os.path.join(self.working_dir, f"{cache_prefix}_convnext.npy"),
            "tabular": os.path.join(self.working_dir, f"{cache_prefix}_tabular.npy"),
            "labels": os.path.join(self.working_dir, f"{cache_prefix}_labels.npy"),
            "ids": os.path.join(self.working_dir, f"{cache_prefix}_ids.npy"),
        }

        # Check if all cache files exist
        if load_cached_data and all(os.path.exists(p) for p in cache_paths.values()):
            return {k: np.load(p, allow_pickle=True) for k, p in cache_paths.items()}

        # Container lists for the augmented data
        dino_list = []
        conv_list = []
        tab_list = []
        label_list = []
        id_list = []

        # Iterate through the 9 orthogonal sets defined in Config
        # Each set contains 4 indices corresponding to mutually exclusive orthogonal views
        for indices in Config.ORTHOGONAL_SETS:
            # Average the visual features across the 4 views for this set
            # Input shape: (N, 36, D) -> Select (N, 4, D) -> Mean (N, D)
            d_mean = np.mean(dino_feats[:, indices, :], axis=1)
            c_mean = np.mean(conv_feats[:, indices, :], axis=1)

            dino_list.append(d_mean)
            conv_list.append(c_mean)

            # Replicate tabular features, labels, and ids for this centroid set
            # This ensures alignment with the new visual centroids
            tab_list.append(tab_feats)
            label_list.append(labels)
            id_list.append(ids)

        # Concatenate all sets along the sample axis
        # Resulting shape will be (9*N, Feature_Dim)
        densified_data = {
            "dino": np.concatenate(dino_list, axis=0),
            "convnext": np.concatenate(conv_list, axis=0),
            "tabular": np.concatenate(tab_list, axis=0),
            "labels": np.concatenate(label_list, axis=0),
            "ids": np.concatenate(id_list, axis=0),
        }

        # Save generated data to cache
        for k, v in densified_data.items():
            np.save(cache_paths[k], v)

        return densified_data

    def create_inference_data(
        self,
        dino_feats,
        conv_feats,
        tab_feats,
        ids,
        labels=None,
        load_cached_data=True,
        cache_prefix="inference_canonical",
    ):
        """
        Generates a single Canonical Centroid per image for inference or validation.
        Uses the standard orthogonal axes (0, 90, 180, 270 degrees).

        Args:
            dino_feats (np.ndarray): Raw DINOv2 features (N, 36, D_dino).
            conv_feats (np.ndarray): Raw ConvNeXt features (N, 36, D_conv).
            tab_feats (np.ndarray): Tabular features (N, D_tab).
            ids (np.ndarray): Image IDs (N,).
            labels (np.ndarray, optional): Target labels (N,). Provided for validation sets.
            load_cached_data (bool): Whether to attempt loading from cache.
            cache_prefix (str): Prefix for generated cache files.

        Returns:
            dict: Dictionary containing canonical arrays with keys:
                  'dino', 'convnext', 'tabular', 'ids', and optionally 'labels'.
                  Visual features will have shape (N, D).
        """
        # Define cache paths
        cache_paths = {
            "dino": os.path.join(self.working_dir, f"{cache_prefix}_dino.npy"),
            "convnext": os.path.join(self.working_dir, f"{cache_prefix}_convnext.npy"),
            "tabular": os.path.join(self.working_dir, f"{cache_prefix}_tabular.npy"),
            "ids": os.path.join(self.working_dir, f"{cache_prefix}_ids.npy"),
        }
        if labels is not None:
            cache_paths["labels"] = os.path.join(
                self.working_dir, f"{cache_prefix}_labels.npy"
            )

        # Check if all cache files exist
        if load_cached_data and all(os.path.exists(p) for p in cache_paths.values()):
            return {k: np.load(p, allow_pickle=True) for k, p in cache_paths.items()}

        # Use the single Canonical Set defined in Config (typically 0, 90, 180, 270)
        indices = Config.CANONICAL_SET

        # Average visual features across the canonical views
        # Input shape: (N, 36, D) -> Select (N, 4, D) -> Mean (N, D)
        d_mean = np.mean(dino_feats[:, indices, :], axis=1)
        c_mean = np.mean(conv_feats[:, indices, :], axis=1)

        # Tabular features and IDs are passed through without replication
        canonical_data = {
            "dino": d_mean,
            "convnext": c_mean,
            "tabular": tab_feats,
            "ids": ids,
        }

        if labels is not None:
            canonical_data["labels"] = labels

        # Save generated data to cache
        for k, v in canonical_data.items():
            np.save(cache_paths[k], v)

        return canonical_data
