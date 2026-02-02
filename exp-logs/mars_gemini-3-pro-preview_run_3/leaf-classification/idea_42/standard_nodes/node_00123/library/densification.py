import numpy as np
import os
from library.utils import save_to_cache, load_from_cache


class Densifier:
    """
    Implements Convex-Hull Manifold Densification.

    This class transforms raw multi-view features into a densified representation
    suitable for Linear Discriminant Analysis. It constructs a convex hull
    around the image manifold by interpolating between orthogonal centroids.
    """

    def __init__(self, cache_subdir="idea_42"):
        self.cache_subdir = cache_subdir

    def _compute_primary_centroids(self, features_12_view):
        """
        Aggregates 12 views into 3 orthogonal centroids.

        Args:
            features_12_view: (N, 12, D) numpy array.

        Returns:
            (N, 3, D) numpy array containing centroids [C_A, C_B, C_C].

        Mapping (assuming 30-degree steps starting at 0):
            C_A: 0, 90, 180, 270 deg -> Indices [0, 3, 6, 9]
            C_B: 30, 120, 210, 300 deg -> Indices [1, 4, 7, 10]
            C_C: 60, 150, 240, 330 deg -> Indices [2, 5, 8, 11]
        """
        # Indices based on 30 degree steps
        idx_a = [0, 3, 6, 9]
        idx_b = [1, 4, 7, 10]
        idx_c = [2, 5, 8, 11]

        # Compute means along the view dimension (axis 1)
        c_a = np.mean(features_12_view[:, idx_a, :], axis=1)  # (N, D)
        c_b = np.mean(features_12_view[:, idx_b, :], axis=1)  # (N, D)
        c_c = np.mean(features_12_view[:, idx_c, :], axis=1)  # (N, D)

        return np.stack([c_a, c_b, c_c], axis=1)  # (N, 3, D)

    def _compute_synthetic_centroids(self, primary_centroids):
        """
        Computes 3 synthetic centroids via linear interpolation (MixUp).

        Args:
            primary_centroids: (N, 3, D) array -> [C_A, C_B, C_C]

        Returns:
            (N, 3, D) array -> [C_AB, C_BC, C_CA]
        """
        c_a = primary_centroids[:, 0, :]
        c_b = primary_centroids[:, 1, :]
        c_c = primary_centroids[:, 2, :]

        # Linear interpolation (midpoint)
        c_ab = 0.5 * c_a + 0.5 * c_b
        c_bc = 0.5 * c_b + 0.5 * c_c
        c_ca = 0.5 * c_c + 0.5 * c_a

        return np.stack([c_ab, c_bc, c_ca], axis=1)

    def densify_training_data(
        self,
        dino_feats,
        conv_feats,
        tab_feats,
        labels,
        ids,
        split_name="train",
        load_cached_data=True,
    ):
        """
        Generates 6x densified training data (3 Primary + 3 Synthetic centroids per image).

        Args:
            dino_feats: (N, 12, 1024)
            conv_feats: (N, 12, 1536)
            tab_feats: (N, 192)
            labels: (N,)
            ids: (N,)
            split_name: str, used for cache filename differentiation.
            load_cached_data: bool, whether to load from disk if available.

        Returns:
            Tuple of densified arrays: (dino, conv, tab, y, ids)
            Shapes: (N*6, D), (N*6, D), (N*6, 192), (N*6,), (N*6,)
        """
        # Define cache filenames
        files = {
            "dino": f"densified_{split_name}_dino.npy",
            "conv": f"densified_{split_name}_conv.npy",
            "tab": f"densified_{split_name}_tab.npy",
            "y": f"densified_{split_name}_y.npy",
            "ids": f"densified_{split_name}_ids.npy",
        }

        # 1. Attempt Load from Cache
        if load_cached_data:
            data = {}
            all_exist = True
            for k, f in files.items():
                val = load_from_cache(f, sub_dir=self.cache_subdir)
                if val is None:
                    all_exist = False
                    break
                data[k] = val

            if all_exist:
                print(
                    f"[{split_name}] Loaded densified data from cache ({self.cache_subdir})."
                )
                return data["dino"], data["conv"], data["tab"], data["y"], data["ids"]

        # 2. Compute from Scratch
        print(f"[{split_name}] Densifying training data (Convex-Hull)...")

        # Process DINOv2
        dino_primary = self._compute_primary_centroids(dino_feats)  # (N, 3, D)
        dino_synth = self._compute_synthetic_centroids(dino_primary)  # (N, 3, D)
        dino_all = np.concatenate([dino_primary, dino_synth], axis=1)  # (N, 6, D)

        # Process ConvNeXt
        conv_primary = self._compute_primary_centroids(conv_feats)
        conv_synth = self._compute_synthetic_centroids(conv_primary)
        conv_all = np.concatenate([conv_primary, conv_synth], axis=1)  # (N, 6, D)

        # Flatten Visual Features
        # Reshape (N, 6, D) -> (N*6, D). Order is row-major (Image 0 [6 views], Image 1 [6 views]...)
        N, _, D_dino = dino_all.shape
        _, _, D_conv = conv_all.shape

        dino_dense = dino_all.reshape(N * 6, D_dino)
        conv_dense = conv_all.reshape(N * 6, D_conv)

        # Replicate Tabular, Labels, IDs
        # np.repeat with axis=0 repeats rows: [row0, row0..., row1, row1...]
        # This matches the reshape order of the visual features.
        tab_dense = np.repeat(tab_feats, 6, axis=0)  # (N*6, 192)
        y_dense = np.repeat(labels, 6, axis=0)  # (N*6,)
        ids_dense = np.repeat(ids, 6, axis=0)  # (N*6,)

        # 3. Save to Cache
        save_to_cache(dino_dense, files["dino"], sub_dir=self.cache_subdir)
        save_to_cache(conv_dense, files["conv"], sub_dir=self.cache_subdir)
        save_to_cache(tab_dense, files["tab"], sub_dir=self.cache_subdir)
        save_to_cache(y_dense, files["y"], sub_dir=self.cache_subdir)
        save_to_cache(ids_dense, files["ids"], sub_dir=self.cache_subdir)

        return dino_dense, conv_dense, tab_dense, y_dense, ids_dense

    def densify_inference_data(
        self,
        dino_feats,
        conv_feats,
        tab_feats,
        ids,
        split_name="test",
        load_cached_data=True,
    ):
        """
        Generates 3x canonical inference data (3 Primary Centroids only).
        Used for test-time aggregation.

        Args:
            dino_feats: (N, 12, 1024)
            conv_feats: (N, 12, 1536)
            tab_feats: (N, 192)
            ids: (N,)
            split_name: str
            load_cached_data: bool

        Returns:
            Tuple of canonical arrays: (dino, conv, tab, ids)
            Shapes: (N*3, D), (N*3, D), (N*3, 192), (N*3,)
        """
        # Define cache filenames
        files = {
            "dino": f"canonical_{split_name}_dino.npy",
            "conv": f"canonical_{split_name}_conv.npy",
            "tab": f"canonical_{split_name}_tab.npy",
            "ids": f"canonical_{split_name}_ids.npy",
        }

        # 1. Attempt Load from Cache
        if load_cached_data:
            data = {}
            all_exist = True
            for k, f in files.items():
                val = load_from_cache(f, sub_dir=self.cache_subdir)
                if val is None:
                    all_exist = False
                    break
                data[k] = val

            if all_exist:
                print(
                    f"[{split_name}] Loaded canonical inference data from cache ({self.cache_subdir})."
                )
                return data["dino"], data["conv"], data["tab"], data["ids"]

        # 2. Compute from Scratch
        print(
            f"[{split_name}] Preparing canonical inference data (Primary Centroids)..."
        )

        # Compute Primary Centroids Only (No Synthetic for Inference)
        dino_primary = self._compute_primary_centroids(dino_feats)  # (N, 3, D)
        conv_primary = self._compute_primary_centroids(conv_feats)  # (N, 3, D)

        # Flatten
        N, _, D_dino = dino_primary.shape
        _, _, D_conv = conv_primary.shape

        dino_canon = dino_primary.reshape(N * 3, D_dino)
        conv_canon = conv_primary.reshape(N * 3, D_conv)

        # Replicate Tabular, IDs
        tab_canon = np.repeat(tab_feats, 3, axis=0)
        ids_canon = np.repeat(ids, 3, axis=0)

        # 3. Save to Cache
        save_to_cache(dino_canon, files["dino"], sub_dir=self.cache_subdir)
        save_to_cache(conv_canon, files["conv"], sub_dir=self.cache_subdir)
        save_to_cache(tab_canon, files["tab"], sub_dir=self.cache_subdir)
        save_to_cache(ids_canon, files["ids"], sub_dir=self.cache_subdir)

        return dino_canon, conv_canon, tab_canon, ids_canon
