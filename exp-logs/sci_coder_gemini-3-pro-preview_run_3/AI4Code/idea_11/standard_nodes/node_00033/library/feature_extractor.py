import os
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import torch

from library.config import Config
from library.data_loader import NotebookTextLoader
from library.backbone import BackboneTrainer


class FeatureEngineer:
    """
    Generates features for the Multi-Scale Structural Heatmap Regressor.
    Orchestrates backbone encoding, similarity computation, smoothing, and feature extraction.
    """

    def __init__(self, backbone_model=None):
        """
        Args:
            backbone_model (BackboneTrainer, optional): Instance of the backbone trainer.
                                                        If None, a new one is instantiated.
        """
        if backbone_model is None:
            self.backbone = BackboneTrainer()
        else:
            self.backbone = backbone_model

    def extract_features(self, metadata_path, mode="train", load_cached_data=True):
        """
        Main method to extract features for a dataset defined by metadata_path.
        Handles caching logic.

        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'. Determines cache location.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: DataFrame containing features and targets (if available).
        """
        # Determine cache path
        if mode == "train":
            cache_path = Config.TRAIN_FEATURES_PATH
        elif mode == "val":
            cache_path = Config.VAL_FEATURES_PATH
        elif mode == "test":
            cache_path = Config.TEST_FEATURES_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {mode} features from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Computing {mode} features from scratch...")

        # 2. Load Data
        loader = NotebookTextLoader(metadata_path)

        # 3. Process Notebooks
        features_list = []

        # Iterate through all notebooks
        # Note: We process notebook by notebook to maintain context boundaries
        for i in range(len(loader)):
            nb_data = loader[i]
            nb_features = self._process_single_notebook(nb_data, mode)
            if nb_features:
                features_list.extend(nb_features)

        # 4. Create DataFrame
        features_df = pd.DataFrame(features_list)

        # 5. Save to Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        features_df.to_parquet(cache_path, index=False)
        print(f"Saved {len(features_df)} rows to cache: {cache_path}")

        return features_df

    def _process_single_notebook(self, nb_data, mode):
        """
        Extracts features for a single notebook.

        Args:
            nb_data (dict): Dictionary from NotebookTextLoader.
            mode (str): 'train', 'val', or 'test'.

        Returns:
            list of dict: List of feature dictionaries (one per markdown cell).
        """
        notebook_id = nb_data["id"]
        code_cells = nb_data["code_cells"]
        markdown_cells = nb_data["markdown_cells"]
        cell_order = nb_data["cell_order"]

        n_code = len(code_cells)
        n_md = len(markdown_cells)

        # Edge case: If no code cells, we cannot position markdown relative to code.
        # In a real scenario, these might be placed at the end or start.
        # For this pipeline, we skip them or handle them minimally.
        if n_code == 0:
            return []

        if n_md == 0:
            return []

        # 1. Encode Cells
        # Extract texts
        code_texts = [c[1] for c in code_cells]
        md_texts = [m[1] for m in markdown_cells]

        # Batch encode all cells in the notebook to utilize GPU efficiently
        # We concatenate to do one pass if memory allows, or just two calls.
        # Given typical notebook sizes, two calls is fine.
        code_embeddings = self.backbone.encode(code_texts, show_progress_bar=False)
        md_embeddings = self.backbone.encode(md_texts, show_progress_bar=False)

        # 2. Compute Similarity Matrix
        # Shape: (n_md, n_code)
        # Cosine similarity is dot product of normalized vectors
        similarity_matrix = np.matmul(md_embeddings, code_embeddings.T)

        # 3. Prepare Target Mapping (if train/val)
        md_targets = {}
        if mode in ["train", "val"] and cell_order:
            # Filter ground truth to only code cells to establish the coordinate system
            code_order_ids = [
                cid for cid in cell_order if cid in {c[0] for c in code_cells}
            ]

            # Map code cell ID to its rank (0 to n_code-1)
            # However, the code cells in `code_cells` (from JSON) are ALREADY in correct order.
            # So `code_cells[i]` is at rank `i`.
            # We need to find where markdown fits.

            # Efficient target calculation:
            # Iterate through cell_order. Keep a counter of code cells seen so far.
            # When a markdown cell is found, its rank is the current counter.
            current_rank = 0
            for cid in cell_order:
                if cid in {c[0] for c in code_cells}:
                    current_rank += 1
                elif cid in {m[0] for m in markdown_cells}:
                    md_targets[cid] = current_rank

        # 4. Extract Features per Markdown Cell
        notebook_features = []

        for i, (md_id, md_text) in enumerate(markdown_cells):
            # Base features
            row = {
                "id": notebook_id,
                "cell_id": md_id,
                "n_code": n_code,
                "md_len": len(md_text),
            }

            # Raw similarity vector for this markdown cell
            sim_vector = similarity_matrix[i]  # Shape (n_code,)

            # Multi-Scale Smoothing and Feature Extraction
            for k in Config.SMOOTHING_SCALES:
                # Apply 1D convolution (smoothing)
                # mode='nearest' handles boundaries by replicating the edge value
                if n_code >= k:
                    # Use a uniform kernel of size k
                    kernel = np.ones(k) / k
                    smoothed = convolve1d(sim_vector, kernel, mode="nearest")
                else:
                    # Fallback if n_code is smaller than kernel
                    smoothed = sim_vector

                # Spatial Anchors
                # 1. Max Location (normalized)
                max_idx = np.argmax(smoothed)
                row[f"sim_k{k}_max_loc"] = max_idx / n_code
                row[f"sim_k{k}_max_val"] = smoothed[max_idx]

                # 2. Center of Mass (normalized)
                # sum(i * w_i) / sum(w_i)
                # We clip values to be non-negative for center of mass calculation to avoid issues
                weights = np.maximum(smoothed, 0)
                total_weight = np.sum(weights)
                if total_weight > 1e-6:
                    com = np.sum(np.arange(n_code) * weights) / total_weight
                    row[f"sim_k{k}_mean_loc"] = com / n_code
                else:
                    row[f"sim_k{k}_mean_loc"] = 0.5  # Default to middle

                row[f"sim_k{k}_mean_val"] = np.mean(smoothed)

                # Structural Heatmap (Specific to k=3 as per design)
                if k == 3:
                    heatmap = self._generate_heatmap(smoothed, Config.NUM_BINS)
                    for bin_idx, val in enumerate(heatmap):
                        row[f"heatmap_{bin_idx}"] = val

            # Add Target
            if mode in ["train", "val"]:
                if md_id in md_targets:
                    # Normalize target to [0, 1]
                    # Target is number of code cells before. Max is n_code.
                    row["target"] = md_targets[md_id] / n_code
                else:
                    # Should not happen if data integrity is good
                    row["target"] = 0.0

            notebook_features.append(row)

        return notebook_features

    def _generate_heatmap(self, vector, target_len):
        """
        Resamples a vector to a fixed target length using linear interpolation.

        Args:
            vector (np.ndarray): Input vector of shape (n,).
            target_len (int): Desired output length.

        Returns:
            np.ndarray: Resampled vector of shape (target_len,).
        """
        n = len(vector)
        if n == target_len:
            return vector

        if n < 2:
            # If vector is too short (length 1), repeat the value
            return np.full(target_len, vector[0])

        # Create coordinate systems
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)

        # Interpolate
        f = interp1d(x_old, vector, kind="linear", fill_value="extrapolate")
        resampled = f(x_new)

        return resampled
