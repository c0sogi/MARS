import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_loader import get_notebook_iterator
from library.utils import set_seed


class FeatureEngineer:
    """
    Handles the extraction of semantic and structural features from notebooks
    using the fine-tuned DASR backbone.
    """

    def __init__(self):
        """
        Initializes the feature engineer and loads the semantic model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Fine-Tuned Model if available, otherwise fallback to base model
        if os.path.exists(Config.FINE_TUNED_MODEL_PATH):
            print(f"Loading fine-tuned model from {Config.FINE_TUNED_MODEL_PATH}...")
            model_path = Config.FINE_TUNED_MODEL_PATH
        else:
            print(
                f"Fine-tuned model not found. Loading base model {Config.MODEL_NAME}..."
            )
            model_path = Config.MODEL_NAME

        self.model = SentenceTransformer(model_path, device=self.device)
        self.heatmap_size = Config.HEATMAP_SIZE

    def _compute_heatmap(self, similarity_vector, target_size):
        """
        Resamples the similarity vector to a fixed size using linear interpolation.
        This creates a 'Global Structural Heatmap' invariant to notebook length.

        Args:
            similarity_vector (np.array): 1D array of similarity scores.
            target_size (int): Desired output size (K).

        Returns:
            np.array: Resampled vector of size K.
        """
        n = len(similarity_vector)
        if n == 0:
            return np.zeros(target_size)
        if n == 1:
            return np.full(target_size, similarity_vector[0])

        # Original coordinates
        x_old = np.linspace(0, 1, n)
        # Target coordinates
        x_new = np.linspace(0, 1, target_size)

        # Linear interpolation
        heatmap = np.interp(x_new, x_old, similarity_vector)
        return heatmap

    def _process_notebook(self, nb_id, code_cells, md_cells, cell_order=None):
        """
        Extracts features for a single notebook.

        Args:
            nb_id (str): Notebook ID.
            code_cells (list): List of code cell dicts.
            md_cells (list): List of markdown cell dicts.
            cell_order (str, optional): Ground truth cell order string.

        Returns:
            list: List of dicts, one per markdown cell containing features.
        """
        if not code_cells or not md_cells:
            return []

        n_code = len(code_cells)
        if n_code == 0:
            return []

        # 1. Encode Texts
        code_sources = [c["source"] for c in code_cells]
        md_sources = [c["source"] for c in md_cells]

        # Batch encode (returns numpy array)
        code_embs = self.model.encode(
            code_sources,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        md_embs = self.model.encode(
            md_sources,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 2. Compute Similarity Matrix (MD x Code)
        # Normalize embeddings for cosine similarity
        code_norm = np.linalg.norm(code_embs, axis=1, keepdims=True)
        md_norm = np.linalg.norm(md_embs, axis=1, keepdims=True)

        # Avoid division by zero
        code_norm[code_norm == 0] = 1e-9
        md_norm[md_norm == 0] = 1e-9

        code_embs = code_embs / code_norm
        md_embs = md_embs / md_norm

        # Cosine similarity: (N_md, D) @ (D, N_code) -> (N_md, N_code)
        sim_matrix = np.dot(md_embs, code_embs.T)

        # Pre-calculate target ranks if cell_order is provided (Train/Val)
        md_targets = {}
        if cell_order:
            # Parse cell order to find rank
            # Target = number of code cells strictly before the markdown cell
            current_code_count = 0
            for cell_id in cell_order.split():
                if any(c["id"] == cell_id for c in code_cells):
                    current_code_count += 1
                elif any(c["id"] == cell_id for c in md_cells):
                    md_targets[cell_id] = current_code_count

        features_list = []

        # 3. Extract Features per Markdown Cell
        for i, md_cell in enumerate(md_cells):
            md_id = md_cell["id"]
            sim_vec = sim_matrix[i]  # Shape (n_code,)

            # A. Smoothing (Local Spatial Anchors)
            # Apply Gaussian smoothing to reduce noise in similarity signal
            smoothed_sim = gaussian_filter1d(sim_vec, sigma=1.0)

            # B. Local Anchors
            # best_match_loc: position of max similarity (normalized 0-1)
            best_match_idx = np.argmax(smoothed_sim)
            best_match_loc = best_match_idx / max(1, n_code - 1)

            # center_of_mass: weighted average position
            # Clip negative similarities to 0 for weight calculation
            weights = np.maximum(sim_vec, 0)
            if np.sum(weights) > 0:
                com_idx = np.average(np.arange(n_code), weights=weights)
                center_of_mass = com_idx / max(1, n_code - 1)
            else:
                center_of_mass = 0.5  # Default fallback

            # C. Global Structural Heatmap
            heatmap = self._compute_heatmap(sim_vec, self.heatmap_size)

            # D. Context
            md_len = len(md_cell["source"])

            row = {
                "id": nb_id,
                "cell_id": md_id,
                "n_code": n_code,
                "md_len": md_len,
                "best_match_loc": best_match_loc,
                "center_of_mass": center_of_mass,
            }

            # Add heatmap features
            for h_idx, val in enumerate(heatmap):
                row[f"heatmap_{h_idx}"] = val

            # Add target if available
            if cell_order:
                if md_id in md_targets:
                    # Normalize target to [0, 1]
                    # If n_code is large, this approximates the position distribution
                    row["target"] = md_targets[md_id] / n_code
                else:
                    # Fallback if ID mismatch (should rarely happen)
                    continue

            features_list.append(row)

        return features_list

    def extract_features(self, metadata_path, save_path, load_cached_data=True):
        """
        Main pipeline to extract features for a dataset.

        Args:
            metadata_path (str): Path to metadata CSV.
            save_path (str): Path to save/load Parquet file.
            load_cached_data (bool): Whether to use cached data.

        Returns:
            pd.DataFrame: DataFrame containing features and targets.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(save_path):
            print(f"Loading cached features from {save_path}")
            try:
                return pd.read_parquet(save_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

        # 2. Process Data
        print(f"Extracting features from {metadata_path}...")
        set_seed(Config.SEED)

        # Load metadata to get cell_order map
        df_meta = pd.read_csv(metadata_path)

        if "cell_order" in df_meta.columns:
            order_map = dict(zip(df_meta["id"], df_meta["cell_order"]))
        else:
            order_map = {}

        iterator = get_notebook_iterator(metadata_path)

        all_features = []

        # Iterate through notebooks
        for nb_id, code_cells, md_cells in iterator:
            cell_order = order_map.get(nb_id)
            nb_feats = self._process_notebook(nb_id, code_cells, md_cells, cell_order)
            all_features.extend(nb_feats)

        df_features = pd.DataFrame(all_features)

        # 3. Save to Cache
        if not df_features.empty:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df_features.to_parquet(save_path, index=False)
            print(f"Features saved to {save_path}")
        else:
            print("Warning: No features extracted.")

        return df_features
