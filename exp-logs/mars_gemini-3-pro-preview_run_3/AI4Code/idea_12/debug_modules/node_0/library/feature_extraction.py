import os
import numpy as np
import pandas as pd
import torch
import scipy.ndimage
from scipy.interpolate import interp1d
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_utils import get_metadata, get_notebook_cells


class FeatureExtractor:
    """
    Handles the generation of features for the Bidirectional Contextualized Regressor.
    Extracts semantic similarities, structural heatmaps, and context features.
    """

    def __init__(self):
        self.device = Config.Model.DEVICE
        self.base_model_name = Config.Model.BASE_MODEL_NAME
        self.fine_tuned_path = Config.Paths.MODEL_OUTPUT_DIR
        self.max_seq_len = Config.Model.MAX_SEQ_LEN
        self.heatmap_bins = Config.Features.HEATMAP_BINS
        self.smoothing_window = Config.Features.SMOOTHING_WINDOW
        self.cache_dir = Config.Paths.CACHE_DIR

    def _load_model(self):
        """
        Loads the SentenceTransformer model.
        Prioritizes the fine-tuned model; falls back to base model if not found.
        """
        # Check if fine-tuned model exists (look for config.json or model.safetensors/pytorch_model.bin)
        # SentenceTransformers saves a config.json in the root of the save path
        if os.path.exists(os.path.join(self.fine_tuned_path, "config.json")):
            print(f"Loading fine-tuned model from {self.fine_tuned_path}...")
            model = SentenceTransformer(self.fine_tuned_path, device=self.device)
        else:
            print(
                f"Fine-tuned model not found at {self.fine_tuned_path}. "
                f"Falling back to base model: {self.base_model_name}"
            )
            model = SentenceTransformer(self.base_model_name, device=self.device)

        model.max_seq_length = self.max_seq_len
        return model

    def _compute_target(self, markdown_id, cell_order_list, code_ids_set):
        """
        Calculates the normalized rank target for a markdown cell.
        Target = (Number of code cells before this markdown) / (Total code cells)
        """
        if markdown_id not in cell_order_list:
            return 0.0

        md_idx = cell_order_list.index(markdown_id)

        # Count code cells before this markdown cell
        preceding_cells = cell_order_list[:md_idx]
        n_preceding_code = sum(1 for cid in preceding_cells if cid in code_ids_set)

        total_code = len(code_ids_set)
        if total_code == 0:
            return 0.0

        return n_preceding_code / total_code

    def _process_notebook(self, row, model):
        """
        Processes a single notebook to extract features for all its markdown cells.
        """
        notebook_id = row["id"]
        rel_path = row["file_path"]

        # Get cell content
        try:
            notebook_data = get_notebook_cells(notebook_id, rel_path)
        except Exception as e:
            # In case of read error, return empty list
            return []

        code_cells = notebook_data["code_cells"]
        markdown_cells = notebook_data["markdown_cells"]

        n_code = len(code_cells)
        n_md = len(markdown_cells)

        if n_md == 0:
            return []

        # If no code cells, we can't compute relative position meaningfully for this task formulation.
        # We return features with 0 similarity.
        if n_code == 0:
            features_list = []
            for md_cell in markdown_cells:
                feat = {
                    "notebook_id": notebook_id,
                    "markdown_id": md_cell["id"],
                    "n_code": 0,
                    "md_len": len(md_cell["source"]),
                    "best_match_loc": 0.0,
                    "center_of_mass": 0.0,
                    "sim_max": 0.0,
                }
                # Add empty heatmap
                for b in range(self.heatmap_bins):
                    feat[f"heatmap_{b}"] = 0.0

                # Add target if available
                if "cell_order" in row:
                    feat["target"] = 0.0  # Default for no code

                features_list.append(feat)
            return features_list

        # Encode cells
        # Batch encoding is faster
        code_texts = [c["source"] for c in code_cells]
        md_texts = [m["source"] for m in markdown_cells]

        # Encode
        # convert_to_tensor=True returns torch tensors on device
        code_embeddings = model.encode(
            code_texts, convert_to_tensor=True, show_progress_bar=False
        )
        md_embeddings = model.encode(
            md_texts, convert_to_tensor=True, show_progress_bar=False
        )

        # Normalize embeddings for cosine similarity
        code_embeddings = torch.nn.functional.normalize(code_embeddings, p=2, dim=1)
        md_embeddings = torch.nn.functional.normalize(md_embeddings, p=2, dim=1)

        # Compute Cosine Similarity Matrix (Markdown x Code)
        # Shape: [n_md, n_code]
        cosine_sim_matrix = (
            torch.mm(md_embeddings, code_embeddings.transpose(0, 1)).cpu().numpy()
        )

        # Prepare target info if training/val
        cell_order_list = []
        code_ids_set = set()
        if "cell_order" in row:
            cell_order_list = row["cell_order"].split()
            code_ids_set = {c["id"] for c in code_cells}

        features_list = []

        # Feature Engineering per Markdown Cell
        for i, md_cell in enumerate(markdown_cells):
            sim_vector = cosine_sim_matrix[i]  # Shape: [n_code]

            # 1. Smoothing
            # Apply 1D uniform filter
            if n_code >= self.smoothing_window:
                smoothed_sim = scipy.ndimage.uniform_filter1d(
                    sim_vector, size=self.smoothing_window, mode="nearest"
                )
            else:
                smoothed_sim = sim_vector

            # 2. Anchors
            # best_match_loc
            best_match_idx = np.argmax(smoothed_sim)
            best_match_loc = best_match_idx / n_code

            # center_of_mass
            # sum(indices * weights) / sum(weights)
            # Clip weights to be non-negative for center of mass calculation
            weights = np.maximum(smoothed_sim, 0)
            total_weight = np.sum(weights)
            if total_weight > 1e-6:
                indices = np.arange(n_code)
                center_mass_idx = np.sum(indices * weights) / total_weight
                center_of_mass = center_mass_idx / n_code
            else:
                center_of_mass = 0.5  # Default to middle if no signal

            # sim_max
            sim_max = np.max(smoothed_sim)

            # 3. Structural Heatmap (Interpolation)
            # Interpolate smoothed vector to fixed bins
            x_original = np.linspace(0, 1, n_code)
            x_target = np.linspace(0, 1, self.heatmap_bins)

            if n_code > 1:
                f_interp = interp1d(
                    x_original, smoothed_sim, kind="linear", fill_value="extrapolate"
                )
                heatmap = f_interp(x_target)
            else:
                # If only 1 code cell, heatmap is constant
                heatmap = np.full(self.heatmap_bins, smoothed_sim[0])

            # Construct Feature Dict
            feat = {
                "notebook_id": notebook_id,
                "markdown_id": md_cell["id"],
                "n_code": n_code,
                "md_len": len(md_cell["source"]),
                "best_match_loc": best_match_loc,
                "center_of_mass": center_of_mass,
                "sim_max": sim_max,
            }

            # Add heatmap bins
            for b in range(self.heatmap_bins):
                feat[f"heatmap_{b}"] = float(heatmap[b])

            # Add target if available
            if "cell_order" in row:
                target = self._compute_target(
                    md_cell["id"], cell_order_list, code_ids_set
                )
                feat["target"] = target

            features_list.append(feat)

        return features_list

    def extract_features(self, split, load_cached_data=True):
        """
        Main driver function to generate features for a dataset split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame containing features and targets (if applicable).
        """
        cache_path = os.path.join(self.cache_dir, f"features_{split}.parquet")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features for {split} from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {split} from scratch...")

        # 2. Load Metadata and Model
        df_metadata = get_metadata(split)

        # For debugging/development, one might limit the number of notebooks,
        # but the config specifies using full datasets for the regressor.
        # However, if Config.Training.NUM_NOTEBOOKS_LGBM is set and we are training,
        # we could sample. But usually feature extraction happens on full sets
        # and sampling happens at training time. We process all here.

        model = self._load_model()

        all_features = []

        # 3. Processing Loop
        # Avoid tqdm as per instructions
        total = len(df_metadata)
        print(f"Processing {total} notebooks...")

        for idx, row in df_metadata.iterrows():
            nb_features = self._process_notebook(row, model)
            all_features.extend(nb_features)

            # Simple logging every 10%
            if (idx + 1) % max(1, total // 10) == 0:
                print(f"Processed {idx + 1}/{total} notebooks.")

        # 4. Create DataFrame
        if not all_features:
            print("Warning: No features generated. Returning empty DataFrame.")
            return pd.DataFrame()

        df_features = pd.DataFrame(all_features)

        # Optimize types
        float_cols = [
            c for c in df_features.columns if df_features[c].dtype == "float64"
        ]
        df_features[float_cols] = df_features[float_cols].astype("float32")

        # 5. Save Cache
        print(f"Saving features to {cache_path}")
        df_features.to_parquet(cache_path, index=False)

        return df_features
