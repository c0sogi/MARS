import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import convolve1d
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import set_seed
from library.data_processing import load_notebook_data


class FeatureExtractor:
    """
    Handles the generation of semantic features using a Sentence Transformer backbone
    and 1D signal smoothing.
    """

    def __init__(self, model_path=None):
        """
        Initializes the feature extractor with a semantic model.
        Prioritizes the fine-tuned model path if provided or available in Config.
        """
        set_seed(Config.SEED)
        self.device = Config.DEVICE

        # Determine model path: explicit arg -> fine-tuned dir -> base config name
        if model_path:
            load_path = model_path
        elif os.path.exists(Config.BACKBONE_OUTPUT_DIR):
            load_path = Config.BACKBONE_OUTPUT_DIR
            print(f"FeatureExtractor: Loading fine-tuned model from {load_path}")
        else:
            load_path = Config.MODEL_NAME
            print(
                f"FeatureExtractor: Fine-tuned model not found. Loading base model {load_path}"
            )

        self.model = SentenceTransformer(load_path, device=self.device)
        self.smoothing_kernel = np.array(Config.SMOOTHING_KERNEL)

    def encode_texts(self, texts):
        """
        Encodes a list of texts into normalized embeddings.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def compute_similarity(self, md_embeddings, code_embeddings):
        """
        Computes pairwise cosine similarity matrix (M x N).
        Rows: Markdown cells, Cols: Code cells.
        """
        if md_embeddings.size == 0 or code_embeddings.size == 0:
            return np.array([])

        # Embeddings are already normalized, so dot product is cosine similarity
        return np.dot(md_embeddings, code_embeddings.T)

    def apply_1d_smoothing(self, sim_matrix):
        """
        Applies 1D convolution along the code axis (axis 1) to smooth the signal.
        """
        if sim_matrix.size == 0:
            return sim_matrix

        # convolve1d with mode='constant' (zero padding) to isolate dense regions
        smoothed = convolve1d(
            sim_matrix, self.smoothing_kernel, axis=1, mode="constant", cval=0.0
        )
        return smoothed

    def get_features_for_notebook(self, nb_data):
        """
        Extracts features for all markdown cells in a single notebook.

        Args:
            nb_data (dict): Dictionary containing 'code' and 'markdown' maps.

        Returns:
            list: List of dictionaries containing features for each markdown cell.
        """
        code_map = nb_data["code"]
        md_map = nb_data["markdown"]

        if not code_map or not md_map:
            return []

        # Preserve order for deterministic processing
        code_ids = list(code_map.keys())
        md_ids = list(md_map.keys())

        code_texts = [code_map[cid] for cid in code_ids]
        md_texts = [md_map[mid] for mid in md_ids]

        # 1. Encode
        code_emb = self.encode_texts(code_texts)
        md_emb = self.encode_texts(md_texts)

        # 2. Compute Similarity
        sim_matrix = self.compute_similarity(md_emb, code_emb)

        # 3. Apply Smoothing
        smoothed_matrix = self.apply_1d_smoothing(sim_matrix)

        n_code = len(code_ids)
        features_list = []

        for i, md_id in enumerate(md_ids):
            row = smoothed_matrix[i]

            # Feature: Signal Strength (Max Similarity)
            sim_max = np.max(row)

            # Feature: Smoothed Best Match Location (Normalized Index)
            # argmax returns the index of the max value
            best_match_idx = np.argmax(row)
            smoothed_best_match_loc = best_match_idx / n_code if n_code > 0 else 0.0

            # Feature: Smoothed Center of Mass
            # sum(indices * weights) / sum(weights)
            weights_sum = np.sum(row)
            if weights_sum > 1e-9:
                indices = np.arange(n_code)
                center_of_mass = np.sum(indices * row) / weights_sum
                smoothed_center_of_mass = center_of_mass / n_code
            else:
                smoothed_center_of_mass = 0.5  # Default to middle if no signal

            features_list.append(
                {
                    "cell_id": md_id,
                    "n_code": n_code,
                    "md_len": len(md_texts[i]),
                    "smoothed_best_match_loc": smoothed_best_match_loc,
                    "smoothed_center_of_mass": smoothed_center_of_mass,
                    "sim_max": sim_max,
                }
            )

        return features_list


def generate_features(metadata_path, output_path, load_cached_data=True, debug=False):
    """
    Main pipeline to generate feature datasets.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        output_path (str): Path to save/load the Parquet file.
        load_cached_data (bool): If True, try to load from output_path first.
        debug (bool): If True, process a subset of data.

    Returns:
        pd.DataFrame: The features dataframe.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        try:
            return pd.read_parquet(output_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Load Data
    print(f"Generating features from {metadata_path} (Debug={debug})...")
    notebooks = load_notebook_data(metadata_path, debug=debug)

    # 3. Initialize Extractor
    extractor = FeatureExtractor()

    all_features = []

    # 4. Process Notebooks
    print("Extracting features...")
    for nb_id, nb_data in tqdm(notebooks.items(), total=len(notebooks)):
        # Extract semantic features
        nb_feats = extractor.get_features_for_notebook(nb_data)

        # If training/val, calculate target rank
        # Target: Fraction of code cells strictly before the markdown cell
        cell_order = nb_data.get("order", [])
        calc_target = len(cell_order) > 0

        code_ids = list(nb_data["code"].keys())
        # Map code cell IDs to their integer index in the code-only sequence
        code_rank_map = {cid: i for i, cid in enumerate(code_ids)}

        # Map all cell IDs to their index in the full ground truth sequence
        full_rank_map = {cid: i for i, cid in enumerate(cell_order)}

        for feat in nb_feats:
            feat["id"] = nb_id

            if calc_target:
                md_id = feat["cell_id"]
                if md_id in full_rank_map:
                    md_rank = full_rank_map[md_id]

                    # Count code cells appearing before this markdown cell
                    # This is the target variable for the regressor
                    code_before = 0
                    for cid in code_ids:
                        if cid in full_rank_map and full_rank_map[cid] < md_rank:
                            code_before += 1

                    # Normalize target to [0, 1]
                    n_code = feat["n_code"]
                    feat["rank"] = code_before / n_code if n_code > 0 else 0.0
                else:
                    # Fallback if ID mismatch (should not happen in clean data)
                    feat["rank"] = 0.0

            all_features.append(feat)

    # 5. Save
    df_features = pd.DataFrame(all_features)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving {len(df_features)} rows to {output_path}")
    df_features.to_parquet(output_path, index=False)

    return df_features
