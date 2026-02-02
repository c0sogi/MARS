import os
import hashlib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from library.config import Config
from library.utils import preprocess_text
from library.data_loader import load_notebook, get_ordered_cells


class SemanticFeatureExtractor:
    """
    Handles the extraction of semantic features from notebooks using Sentence Transformers and
    Cosine Similarity statistics to determine the relative position of markdown cells.
    """

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit_vectorizer(self, df_metadata, sample_size=5000, load_cached=True):
        """
        Loads the Sentence Transformer model.
        Maintains API compatibility with the previous TF-IDF implementation.
        """
        if self.model is None:
            print(
                f"Loading Sentence Transformer: {Config.BERT_MODEL_NAME} on {self.device}..."
            )
            # Cite solution_lesson_node_00006: Semantic Embeddings Outperform Lexical Matching
            self.model = SentenceTransformer(Config.BERT_MODEL_NAME, device=self.device)

    def generate_dataset(self, df_metadata, mode="train", load_cached_data=True):
        """
        Generates a feature dataset for the regressor.

        Args:
            df_metadata (pd.DataFrame): Metadata containing notebook IDs and paths.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from parquet cache if available.

        Returns:
            pd.DataFrame: DataFrame containing features and targets (if train/val).
        """
        # Create a deterministic hash of the input IDs to ensure cache validity
        # Cite debug_lesson_1: Parameterize Context-Dependent Cache Keys
        ids_hash = hashlib.md5(
            pd.util.hash_pandas_object(df_metadata["id"], index=False).values.tobytes()
        ).hexdigest()
        cache_path = os.path.join(
            Config.CACHE_DIR, f"features_{mode}_{ids_hash}.parquet"
        )

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            return pd.read_parquet(cache_path)

        if self.model is None:
            self.fit_vectorizer(None)

        print(f"Generating features for {mode} set ({len(df_metadata)} notebooks)...")

        features_list = []

        for _, row in df_metadata.iterrows():
            nb_id = row["id"]
            file_path = row["file_path"]

            try:
                nb_data = load_notebook(file_path)
            except Exception:
                continue

            code_cells = nb_data["code_cells"]
            markdown_cells = nb_data["markdown_cells"]

            # Skip if no markdown cells (nothing to predict)
            if not markdown_cells:
                continue

            # Determine the ordered list of code cell IDs
            ordered_code_ids = []
            if mode in ["train", "val"]:
                # For training, we use the ground truth to identify code cells and their order
                full_order = get_ordered_cells(row["cell_order"])
                ordered_code_ids = [cid for cid in full_order if cid in code_cells]
            else:
                # For test, we assume the code cells in the JSON are in the correct relative order
                ordered_code_ids = list(code_cells.keys())

            n_code = len(ordered_code_ids)

            # Preprocess texts - Use stem=False for Transformers
            # Cite solution_lesson_node_00006: Semantic Embeddings
            code_texts = [
                preprocess_text(code_cells[cid], stem=False) for cid in ordered_code_ids
            ]
            md_ids = list(markdown_cells.keys())
            md_texts = [
                preprocess_text(markdown_cells[mid], stem=False) for mid in md_ids
            ]

            # Handle case with no code cells
            if n_code == 0:
                for i, mid in enumerate(md_ids):
                    feat = {
                        "id": nb_id,
                        "cell_id": mid,
                        "n_code": 0,
                        "md_len": len(markdown_cells[mid]),
                        "sim_mean": 0.0,
                        "sim_max": 0.0,
                        "sim_min": 0.0,
                        "sim_std": 0.0,
                        "best_match_loc": 0.5,
                        "center_of_mass": 0.5,
                    }
                    if mode in ["train", "val"]:
                        feat["target"] = 0.5
                    features_list.append(feat)
                continue

            # Encode and Compute Similarity
            # We encode in two batches: all code cells and all markdown cells for this notebook
            embeddings_code = self.model.encode(
                code_texts, convert_to_tensor=True, show_progress_bar=False
            )
            embeddings_md = self.model.encode(
                md_texts, convert_to_tensor=True, show_progress_bar=False
            )

            # Compute Cosine Similarity Matrix (Rows: MD, Cols: Code)
            # util.cos_sim returns a tensor on the same device
            sim_matrix = util.cos_sim(embeddings_md, embeddings_code).cpu().numpy()

            # Calculate Target Ranks (only for train/val)
            targets = {}
            if mode in ["train", "val"]:
                full_order = get_ordered_cells(row["cell_order"])
                rank_map = {cid: i for i, cid in enumerate(full_order)}
                code_ranks = [rank_map[cid] for cid in ordered_code_ids]

                for mid in md_ids:
                    if mid in rank_map:
                        my_rank = rank_map[mid]
                        pos = sum(1 for r in code_ranks if r < my_rank)
                        targets[mid] = pos / n_code
                    else:
                        targets[mid] = 0.5

            # Extract features for each markdown cell
            # Cite solution_lesson_node_00007: Prioritize Spatial Location Over Distributional Confidence
            for i, mid in enumerate(md_ids):
                sim_row = sim_matrix[i]

                # Statistics
                s_mean = np.mean(sim_row)
                s_max = np.max(sim_row)
                s_min = np.min(sim_row)
                s_std = np.std(sim_row)

                # Best match location (normalized index)
                best_idx = np.argmax(sim_row)
                best_match_loc = best_idx / n_code

                # Center of Mass
                total_sim = np.sum(sim_row)
                if total_sim > 1e-6:
                    indices = np.arange(n_code)
                    center_idx = np.sum(sim_row * indices) / total_sim
                    center_of_mass = center_idx / n_code
                else:
                    center_of_mass = 0.5

                feat = {
                    "id": nb_id,
                    "cell_id": mid,
                    "n_code": n_code,
                    "md_len": len(markdown_cells[mid]),
                    "sim_mean": s_mean,
                    "sim_max": s_max,
                    "sim_min": s_min,
                    "sim_std": s_std,
                    "best_match_loc": best_match_loc,
                    "center_of_mass": center_of_mass,
                }

                if mode in ["train", "val"]:
                    feat["target"] = targets.get(mid, 0.5)

                features_list.append(feat)

        # Create DataFrame
        df_features = pd.DataFrame(features_list)

        # Save to cache
        print(f"Saving {len(df_features)} rows to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        return df_features
