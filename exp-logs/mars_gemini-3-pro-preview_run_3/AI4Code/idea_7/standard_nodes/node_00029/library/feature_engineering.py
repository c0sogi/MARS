import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_utils import read_notebook


class SBERTVectorizer:
    """
    Wrapper for SentenceTransformer to handle model loading and encoding.
    """

    def __init__(self):
        # Prioritize loading the fine-tuned model
        if os.path.exists(Config.FINE_TUNED_MODEL_PATH):
            print(
                f"SBERTVectorizer: Loading fine-tuned model from {Config.FINE_TUNED_MODEL_PATH}"
            )
            model_path = Config.FINE_TUNED_MODEL_PATH
        else:
            print(f"SBERTVectorizer: Loading base model {Config.MODEL_CHECKPOINT}")
            model_path = Config.MODEL_CHECKPOINT

        self.model = SentenceTransformer(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode(self, texts):
        """
        Encodes a list of texts into embeddings.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=Config.EVAL_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


def apply_smoothing(sim_vector, kernel):
    """
    Applies 1D convolution smoothing to the similarity vector.
    """
    if len(sim_vector) < len(kernel):
        return sim_vector
    return np.convolve(sim_vector, kernel, mode="same")


def process_notebook(row, vectorizer, mode, kernel):
    """
    Extracts features for a single notebook.
    """
    nb_id = row["id"]
    data = read_notebook(row["file_path"])
    if data is None:
        return []

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    # Determine Code Cell Order
    # For Train/Val: Use ground truth order to ensure correct sequence of code cells
    # For Test: Use JSON insertion order (assuming code cells are ordered in file as per task desc)
    all_cells = list(cell_types.keys())

    if mode in ["train", "val"]:
        gt_order = row["cell_order"].split()
        code_ids = [c for c in gt_order if cell_types.get(c) == "code"]
    else:
        # In test, we assume the list of keys preserves the code order
        code_ids = [c for c in all_cells if cell_types.get(c) == "code"]

    md_ids = [c for c in all_cells if cell_types.get(c) == "markdown"]

    n_code = len(code_ids)

    # Handle edge case: No code cells
    if n_code == 0:
        results = []
        for mid in md_ids:
            feat = {
                "id": nb_id,
                "cell_id": mid,
                "best_match_loc": 0.0,
                "center_of_mass": 0.0,
                "sim_max": 0.0,
                "n_code": 0,
                "md_len": len(sources.get(mid, "")),
            }
            if mode in ["train", "val"]:
                feat["target"] = 0.0
            results.append(feat)
        return results

    # Encode
    code_texts = [str(sources.get(c, "")) for c in code_ids]
    md_texts = [str(sources.get(c, "")) for c in md_ids]

    emb_code = vectorizer.encode(code_texts)
    emb_md = vectorizer.encode(md_texts)

    if len(emb_md) == 0:
        return []

    # Compute Similarity Matrix (n_md, n_code)
    sim_matrix = np.matmul(emb_md, emb_code.T)

    # Calculate Targets (Train/Val only)
    targets = {}
    if mode in ["train", "val"]:
        gt_order = row["cell_order"].split()
        current_code_idx = 0
        for cid in gt_order:
            if cell_types.get(cid) == "code":
                current_code_idx += 1
            elif cell_types.get(cid) == "markdown":
                targets[cid] = current_code_idx / n_code

    results = []
    for i, mid in enumerate(md_ids):
        sim_vec = sim_matrix[i]

        # Apply Smoothing
        sim_vec_smooth = apply_smoothing(sim_vec, kernel)

        # Feature Extraction
        sim_max = float(np.max(sim_vec_smooth))
        argmax = np.argmax(sim_vec_smooth)
        best_match_loc = argmax / n_code

        # Center of Mass
        weights = np.maximum(sim_vec_smooth, 0)
        w_sum = np.sum(weights)
        if w_sum > 1e-9:
            indices = np.arange(n_code)
            com = np.sum(indices * weights) / w_sum
            center_of_mass = com / n_code
        else:
            center_of_mass = best_match_loc

        md_len = len(md_texts[i])

        feat = {
            "id": nb_id,
            "cell_id": mid,
            "best_match_loc": best_match_loc,
            "center_of_mass": center_of_mass,
            "sim_max": sim_max,
            "n_code": n_code,
            "md_len": md_len,
        }

        if mode in ["train", "val"]:
            feat["target"] = targets.get(mid, 0.0)

        results.append(feat)

    return results


def generate_features(df, mode, load_cached_data=True):
    """
    Main entry point for feature generation.
    Handles caching and orchestration.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing features.
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

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} features from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating.")

    # Generate
    print(f"Generating {mode} features for {len(df)} notebooks...")
    vectorizer = SBERTVectorizer()
    kernel = np.array(Config.SMOOTHING_KERNEL)

    all_features = []

    # Process notebooks
    for _, row in df.iterrows():
        nb_feats = process_notebook(row, vectorizer, mode, kernel)
        all_features.extend(nb_feats)

    features_df = pd.DataFrame(all_features)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    features_df.to_parquet(cache_path, index=False)
    print(f"Saved {mode} features to {cache_path}")

    return features_df
