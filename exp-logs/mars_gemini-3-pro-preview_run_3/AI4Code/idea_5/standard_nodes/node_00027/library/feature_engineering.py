import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import convolve1d
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed
from library.dataset import get_notebook_data


class ModelEncoder:
    """
    Wrapper to handle multiple SentenceTransformer models for multi-view encoding.
    Manages loading of fine-tuned weights and batch encoding.
    """

    def __init__(self):
        self.models = []
        self.device = Config.DEVICE

        for name in Config.MODEL_NAMES:
            # Determine path: check for fine-tuned version first
            save_path = Config.MODEL_SAVE_PATHS[name]
            if os.path.exists(save_path):
                print(f"Loading fine-tuned model from {save_path}")
                model_path = save_path
            else:
                print(
                    f"Fine-tuned model not found at {save_path}. Loading base model {name}"
                )
                model_path = name

            model = SentenceTransformer(model_path, device=str(self.device))
            self.models.append(model)

    def encode(self, text_list, model_idx):
        """
        Encodes a list of texts using the specified model index.
        """
        if not text_list:
            return np.array([])

        # Use batch_size from config
        return self.models[model_idx].encode(
            text_list,
            batch_size=Config.VALID_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def apply_smoothing(sim_matrix, window_size=Config.SMOOTHING_WINDOW):
    """
    Applies 1D convolution smoothing along the code axis (axis 1).
    This helps capture relationships where a markdown cell describes a block of code.
    """
    if sim_matrix.size == 0:
        return sim_matrix

    # Create uniform kernel
    kernel = np.full(window_size, 1.0 / window_size)
    # Convolve along the code cells axis. 'nearest' mode handles boundaries gracefully.
    return convolve1d(sim_matrix, kernel, axis=1, mode="nearest")


def extract_matrix_features(sim_matrix, prefix, n_code):
    """
    Extracts statistical and spatial features from a similarity matrix.

    Args:
        sim_matrix: (n_md, n_code) similarity matrix
        prefix: string prefix for feature names
        n_code: number of code cells (for normalization)
    """
    features = []
    n_md = sim_matrix.shape[0]

    if n_code == 0:
        # Fallback for notebooks with no code cells
        for _ in range(n_md):
            features.append(
                {
                    f"{prefix}_best_match_loc": 0.5,
                    f"{prefix}_center_of_mass": 0.5,
                    f"{prefix}_sim_max": 0.0,
                    f"{prefix}_sim_mean": 0.0,
                }
            )
        return features

    # 1. Best Match Location (Argmax)
    best_match_indices = np.argmax(sim_matrix, axis=1)

    # 2. Signal Strength
    sim_max = np.max(sim_matrix, axis=1)
    sim_mean = np.mean(sim_matrix, axis=1)

    # 3. Center of Mass (Weighted Average)
    # We use max(0, sim) as weights to ignore negative correlations
    x_indices = np.arange(n_code)

    for i in range(n_md):
        row = sim_matrix[i]
        weights = np.maximum(0, row)
        sum_w = np.sum(weights)

        if sum_w > 1e-9:
            com = np.sum(weights * x_indices) / sum_w
        else:
            com = n_code / 2.0  # Default to middle if no signal

        features.append(
            {
                f"{prefix}_best_match_loc": best_match_indices[i] / max(1, n_code),
                f"{prefix}_center_of_mass": com / max(1, n_code),
                f"{prefix}_sim_max": sim_max[i],
                f"{prefix}_sim_mean": sim_mean[i],
            }
        )

    return features


def calculate_ranks(cell_order, cell_types):
    """
    Calculates the rank for each markdown cell based on the ground truth cell order.
    The rank is defined as the number of code cells that precede the markdown cell.
    """
    order_list = cell_order.split()
    ranks = {}

    # Assign ranks
    current_code_pos = 0
    for cell_id in order_list:
        ctype = cell_types.get(cell_id)
        if ctype == "code":
            current_code_pos += 1
        elif ctype == "markdown":
            ranks[cell_id] = current_code_pos

    return ranks


def process_notebook(row, encoder):
    """
    Processes a single notebook to generate features for all its markdown cells.
    """
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
    code_cells, md_cells = get_notebook_data(file_path)

    if not md_cells:
        return []

    # Prepare text lists
    code_sources = [c["source"] for c in code_cells]
    md_ids = [m["id"] for m in md_cells]
    md_sources = [m["source"] for m in md_cells]

    n_code = len(code_cells)
    n_md = len(md_cells)

    # Global context features
    global_features = {
        "n_code": n_code,
        "n_md": n_md,
        "md_mean_len": np.mean([len(s) for s in md_sources]) if md_sources else 0,
    }

    # Initialize feature dictionaries for each markdown cell
    cell_features_list = [
        {"id": mid, "notebook_id": row["id"], **global_features} for mid in md_ids
    ]

    # If training/val, calculate targets
    if "cell_order" in row and pd.notna(row["cell_order"]):
        # We need a map of cell types for rank calculation
        cell_types = {c["id"]: "code" for c in code_cells}
        cell_types.update({m["id"]: "markdown" for m in md_cells})

        ranks = calculate_ranks(row["cell_order"], cell_types)

        for i, mid in enumerate(md_ids):
            if mid in ranks:
                # Target is normalized rank
                # We normalize by n_code. If n_code is 0, target is 0.5.
                target = ranks[mid] / n_code if n_code > 0 else 0.5
                cell_features_list[i]["target"] = target
            else:
                # Should not happen if data is consistent
                cell_features_list[i]["target"] = 0.5

    # Multi-View Feature Extraction
    # Iterate over our two backbones
    for model_idx, model_name in enumerate(Config.MODEL_NAMES):
        # Short name for feature prefix (e.g., 'codebert', 'mpnet')
        short_name = model_name.split("/")[-1].replace("-", "_")

        # Encode
        code_emb = encoder.encode(code_sources, model_idx)
        md_emb = encoder.encode(md_sources, model_idx)

        if n_code > 0 and n_md > 0:
            # Compute Similarity Matrix (MD x Code)
            sim_matrix = cosine_similarity(md_emb, code_emb)

            # 1. Raw Features
            raw_feats = extract_matrix_features(sim_matrix, f"{short_name}_raw", n_code)

            # 2. Smoothed Features
            smooth_matrix = apply_smoothing(sim_matrix)
            smooth_feats = extract_matrix_features(
                smooth_matrix, f"{short_name}_smooth", n_code
            )

            # Merge into cell_features_list
            for i in range(n_md):
                cell_features_list[i].update(raw_feats[i])
                cell_features_list[i].update(smooth_feats[i])
        else:
            # Handle empty code case with default values
            defaults = {}
            for t in ["raw", "smooth"]:
                for m in ["best_match_loc", "center_of_mass"]:
                    defaults[f"{short_name}_{t}_{m}"] = 0.5
                for m in ["sim_max", "sim_mean"]:
                    defaults[f"{short_name}_{t}_{m}"] = 0.0

            for i in range(n_md):
                cell_features_list[i].update(defaults)

    return cell_features_list


def generate_features(metadata_path, output_path, load_cached_data=True, debug=False):
    """
    Main function to generate features for a dataset split.
    Handles caching and debug sampling.
    """
    set_seed(Config.SEED)

    # Adjust output path for debug mode to avoid overwriting real features
    if debug:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_debug{ext}"

    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        return pd.read_parquet(output_path)

    print(f"Generating features from {metadata_path} (Debug={debug})...")

    # 2. Load Metadata
    df = pd.read_csv(metadata_path)
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: sampled {len(df)} notebooks.")

    # 3. Initialize Encoder (Loads models once)
    encoder = ModelEncoder()

    # 4. Process Notebooks
    all_features = []

    # Iterate through notebooks
    for _, row in df.iterrows():
        nb_feats = process_notebook(row, encoder)
        all_features.extend(nb_feats)

    # 5. Create DataFrame
    features_df = pd.DataFrame(all_features)

    # 6. Save to Cache
    print(f"Saving {len(features_df)} feature rows to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    return features_df
