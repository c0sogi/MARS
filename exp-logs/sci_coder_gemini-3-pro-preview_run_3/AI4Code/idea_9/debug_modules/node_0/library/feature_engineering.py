import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_loader import read_notebook


def compute_multiscale_features(
    md_embeddings: np.ndarray, code_embeddings: np.ndarray, md_lens: list, kernels: list
) -> np.ndarray:
    """
    Computes multi-scale structural alignment features for markdown cells against code cells.

    Args:
        md_embeddings: (M, D) array of markdown cell embeddings.
        code_embeddings: (C, D) array of code cell embeddings.
        md_lens: List of length M containing character lengths of markdown cells.
        kernels: List of integer kernel sizes for smoothing (e.g., [1, 3, 5]).

    Returns:
        np.ndarray: Feature matrix of shape (M, n_features).
    """
    n_md, dim = md_embeddings.shape
    n_code, _ = code_embeddings.shape

    # Handle edge case with no code cells
    if n_code == 0:
        # Return a matrix of zeros with appropriate width
        # Features per kernel: best_match_loc, center_of_mass, sim_max (3)
        # Global features: n_code, md_len (2)
        n_feats = len(kernels) * 3 + 2
        return np.zeros((n_md, n_feats))

    # 1. Compute Pairwise Cosine Similarity Matrix (M x C)
    # Normalize embeddings to unit length for cosine similarity
    md_norm = md_embeddings / (
        np.linalg.norm(md_embeddings, axis=1, keepdims=True) + 1e-9
    )
    code_norm = code_embeddings / (
        np.linalg.norm(code_embeddings, axis=1, keepdims=True) + 1e-9
    )

    # Similarity matrix: rows=markdown, cols=code
    sim_matrix = np.dot(md_norm, code_norm.T)

    features_list = []

    # Indices for center of mass calculation (0 to n_code-1)
    indices = np.arange(n_code)

    for i in range(n_md):
        row_feats = []
        raw_sims = sim_matrix[i]

        # 2. Multi-Scale Smoothing and Feature Extraction
        for k in kernels:
            if k == 1:
                smoothed = raw_sims
            else:
                # Apply uniform filter (moving average)
                # mode='constant', cval=0.0 assumes zero similarity outside boundaries
                smoothed = uniform_filter1d(raw_sims, size=k, mode="constant", cval=0.0)

            # Feature: Maximum Similarity
            sim_max = np.max(smoothed)

            # Feature: Best Match Location (Normalized)
            # argmax returns the first occurrence of max
            best_loc_idx = np.argmax(smoothed)
            best_match_loc = best_loc_idx / n_code

            # Feature: Center of Mass (Normalized)
            # We clip negative similarities to 0 for CoM calculation to focus on alignment
            pos_sims = np.maximum(smoothed, 0)
            sum_sims = np.sum(pos_sims)

            if sum_sims > 1e-9:
                com_idx = np.sum(indices * pos_sims) / sum_sims
                center_of_mass = com_idx / n_code
            else:
                center_of_mass = 0.5  # Default to middle if no signal

            row_feats.extend([best_match_loc, center_of_mass, sim_max])

        # 3. Global Context Features
        row_feats.append(n_code)
        row_feats.append(md_lens[i])

        features_list.append(row_feats)

    return np.array(features_list, dtype=np.float32)


def process_notebook(
    row: pd.Series, model: SentenceTransformer, kernels: list, mode: str
) -> pd.DataFrame:
    """
    Process a single notebook to extract features for all its markdown cells.
    """
    notebook_id = row["id"]
    file_path = row["file_path"]

    # Load notebook content
    data = read_notebook(file_path)
    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    if not cell_types:
        return pd.DataFrame()

    # Determine cell order
    # For train/val, we have ground truth 'cell_order' in metadata
    # For test, we use the order in the JSON (which is shuffled for md, sorted for code)
    # However, we need to separate MD and Code regardless.

    # Get all cell IDs present in the file
    all_cell_ids = list(cell_types.keys())

    code_ids = [cid for cid in all_cell_ids if cell_types[cid] == "code"]
    md_ids = [cid for cid in all_cell_ids if cell_types[cid] == "markdown"]

    # If using ground truth for training targets
    ground_truth_ranks = {}
    if mode in ["train", "val"]:
        if isinstance(row["cell_order"], str):
            gt_order = row["cell_order"].split()
            # Filter GT order to only include cells actually present in the file
            gt_order = [cid for cid in gt_order if cid in cell_types]

            # Identify code cells in GT order to establish the "skeleton"
            gt_code_order = [cid for cid in gt_order if cell_types[cid] == "code"]

            # Map each code cell to its rank (0, 1, 2...)
            # This is critical: The code cells are the anchors.
            # Actually, the problem is predicting where MD fits relative to code.
            # We need the rank of MD cells *relative* to the sequence of code cells.

            # Calculate target for each markdown cell
            # Target = (Number of code cells appearing before this markdown cell) / Total Code Cells

            current_code_count = 0
            for cid in gt_order:
                if cell_types[cid] == "code":
                    current_code_count += 1
                elif cell_types[cid] == "markdown":
                    ground_truth_ranks[cid] = current_code_count

        # Ensure code_ids are sorted according to ground truth for consistency
        # (Though usually code cells are already sorted in the input JSON for train set)
        # We rely on the order provided in the JSON for 'source' extraction,
        # but for embedding comparison, the order of code_ids matters.
        # In the competition, code cells are always in correct order in the JSON.
        # So we can trust the list comprehension order if we iterate over JSON keys?
        # No, dict keys are insertion ordered in Py3.7+, but safer to sort if we had rank.
        # For train set, code is sorted. For test set, code is sorted.
        pass

    if not code_ids or not md_ids:
        return pd.DataFrame()

    # Extract text
    # Limit text length for efficiency (Config.MAX_LENGTH handled by tokenizer,
    # but good to truncate source string too)
    md_texts = [sources.get(cid, "")[:1000] for cid in md_ids]
    code_texts = [sources.get(cid, "")[:1000] for cid in code_ids]
    md_lens = [len(t) for t in md_texts]

    # Encode
    # Batch encoding within the notebook
    embeddings = model.encode(
        md_texts + code_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    md_emb = embeddings[: len(md_texts)]
    code_emb = embeddings[len(md_texts) :]

    # Compute Features
    feats = compute_multiscale_features(md_emb, code_emb, md_lens, kernels)

    # Construct DataFrame
    df_feats = pd.DataFrame(
        feats,
        columns=[f"k{k}_{metric}" for k in kernels for metric in ["loc", "com", "max"]]
        + ["n_code", "md_len"],
    )

    df_feats["id"] = notebook_id
    df_feats["cell_id"] = md_ids

    # Add Target if available
    if mode in ["train", "val"]:
        n_code = len(code_ids)
        targets = []
        for cid in md_ids:
            # Rank is number of code cells before.
            # If cid not in GT (rare error case), default to end
            rank = ground_truth_ranks.get(cid, n_code)
            # Normalize
            targets.append(rank / n_code if n_code > 0 else 0.0)
        df_feats["target"] = targets

    return df_feats


def generate_features_pipeline(
    df_metadata: pd.DataFrame,
    mode: str,
    model: SentenceTransformer = None,
    load_cached_data: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Main pipeline to generate features for a dataset split.

    Args:
        df_metadata: DataFrame containing notebook metadata.
        mode: 'train', 'val', or 'test'.
        model: Pre-loaded SentenceTransformer model. If None, loads from Config.
        load_cached_data: Whether to load from parquet cache.
        debug: If True, process only a small subset.

    Returns:
        pd.DataFrame: DataFrame containing features and targets (if train/val).
    """
    # 1. Setup Paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if mode == "train":
        cache_path = Config.TRAIN_FEATURES_PATH
    elif mode == "val":
        cache_path = Config.VAL_FEATURES_PATH
    else:
        cache_path = Config.TEST_FEATURES_PATH

    if debug:
        cache_path = cache_path.replace(".parquet", "_debug.parquet")

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{mode.upper()}] Loading features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    # 3. Load Model if not provided
    if model is None:
        model_path = Config.MODEL_OUTPUT_PATH
        if not os.path.exists(model_path):
            print(
                f"Fine-tuned model not found at {model_path}. Loading base model: {Config.MODEL_NAME}"
            )
            model_path = Config.MODEL_NAME
        else:
            print(f"Loading fine-tuned model from: {model_path}")

        model = SentenceTransformer(model_path)
        model.to(Config.DEVICE)
        model.max_seq_length = Config.MAX_LENGTH

    # 4. Process Notebooks
    if debug:
        print(
            f"[{mode.upper()}] Debug mode: Processing first {Config.DEBUG_SAMPLE_SIZE} notebooks."
        )
        df_metadata = df_metadata.head(Config.DEBUG_SAMPLE_SIZE)
    else:
        print(f"[{mode.upper()}] Processing {len(df_metadata)} notebooks...")

    all_features = []

    # Iterate through notebooks
    # Note: We process sequentially. For massive scale, parallelization via multiprocessing
    # (with model on CPU or multiple GPUs) would be better, but sequential is safer for
    # single GPU memory management within this environment.

    for _, row in df_metadata.iterrows():
        try:
            df_nb = process_notebook(row, model, Config.SMOOTHING_KERNELS, mode)
            if not df_nb.empty:
                all_features.append(df_nb)
        except Exception as e:
            # Skip problematic notebooks to avoid crashing the whole pipeline
            continue

    if not all_features:
        print(f"[{mode.upper()}] Warning: No features extracted.")
        return pd.DataFrame()

    # 5. Aggregate and Save
    final_df = pd.concat(all_features, ignore_index=True)

    # Optimize types
    float_cols = final_df.select_dtypes(include=["float64"]).columns
    final_df[float_cols] = final_df[float_cols].astype("float32")

    print(f"[{mode.upper()}] Saving {len(final_df)} rows to {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df
