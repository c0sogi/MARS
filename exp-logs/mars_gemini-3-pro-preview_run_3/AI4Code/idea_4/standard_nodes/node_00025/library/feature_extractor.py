import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

from library.config import Config
from library.data_loader import get_notebook_cells


class FeatureEngineer:
    """
    Extracts dense features from notebooks using a SemanticModel backbone.
    Generates tabular data for the regressor (LightGBM).
    """

    def __init__(self, semantic_model):
        """
        Args:
            semantic_model: Instance of library.backbone.SemanticModel
        """
        self.semantic_model = semantic_model

    def extract_features(
        self,
        metadata_path,
        mode="train",
        cache_name="features",
        load_cached_data=True,
        debug=False,
        batch_size=200,
    ):
        """
        Main pipeline to process notebooks and generate a feature DataFrame.

        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train' (calculates targets) or 'test' (inference only).
            cache_name (str): Filename for the cache (without extension).
            load_cached_data (bool): Whether to load from cache if available.
            debug (bool): If True, processes a small subset of data.
            batch_size (int): Number of notebooks to process in one embedding batch.

        Returns:
            pd.DataFrame: DataFrame containing features and targets (if train mode).
        """
        # 1. Cache Handling
        cache_file = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}...")
            return pd.read_parquet(cache_file)

        # 2. Load Metadata
        print(f"Generating features from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        if debug:
            print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks.")
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        # 3. Processing Loop
        features_list = []

        # Split dataframe into chunks to manage memory and batching
        chunks = [
            df_meta[i : i + batch_size] for i in range(0, df_meta.shape[0], batch_size)
        ]

        print(f"Processing {len(df_meta)} notebooks in {len(chunks)} batches...")

        for i, chunk in enumerate(chunks):
            chunk_md_texts = []
            chunk_code_texts = []
            chunk_structs = []

            # --- Step A: Parse Notebooks in Chunk ---
            for _, row in chunk.iterrows():
                nb_id = row["id"]
                file_path = row["file_path"]
                # For train/val, passing cell_order allows get_notebook_cells to assign ground truth ranks
                cell_order = row["cell_order"] if mode != "test" else None

                nb_data = get_notebook_cells(nb_id, file_path, cell_order)
                md_cells = nb_data["markdown_cells"]
                code_cells = nb_data["code_cells"]

                # Skip notebooks that are missing code or markdown cells (cannot align)
                if not code_cells or not md_cells:
                    chunk_structs.append(None)
                    continue

                # Structure to hold metadata for reconstruction after bulk encoding
                struct = {
                    "id": nb_id,
                    "md_cells": md_cells,
                    "code_cells": code_cells,
                    "n_md": len(md_cells),
                    "n_code": len(code_cells),
                    "md_offset": len(chunk_md_texts),
                    "code_offset": len(chunk_code_texts),
                }

                # Calculate Targets (Training Mode Only)
                if mode != "test":
                    # Determine the number of code cells strictly preceding each markdown cell
                    targets = {}
                    code_counter = 0
                    # nb_data["all_cells_ordered"] is guaranteed to be in ground truth order
                    for cell in nb_data["all_cells_ordered"]:
                        if cell["type"] == "code":
                            code_counter += 1
                        elif cell["type"] == "markdown":
                            targets[cell["id"]] = code_counter
                    struct["targets"] = targets

                chunk_structs.append(struct)

                # Collect texts for bulk encoding
                chunk_md_texts.extend([c["text"] for c in md_cells])
                chunk_code_texts.extend([c["text"] for c in code_cells])

            # If chunk yielded no valid data, continue
            if not chunk_md_texts:
                continue

            # --- Step B: Bulk Encoding ---
            # SemanticModel handles GPU batching internally
            md_embeddings = self.semantic_model.encode(
                chunk_md_texts, show_progress_bar=False
            )
            code_embeddings = self.semantic_model.encode(
                chunk_code_texts, show_progress_bar=False
            )

            # --- Step C: Feature Extraction per Notebook ---
            for struct in chunk_structs:
                if struct is None:
                    continue

                nb_id = struct["id"]
                n_code = struct["n_code"]

                # Slice embeddings for this notebook
                md_start = struct["md_offset"]
                md_end = md_start + struct["n_md"]
                code_start = struct["code_offset"]
                code_end = code_start + n_code

                curr_md_embs = md_embeddings[md_start:md_end]
                curr_code_embs = code_embeddings[code_start:code_end]

                # Compute Cosine Similarity Matrix (n_md, n_code)
                sim_matrix = cosine_similarity(curr_md_embs, curr_code_embs)

                # Generate feature row for each markdown cell
                for idx, md_cell in enumerate(struct["md_cells"]):
                    cell_id = md_cell["id"]
                    sim_scores = sim_matrix[idx]  # Shape: (n_code,)

                    # 1. Signal Strength Features
                    sim_max = np.max(sim_scores)
                    sim_mean = np.mean(sim_scores)

                    # 2. Spatial Alignment Features
                    # Index of the code cell with highest similarity
                    best_match_loc = np.argmax(sim_scores)

                    # Center of Mass: Softmax-weighted average index
                    # Softmax sharpens the distribution to focus on relevant code regions
                    weights = softmax(sim_scores)
                    center_of_mass = np.sum(np.arange(n_code) * weights)

                    # 3. Global Context Features
                    md_len = len(md_cell["text"])

                    feat_row = {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "n_code": n_code,
                        "md_len": md_len,
                        "best_match_loc": best_match_loc,
                        "center_of_mass": center_of_mass,
                        "sim_max": sim_max,
                        "sim_mean": sim_mean,
                    }

                    # Add Target for Training
                    if mode != "test":
                        # Target is the normalized rank: (num_code_before / total_code)
                        # Range [0.0, 1.0]
                        raw_target = struct["targets"].get(cell_id, 0)
                        feat_row["target"] = raw_target / n_code if n_code > 0 else 0.0

                    features_list.append(feat_row)

            # Log progress periodically
            if (i + 1) % 10 == 0:
                print(f"Processed batch {i + 1}/{len(chunks)}")

        # 4. Save to Cache
        df_features = pd.DataFrame(features_list)
        print(f"Saving {len(df_features)} feature rows to {cache_file}...")
        df_features.to_parquet(cache_file, index=False)

        return df_features
