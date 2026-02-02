import os
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class EmbeddingGenerator:
    """
    Handles the conversion of raw notebook text into semantic embeddings using
    a pre-trained Sentence Transformer model. Caches results to Parquet files.
    """

    def __init__(self):
        """
        Initialize the EmbeddingGenerator.
        Loads the Sentence Transformer model onto the configured device.
        """
        Config.set_seed(Config.SEED)
        self.device = Config.DEVICE
        print(
            f"Loading Sentence Transformer model: {Config.MODEL_NAME} on {self.device}..."
        )
        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)
        self.model.eval()

    def process_split(
        self, split_name: str, load_cached_data: bool = True, debug_limit: int = None
    ):
        """
        Process a specific data split (train, val, test).

        Args:
            split_name (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load from disk cache first.
            debug_limit (int, optional): Limit the number of notebooks to process for debugging.

        Returns:
            pd.DataFrame: DataFrame containing embeddings and metadata.
        """
        # Determine paths based on split
        if split_name == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
            cache_path = Config.TRAIN_CACHE_PATH
        elif split_name == "val":
            metadata_path = Config.VAL_METADATA_PATH
            cache_path = Config.VAL_CACHE_PATH
        elif split_name == "test":
            metadata_path = Config.TEST_METADATA_PATH
            cache_path = Config.TEST_CACHE_PATH
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        # Try loading cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features for {split_name} from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                print(f"Successfully loaded {len(df)} cell records.")
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Generating features for {split_name}...")
        df_metadata = pd.read_csv(metadata_path)

        if debug_limit:
            df_metadata = df_metadata.head(debug_limit)
            print(f"Debug mode: Limiting to {debug_limit} notebooks.")

        # Prepare buffers for batch processing
        records = []
        text_buffer = []
        text_meta_buffer = []  # Stores (notebook_id, cell_id, cell_type, rank)

        # Batch size for model inference (sentences)
        # Note: Config.BATCH_SIZE is typically for training the downstream model.
        # We can use a larger batch size for inference here.
        INFERENCE_BATCH_SIZE = 256

        total_notebooks = len(df_metadata)

        for idx, row in df_metadata.iterrows():
            notebook_id = row["id"]
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Determine cell ranks if available
            cell_ranks = {}
            if "cell_order" in row and pd.notna(row["cell_order"]):
                order_list = row["cell_order"].split()
                for rank, cell_id in enumerate(order_list):
                    cell_ranks[cell_id] = rank

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            cell_types = data.get("cell_type", {})
            sources = data.get("source", {})

            for cell_id, c_type in cell_types.items():
                source_text = sources.get(cell_id, "")

                # Pre-processing: Truncate very long text to avoid excessive tokenization overhead
                # The model (all-MiniLM-L6-v2) has a limit of usually 256 or 512 tokens.
                # 1000 chars is a safe heuristic to capture the start.
                processed_text = source_text[:1000] if source_text else ""

                # Determine rank
                # For test set, rank is -1. For train/val, if cell not in order list (rare), -1.
                rank = cell_ranks.get(cell_id, -1)

                text_buffer.append(processed_text)
                text_meta_buffer.append(
                    {
                        "notebook_id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "rank": rank,
                    }
                )

                # If buffer is full, encode and flush
                if len(text_buffer) >= INFERENCE_BATCH_SIZE:
                    self._flush_buffer(text_buffer, text_meta_buffer, records)
                    text_buffer = []
                    text_meta_buffer = []

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{total_notebooks} notebooks...")

        # Process remaining items in buffer
        if text_buffer:
            self._flush_buffer(text_buffer, text_meta_buffer, records)

        # Create DataFrame
        df_features = pd.DataFrame(records)

        # Ensure embedding column is recognized correctly (though parquet handles lists well)
        # We don't need to do anything special for pyarrow/parquet with lists usually.

        # Save to cache
        print(f"Saving {len(df_features)} features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        return df_features

    def _flush_buffer(self, text_buffer, meta_buffer, records_list):
        """
        Encodes texts in the buffer and appends combined metadata and embeddings to records_list.
        """
        # Encode batch
        # show_progress_bar=False to reduce clutter
        embeddings = self.model.encode(
            text_buffer,
            batch_size=len(text_buffer),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Combine metadata with embeddings
        for meta, emb in zip(meta_buffer, embeddings):
            meta["embedding"] = (
                emb.tolist()
            )  # Convert numpy array to list for Parquet compatibility
            records_list.append(meta)


class RegressionFeatureGenerator:
    """
    Converts semantic embeddings into pairwise similarity features for LightGBM.
    Cite solution_lesson_node_00004: Uses similarity-weighted positional statistics.
    Cite solution_lesson_node_00012: Uses explicit alignment features (best_match_loc).
    Cite solution_lesson_node_00015: Uses global context (n_code).
    """

    def process_split(self, split: str):
        if split == "train":
            input_path = Config.TRAIN_CACHE_PATH
            output_path = Config.TRAIN_TABULAR_PATH
        elif split == "val":
            input_path = Config.VAL_CACHE_PATH
            output_path = Config.VAL_TABULAR_PATH
        elif split == "test":
            input_path = Config.TEST_CACHE_PATH
            output_path = Config.TEST_TABULAR_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if os.path.exists(output_path):
            print(f"Loading cached tabular features from {output_path}...")
            return pd.read_parquet(output_path)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input features not found at {input_path}")

        print(f"Generating regression features for {split}...")
        df = pd.read_parquet(input_path)

        # Helper to maintain order for test set
        df["orig_idx"] = np.arange(len(df))

        features_list = []

        # Group by notebook
        grouped = df.groupby("notebook_id")

        for nb_id, group in grouped:
            # Separate Code and Markdown
            # For train/val, rank is ground truth. For test, use orig_idx as proxy for code order
            if split in ["train", "val"]:
                code_df = group[group["cell_type"] == "code"].sort_values("rank")
            else:
                code_df = group[group["cell_type"] == "code"].sort_values("orig_idx")

            md_df = group[group["cell_type"] == "markdown"]

            n_code = len(code_df)

            # If no code cells, we can't align. Return defaults.
            if n_code == 0:
                for _, row in md_df.iterrows():
                    features_list.append(
                        {
                            "notebook_id": nb_id,
                            "cell_id": row["cell_id"],
                            "n_code": 0,
                            "md_len": len(row["embedding"]),
                            "sim_max": 0.0,
                            "sim_mean": 0.0,
                            "sim_std": 0.0,
                            "best_match_loc": 0.5,
                            "center_of_mass": 0.5,
                            "target": row["rank"] if split in ["train", "val"] else -1,
                        }
                    )
                continue

            # Extract embeddings
            code_emb = np.stack(code_df["embedding"].values)  # (n_code, dim)
            md_emb = (
                np.stack(md_df["embedding"].values)
                if len(md_df) > 0
                else np.empty((0, 384))
            )  # (n_md, dim)

            if len(md_emb) == 0:
                continue

            # Compute Cosine Similarity Matrix: (n_md, n_code)
            sims = cosine_similarity(md_emb, code_emb)

            # Calculate Features
            # 1. Max Similarity
            sim_max = sims.max(axis=1)

            # 2. Mean Similarity
            sim_mean = sims.mean(axis=1)

            # 3. Std Similarity
            sim_std = sims.std(axis=1)

            # 4. Best Match Location (Normalized index)
            # Cite solution_lesson_node_00012
            best_match_idx = sims.argmax(axis=1)
            best_match_loc = best_match_idx / n_code

            # 5. Center of Mass
            # Cite solution_lesson_node_00004
            sims_clipped = np.maximum(sims, 0.0)
            sum_sims = sims_clipped.sum(axis=1) + 1e-6

            indices = np.arange(n_code)
            weighted_sum = (sims_clipped * indices).sum(axis=1)
            center_of_mass = (weighted_sum / sum_sims) / n_code

            # Targets
            if split in ["train", "val"]:
                # Relative rank in code sequence
                code_ranks = code_df["rank"].values
                md_ranks = md_df["rank"].values
                targets = []
                for r in md_ranks:
                    n_before = np.sum(code_ranks < r)
                    targets.append(n_before / n_code)
            else:
                targets = [-1] * len(md_df)

            # Assemble rows
            for i in range(len(md_df)):
                features_list.append(
                    {
                        "notebook_id": nb_id,
                        "cell_id": md_df.iloc[i]["cell_id"],
                        "n_code": n_code,  # Cite solution_lesson_node_00015
                        "md_len": 0,
                        "sim_max": sim_max[i],
                        "sim_mean": sim_mean[i],
                        "sim_std": sim_std[i],
                        "best_match_loc": best_match_loc[i],
                        "center_of_mass": center_of_mass[i],
                        "target": targets[i],
                    }
                )

        df_out = pd.DataFrame(features_list)
        print(f"Saving {len(df_out)} tabular features to {output_path}...")
        df_out.to_parquet(output_path, index=False)
        return df_out
