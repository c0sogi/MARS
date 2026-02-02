import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config


class DataPreprocessor:
    def __init__(self):
        """
        Initialize the DataPreprocessor with the specific backbone model.
        """
        self.device = Config.DEVICE
        print(f"Initializing DataPreprocessor with backbone: {Config.BACKBONE_NAME}")

        # Load the pre-trained model
        self.model = SentenceTransformer(Config.BACKBONE_NAME, device=str(self.device))

        # Enforce maximum sequence length from configuration
        self.model.max_seq_length = Config.MAX_LENGTH
        self.model.eval()

    def _read_notebook(self, filepath, cell_order=None):
        """
        Reads a notebook JSON file and extracts code and markdown cells.

        Args:
            filepath (str): Relative path to the JSON file.
            cell_order (str, optional): Space-delimited string of cell IDs representing the correct order.
                                        Used for Train/Val sets to correctly identify the anchor sequence.

        Returns:
            tuple: (code_ids, code_texts, markdown_ids, markdown_texts)
        """
        full_path = os.path.join(Config.INPUT_DIR, filepath)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            return None, None, None, None

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        code_ids = []
        markdown_ids = []

        # Determine the processing order of cells
        if cell_order:
            # For Train/Val: Use the ground truth cell_order to identify the correct sequence of code cells
            order_list = cell_order.split()
            for cid in order_list:
                ctype = cell_types.get(cid)
                if ctype == "code":
                    code_ids.append(cid)
                elif ctype == "markdown":
                    markdown_ids.append(cid)
        else:
            # For Test: Use the order of keys in the JSON file
            # We assume the provided JSON structure preserves the relative order of code cells
            for cid, ctype in cell_types.items():
                if ctype == "code":
                    code_ids.append(cid)
                elif ctype == "markdown":
                    markdown_ids.append(cid)

        # Extract source text
        # Ensure text is a string; handle potential missing keys gracefully
        code_texts = [sources.get(cid, "") for cid in code_ids]
        markdown_texts = [sources.get(cid, "") for cid in markdown_ids]

        return code_ids, code_texts, markdown_ids, markdown_texts

    def process_and_cache(self, metadata_path, cache_path, desc, load_cached_data=True):
        """
        Processes a dataset split (Train/Val/Test), generates embeddings, and caches the result.

        Args:
            metadata_path (str): Path to the metadata CSV.
            cache_path (str): Path where the Parquet file should be saved.
            desc (str): Description of the split (e.g., "Train").
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed dataframe containing embeddings.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{desc}] Cache found at {cache_path}. Loading...")
            return pd.read_parquet(cache_path)

        # 2. Process Data
        print(f"[{desc}] Cache not found or reload requested. Processing...")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Debug Mode: Sample data
        if Config.DEBUG:
            print(
                f"[{desc}] Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks."
            )
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        ids = df_meta["id"].tolist()
        filepaths = df_meta["filepath"].tolist()
        # Get cell_order if available (Train/Val), else None (Test)
        cell_orders = (
            df_meta["cell_order"].tolist()
            if "cell_order" in df_meta.columns
            else [None] * len(ids)
        )

        out_ids = []
        out_code_ids = []
        out_md_ids = []
        out_code_embs = []
        out_md_embs = []

        total = len(ids)
        print(f"[{desc}] Encoding {total} notebooks...")

        for i in range(total):
            nb_id = ids[i]
            fpath = filepaths[i]
            corder = cell_orders[i]

            c_ids, c_texts, m_ids, m_texts = self._read_notebook(fpath, corder)

            if c_ids is None:
                continue

            # Combine texts for efficient batch encoding within the notebook
            all_texts = c_texts + m_texts

            c_embs_list = []
            m_embs_list = []

            if all_texts:
                # Generate embeddings
                # normalize_embeddings=True ensures compatibility with dot-product/cosine similarity
                embeddings = self.model.encode(
                    all_texts,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                # Split embeddings back into code and markdown lists
                split_idx = len(c_texts)
                c_embs_list = embeddings[:split_idx].tolist()
                m_embs_list = embeddings[split_idx:].tolist()

            out_ids.append(nb_id)
            out_code_ids.append(c_ids)
            out_md_ids.append(m_ids)
            out_code_embs.append(c_embs_list)
            out_md_embs.append(m_embs_list)

            if (i + 1) % 1000 == 0:
                print(f"[{desc}] Processed {i + 1}/{total} notebooks", end="\r")

        print(f"\n[{desc}] Finished processing. Saving to {cache_path}...")

        # Construct DataFrame
        df_out = pd.DataFrame(
            {
                "id": out_ids,
                "code_cell_ids": out_code_ids,
                "markdown_cell_ids": out_md_ids,
                "code_embeddings": out_code_embs,
                "markdown_embeddings": out_md_embs,
            }
        )

        # Save to Parquet
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_out.to_parquet(cache_path, index=False)

        return df_out

    def run(self, load_cached_data=True):
        """
        Main execution method to process Train, Validation, and Test datasets.
        """
        # Ensure reproducibility
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)

        # Process Train Set
        self.process_and_cache(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_CACHE_PATH,
            "Train",
            load_cached_data,
        )

        # Process Validation Set
        self.process_and_cache(
            Config.VAL_METADATA_PATH,
            Config.VAL_CACHE_PATH,
            "Validation",
            load_cached_data,
        )

        # Process Test Set
        self.process_and_cache(
            Config.TEST_METADATA_PATH, Config.TEST_CACHE_PATH, "Test", load_cached_data
        )

        print("Data Preprocessing Completed Successfully.")
