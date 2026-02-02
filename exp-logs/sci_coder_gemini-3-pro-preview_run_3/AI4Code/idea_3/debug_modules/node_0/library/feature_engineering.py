import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import config
from library.utils import set_seed, preprocess_text
from library.dataset import NotebookLoader


class FeatureExtractor:
    """
    Extracts semantic alignment features using a fine-tuned SentenceTransformer.
    Generates tabular features for the LightGBM regressor.
    """

    def __init__(self):
        """
        Initializes the FeatureExtractor by loading the semantic model.
        Prioritizes the fine-tuned model; falls back to the base model if not found.
        """
        self.device = config.DEVICE
        self.model_path = config.FINE_TUNED_MODEL_PATH

        # Load model logic
        if os.path.exists(self.model_path):
            print(f"Loading fine-tuned model from {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
        else:
            print(
                f"Fine-tuned model not found at {self.model_path}. Loading base model {config.BASE_MODEL_NAME}"
            )
            self.model = SentenceTransformer(config.BASE_MODEL_NAME)

        self.model.to(self.device)
        self.model.max_seq_length = config.MAX_LENGTH

    def extract_features(
        self, metadata_path, save_path, mode="train", load_cached_data=True
    ):
        """
        Main pipeline to extract features for a dataset.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            save_path (str): Path to save or load the Parquet cache.
            mode (str): 'train', 'val', or 'test'. Determines if targets are calculated.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame containing extracted features and metadata.
        """
        set_seed(config.SEED)

        # Adjust save path for debug mode to avoid overwriting the full dataset cache
        if config.DEBUG_SAMPLE_SIZE is not None:
            base, ext = os.path.splitext(save_path)
            save_path = f"{base}_debug{config.DEBUG_SAMPLE_SIZE}{ext}"

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(save_path):
            print(f"Loading features from cache: {save_path}")
            try:
                df = pd.read_parquet(save_path)
                return df
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing features from scratch...")

        # 2. Compute Features
        print(f"Extracting features from {metadata_path} (Mode: {mode})...")
        loader = NotebookLoader(metadata_path)

        # Determine subset for debug mode
        indices = range(len(loader))
        if config.DEBUG_SAMPLE_SIZE is not None:
            limit = min(len(loader), config.DEBUG_SAMPLE_SIZE)
            indices = range(limit)
            print(f"Debug mode: Processing first {limit} notebooks.")

        features_list = []

        # Iterate through notebooks
        for i, idx in enumerate(indices):
            # Periodic progress update
            if i % 100 == 0:
                print(f"Processing notebook {i}/{len(indices)}...", end="\r")

            notebook_id, data, cell_order = loader[idx]

            if data is None:
                continue

            nb_features = self._process_notebook(notebook_id, data, cell_order, mode)
            if nb_features:
                features_list.extend(nb_features)

        print(
            f"\nProcessed {len(indices)} notebooks. Total rows extracted: {len(features_list)}"
        )
        df = pd.DataFrame(features_list)

        # 3. Save to Cache
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            df.to_parquet(save_path, index=False)
            print(f"Saved features to {save_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return df

    def _process_notebook(self, notebook_id, data, cell_order, mode):
        """
        Extracts features for a single notebook.

        Args:
            notebook_id (str): ID of the notebook.
            data (dict): Notebook content (cell_type, source).
            cell_order (list): Ground truth cell order (or None for test).
            mode (str): 'train', 'val', or 'test'.

        Returns:
            list: List of dictionaries, one per markdown cell.
        """
        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        code_ids = []
        md_ids = []

        # Identify Code and Markdown cells
        # For 'train'/'val', we use cell_order to establish the correct sequence of code cells.
        # For 'test', we rely on the dictionary insertion order (assuming valid JSON structure).

        if cell_order:
            for cid in cell_order:
                # Ensure cid exists in cell_types (robustness)
                if cid not in cell_types:
                    continue
                ctype = cell_types[cid]
                if ctype == "code":
                    code_ids.append(cid)
                elif ctype == "markdown":
                    md_ids.append(cid)
        else:
            # Test mode: separate by type, assume JSON order for code is correct
            for cid, ctype in cell_types.items():
                if ctype == "code":
                    code_ids.append(cid)
                elif ctype == "markdown":
                    md_ids.append(cid)

        n_code = len(code_ids)
        # If there are no code cells to anchor to, or no markdown cells to sort, skip.
        if n_code == 0 or len(md_ids) == 0:
            return []

        # Preprocess texts
        code_texts = [preprocess_text(sources.get(cid, "")) for cid in code_ids]
        md_texts = [preprocess_text(sources.get(cid, "")) for cid in md_ids]

        # Generate Embeddings
        # We use internal batching within encode() for efficiency.
        with torch.no_grad():
            code_emb = self.model.encode(
                code_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            md_emb = self.model.encode(
                md_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )

        # Compute Similarity Matrix (Markdown x Code)
        # Shape: (n_md, n_code)
        sim_matrix = cosine_similarity(md_emb, code_emb)

        # Calculate Targets (only for train/val)
        targets = {}
        if mode != "test" and cell_order:
            # Determine rank of markdown cells relative to code cells
            # Rank = number of code cells appearing before the markdown cell
            curr_code_count = 0
            for cid in cell_order:
                if cid not in cell_types:
                    continue
                ctype = cell_types[cid]
                if ctype == "code":
                    curr_code_count += 1
                elif ctype == "markdown":
                    targets[cid] = curr_code_count

        # Extract Features per Markdown cell
        results = []
        for i, md_id in enumerate(md_ids):
            sims = sim_matrix[i]  # Shape (n_code,)

            # 1. Best Match Features
            best_match_idx = np.argmax(sims)
            best_match_loc = best_match_idx / n_code
            sim_max = sims[best_match_idx]

            # 2. Distribution Features
            sim_mean = np.mean(sims)

            # 3. Center of Mass (weighted by positive similarity)
            # Clip negatives to 0 to avoid distorting the weighted average
            weights = np.maximum(0, sims)
            sum_weights = np.sum(weights)

            if sum_weights > 1e-6:
                # Weighted average of indices
                com_idx = np.average(np.arange(n_code), weights=weights)
                center_of_mass = com_idx / n_code
            else:
                # Fallback if no positive similarity found
                center_of_mass = 0.5

            # 4. Global Context Features
            md_len = len(md_texts[i])

            row = {
                "id": notebook_id,
                "cell_id": md_id,
                "n_code": n_code,
                "md_len": md_len,
                "best_match_loc": best_match_loc,
                "center_of_mass": center_of_mass,
                "sim_max": sim_max,
                "sim_mean": sim_mean,
            }

            # Add Target for Training
            if mode != "test":
                # Normalized rank: [0, 1]
                # 0 means before all code, 1 means after all code
                raw_rank = targets.get(md_id, 0)
                row["target"] = raw_rank / n_code

            results.append(row)

        return results
