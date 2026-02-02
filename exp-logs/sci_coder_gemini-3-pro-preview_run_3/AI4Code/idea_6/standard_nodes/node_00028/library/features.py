import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from library.config import Config
from library.utils import read_notebook, set_seed
from library.dataset import NotebookLoader


class DualViewFeatureExtractor:
    """
    Extracts semantic and structural features using a Dual-View architecture.
    Uses two fine-tuned backbones (Text-View and Code-View) to generate
    alignment signals between markdown and code cells.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model_text = None
        self.model_code = None
        set_seed(Config.SEED)

    def _load_models(self):
        """
        Loads the Text-View and Code-View models.
        Prioritizes fine-tuned models saved in the working directory.
        Falls back to base models if fine-tuned versions are not found.
        """
        # Load Text-View Model
        if os.path.exists(Config.TEXT_MODEL_SAVE_PATH):
            print(
                f"Loading fine-tuned Text-View model from {Config.TEXT_MODEL_SAVE_PATH}"
            )
            self.model_text = SentenceTransformer(
                Config.TEXT_MODEL_SAVE_PATH, device=self.device
            )
        else:
            print(
                f"Fine-tuned Text-View model not found. Loading base: {Config.MODEL_TEXT}"
            )
            self.model_text = SentenceTransformer(Config.MODEL_TEXT, device=self.device)

        # Load Code-View Model
        if os.path.exists(Config.CODE_MODEL_SAVE_PATH):
            print(
                f"Loading fine-tuned Code-View model from {Config.CODE_MODEL_SAVE_PATH}"
            )
            self.model_code = SentenceTransformer(
                Config.CODE_MODEL_SAVE_PATH, device=self.device
            )
        else:
            print(
                f"Fine-tuned Code-View model not found. Loading base: {Config.MODEL_CODE}"
            )
            self.model_code = SentenceTransformer(Config.MODEL_CODE, device=self.device)

    def _compute_center_of_mass(self, sims):
        """
        Computes the similarity-weighted center of mass (index).
        Uses softmax to convert similarities to positive weights.

        Args:
            sims (np.array): Array of similarity scores for a single markdown cell against all code cells.

        Returns:
            float: The weighted average index.
        """
        if len(sims) == 0:
            return 0.0

        # Use softmax for stability and to ensure positive weights
        # Shift by max to avoid overflow
        exp_sims = np.exp(sims - np.max(sims))
        weights = exp_sims / np.sum(exp_sims)

        indices = np.arange(len(sims))
        com = np.sum(weights * indices)
        return com

    def extract_features(
        self, metadata_path, mode="train", debug=False, load_cached_data=True
    ):
        """
        Main method to generate or load features for a dataset.

        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'. Used for cache naming.
            debug (bool): If True, process only a small subset.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: DataFrame containing features and targets (if available).
        """
        # 1. Define Cache Path
        cache_filename = f"{mode}_features{'_debug' if debug else ''}.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 2. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        # 3. Initialize Models (Lazy Loading)
        if self.model_text is None or self.model_code is None:
            self._load_models()

        print(f"Extracting features for {mode} (debug={debug})...")

        loader = NotebookLoader(metadata_path)
        df_meta = loader.df

        if debug:
            df_meta = df_meta.head(100)

        features_list = []

        # 4. Iterate Notebooks
        for idx, row in df_meta.iterrows():
            try:
                data = read_notebook(row["file_path"])
            except Exception:
                continue

            cell_types = data.get("cell_type", {})
            sources = data.get("source", {})

            # Identify cells
            # For train/val, we might have cell_order. For test, we rely on JSON order.
            # However, to process consistently, we just need to identify Code vs Markdown.
            # In test JSONs, order of keys matters for relative code positions.

            # We need a stable list of cell_ids to index properly
            if "cell_order" in row and not pd.isna(row["cell_order"]):
                # Train/Val: Use ground truth order to identify code sequence
                full_order = str(row["cell_order"]).split()
                # Filter to existing cells
                full_order = [c for c in full_order if c in cell_types]
            else:
                # Test: Use JSON key order
                full_order = list(cell_types.keys())

            code_ids = [c for c in full_order if cell_types[c] == "code"]
            md_ids = [c for c in full_order if cell_types[c] == "markdown"]

            n_code = len(code_ids)
            if n_code == 0:
                # Edge case: No code cells. Cannot align.
                continue

            # Get text content
            code_texts = [sources.get(c, "") for c in code_ids]
            md_texts = [sources.get(c, "") for c in md_ids]

            if not md_texts:
                continue

            # 5. Generate Embeddings
            # Text-View
            code_emb_text = self.model_text.encode(
                code_texts, convert_to_tensor=True, show_progress_bar=False
            )
            md_emb_text = self.model_text.encode(
                md_texts, convert_to_tensor=True, show_progress_bar=False
            )

            # Code-View
            code_emb_code = self.model_code.encode(
                code_texts, convert_to_tensor=True, show_progress_bar=False
            )
            md_emb_code = self.model_code.encode(
                md_texts, convert_to_tensor=True, show_progress_bar=False
            )

            # 6. Compute Similarity Matrices (Markdown x Code)
            # Shape: (n_md, n_code)
            sim_matrix_text = util.cos_sim(md_emb_text, code_emb_text).cpu().numpy()
            sim_matrix_code = util.cos_sim(md_emb_code, code_emb_code).cpu().numpy()

            # 7. Extract Features per Markdown Cell
            for i, md_id in enumerate(md_ids):
                row_feats = {
                    "id": row["id"],
                    "cell_id": md_id,
                    "n_code": n_code,
                    "md_len": len(md_texts[i]),
                }

                # Text-View Features
                sims_t = sim_matrix_text[i]
                row_feats["tv_sim_max"] = np.max(sims_t)
                row_feats["tv_best_loc"] = np.argmax(sims_t) / n_code
                row_feats["tv_com"] = self._compute_center_of_mass(sims_t) / n_code

                # Code-View Features
                sims_c = sim_matrix_code[i]
                row_feats["cv_sim_max"] = np.max(sims_c)
                row_feats["cv_best_loc"] = np.argmax(sims_c) / n_code
                row_feats["cv_com"] = self._compute_center_of_mass(sims_c) / n_code

                # 8. Calculate Target (for Train/Val)
                # Target is the normalized rank: position / n_code
                # Position is defined as the number of code cells before this markdown cell
                if "cell_order" in row and not pd.isna(row["cell_order"]):
                    # Find index of md_id in the ground truth code+md sequence
                    # We can infer rank by counting code cells appearing before md_id in full_order
                    # Since full_order is the ground truth order here:
                    try:
                        idx_in_gt = full_order.index(md_id)
                        # Count code cells before this index
                        rank = 0
                        for k in range(idx_in_gt):
                            if cell_types[full_order[k]] == "code":
                                rank += 1

                        row_feats["target"] = rank / n_code
                    except ValueError:
                        # Should not happen given filtering logic
                        row_feats["target"] = 0.0

                features_list.append(row_feats)

        # 9. Create DataFrame and Cache
        df_features = pd.DataFrame(features_list)

        # Ensure directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Save to Parquet
        df_features.to_parquet(cache_path, index=False)
        print(f"Saved {len(df_features)} feature rows to {cache_path}")

        return df_features
