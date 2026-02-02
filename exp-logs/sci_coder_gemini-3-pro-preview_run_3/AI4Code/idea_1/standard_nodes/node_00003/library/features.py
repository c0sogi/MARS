import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.utils import preprocess_text
from library.data_loader import load_notebook, get_ordered_cells


class SemanticFeatureExtractor:
    """
    Handles the extraction of semantic features from notebooks using TF-IDF and
    Cosine Similarity statistics to determine the relative position of markdown cells.
    """

    def __init__(self):
        self.vectorizer_path = os.path.join(Config.CACHE_DIR, "tfidf_vectorizer.joblib")
        self.vectorizer = None

    def fit_vectorizer(self, df_metadata, sample_size=5000, load_cached=True):
        """
        Fits a TF-IDF vectorizer on a subset of the training data or loads a cached one.

        Args:
            df_metadata (pd.DataFrame): Training metadata.
            sample_size (int): Number of notebooks to sample for vocabulary building.
            load_cached (bool): Whether to try loading from disk.
        """
        # Try loading cached vectorizer
        if load_cached and os.path.exists(self.vectorizer_path):
            print(f"Loading cached TF-IDF vectorizer from {self.vectorizer_path}...")
            self.vectorizer = joblib.load(self.vectorizer_path)
            return

        print("Fitting TF-IDF vectorizer...")
        # Sample data for efficiency
        if sample_size and len(df_metadata) > sample_size:
            df_sample = df_metadata.sample(n=sample_size, random_state=Config.SEED)
        else:
            df_sample = df_metadata

        corpus = []

        # Collect text from sampled notebooks
        for _, row in df_sample.iterrows():
            try:
                nb_data = load_notebook(row["file_path"])
                # Add code texts
                for text in nb_data["code_cells"].values():
                    if text:
                        corpus.append(preprocess_text(text))
                # Add markdown texts
                for text in nb_data["markdown_cells"].values():
                    if text:
                        corpus.append(preprocess_text(text))
            except Exception:
                continue

        # Initialize and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            min_df=5,
            stop_words="english",
            dtype=np.float32,
        )
        self.vectorizer.fit(corpus)

        # Save to cache
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Vectorizer fitted and saved to {self.vectorizer_path}")

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
        cache_path = os.path.join(Config.CACHE_DIR, f"features_{mode}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            return pd.read_parquet(cache_path)

        if self.vectorizer is None:
            # Ensure vectorizer is loaded
            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
            else:
                raise RuntimeError("Vectorizer not fitted. Call fit_vectorizer first.")

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
                # load_notebook preserves insertion order of the JSON dict
                ordered_code_ids = list(code_cells.keys())

            n_code = len(ordered_code_ids)

            # Preprocess texts
            code_texts = [preprocess_text(code_cells[cid]) for cid in ordered_code_ids]
            md_ids = list(markdown_cells.keys())
            md_texts = [preprocess_text(markdown_cells[mid]) for mid in md_ids]

            # Vectorize
            # Handle case with no code cells
            if n_code == 0:
                # If no code cells, we can't compute relative features.
                # We just output basic length features and 0.5 target/prediction.
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

            # Transform
            try:
                X_code = self.vectorizer.transform(code_texts)
                X_md = self.vectorizer.transform(md_texts)
            except ValueError:
                # Handle empty vocabulary cases
                continue

            # Compute Similarity Matrix (Rows: MD, Cols: Code)
            sim_matrix = cosine_similarity(X_md, X_code)

            # Calculate Target Ranks (only for train/val)
            targets = {}
            if mode in ["train", "val"]:
                full_order = get_ordered_cells(row["cell_order"])
                # Map cell_id to its rank in the full sequence
                rank_map = {cid: i for i, cid in enumerate(full_order)}

                # Create a list of ranks for code cells to compare against
                code_ranks = [rank_map[cid] for cid in ordered_code_ids]

                for mid in md_ids:
                    if mid in rank_map:
                        my_rank = rank_map[mid]
                        # Count how many code cells appear before this markdown cell
                        # This is the relative position index (0 to n_code)
                        pos = sum(1 for r in code_ranks if r < my_rank)
                        # Normalize target to [0, 1]
                        targets[mid] = pos / n_code
                    else:
                        targets[mid] = 0.5

            # Extract features for each markdown cell
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
                # sum(sim * index) / sum(sim)
                total_sim = np.sum(sim_row)
                if total_sim > 1e-6:
                    # Indices are 0 to n_code-1
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
