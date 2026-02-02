import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config
from library.data_loader import load_dataset


class VectorizationPipeline:
    """
    Manages the TF-IDF and SVD (LSA) models.
    Ensures consistent vectorization across Train, Val, and Test splits.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE,
            ngram_range=Config.NGRAM_RANGE,
            token_pattern=Config.TOKEN_PATTERN,
            sublinear_tf=Config.SUBLINEAR_TF,
            strip_accents=Config.STRIP_ACCENTS,
        )
        self.svd = TruncatedSVD(
            n_components=Config.SVD_COMPONENTS, random_state=Config.SVD_RANDOM_STATE
        )
        self.is_fitted = False

    def fit(self, texts):
        """
        Fits the TF-IDF vectorizer and SVD on the provided texts.
        """
        print("Fitting VectorizationPipeline...")
        print(f"  - Fitting TF-IDF (Vocab Size: {Config.VOCAB_SIZE})...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        print(f"  - Fitting SVD (Components: {Config.SVD_COMPONENTS})...")
        self.svd.fit(tfidf_matrix)

        self.is_fitted = True
        return self

    def transform_tfidf(self, texts):
        """Returns the sparse TF-IDF matrix."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        return self.vectorizer.transform(texts)

    def transform_lsa(self, tfidf_matrix):
        """Returns the dense LSA matrix from a TF-IDF matrix."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        return self.svd.transform(tfidf_matrix)

    def save(self, dir_path):
        """Saves the fitted models to disk."""
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(dir_path, "tfidf_vectorizer.joblib"))
        joblib.dump(self.svd, os.path.join(dir_path, "svd_model.joblib"))
        print(f"VectorizationPipeline saved to {dir_path}")

    def load(self, dir_path):
        """Loads the fitted models from disk."""
        tfidf_path = os.path.join(dir_path, "tfidf_vectorizer.joblib")
        svd_path = os.path.join(dir_path, "svd_model.joblib")

        if not os.path.exists(tfidf_path) or not os.path.exists(svd_path):
            raise FileNotFoundError(f"Pipeline files not found in {dir_path}")

        self.vectorizer = joblib.load(tfidf_path)
        self.svd = joblib.load(svd_path)
        self.is_fitted = True
        print(f"VectorizationPipeline loaded from {dir_path}")
        return self


class AnchorEngine:
    """
    Computes Semantic Anchor Features by finding the nearest code cell neighbors
    for each markdown cell.
    """

    @staticmethod
    def compute_features(df_md, df_nb, pipeline, split_name, load_cached_data=True):
        """
        Computes or loads cached anchor features.

        Features:
        - anchor_rank: Normalized rank of the most similar code cell.
        - anchor_sim: Cosine similarity to the most similar code cell.
        - top3_anchor_rank_mean: Average rank of the top 3 most similar code cells.
        """
        cache_path = os.path.join(
            Config.WORKING_DIR, f"{split_name}_anchor_features.parquet"
        )

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{split_name.upper()}] Loading cached anchor features...")
            return pd.read_parquet(cache_path)

        # 2. Compute Features
        print(
            f"[{split_name.upper()}] Computing anchor features (this may take time)..."
        )

        # Pre-compute TF-IDF for all markdown cells (Batch operation is faster)
        print(f"[{split_name.upper()}] Vectorizing {len(df_md)} markdown cells...")
        md_texts = df_md["source"].fillna("").astype(str).tolist()
        md_tfidf_all = pipeline.transform_tfidf(md_texts)

        # Create a map for quick access to code sources
        nb_code_map = dict(zip(df_nb["notebook_id"], df_nb["code_sources"]))

        # Prepare result arrays
        n_samples = len(df_md)
        anchor_ranks = np.zeros(n_samples, dtype=np.float32)
        anchor_sims = np.zeros(n_samples, dtype=np.float32)
        top3_means = np.zeros(n_samples, dtype=np.float32)

        # Group by notebook to process contexts
        # We use the group indices to map results back to the original DataFrame order
        grouped = df_md.groupby("notebook_id")

        # Iterate over notebooks
        # Note: We avoid tqdm here to comply with "silent" requirements, printing checkpoints instead.
        processed_count = 0
        total_groups = len(grouped)
        checkpoint_step = max(1, total_groups // 5)

        for nb_id, group_indices in grouped.indices.items():
            code_sources = nb_code_map.get(nb_id, [])

            # Handle case with no code cells
            if not code_sources:
                # Default to neutral/middle rank
                anchor_ranks[group_indices] = 0.5
                anchor_sims[group_indices] = 0.0
                top3_means[group_indices] = 0.5
                continue

            # Vectorize code cells for this notebook
            # (n_code, vocab_size)
            code_tfidf = pipeline.transform_tfidf(code_sources)

            # Get corresponding markdown vectors
            # (n_md_in_nb, vocab_size)
            md_vecs = md_tfidf_all[group_indices]

            # Compute Similarity Matrix (Dot product of L2-normalized TF-IDF vectors = Cosine Sim)
            # (n_md_in_nb, n_code)
            sim_matrix = md_vecs.dot(code_tfidf.T)

            # Convert to dense array for aggregation
            sim_dense = sim_matrix.toarray()

            # Find best match
            best_indices = np.argmax(sim_dense, axis=1)
            best_sims = np.max(sim_dense, axis=1)

            n_code = len(code_sources)
            if n_code > 1:
                norm_ranks = best_indices / (n_code - 1)
            else:
                norm_ranks = np.zeros_like(best_indices, dtype=float)

            # Top 3 Mean Rank
            if n_code >= 3:
                # Get indices of top 3 similarities
                # argsort sorts ascending, so we take last 3
                top3_idx = np.argsort(sim_dense, axis=1)[:, -3:]
                top3_r = top3_idx / (n_code - 1)
                top3_mean = np.mean(top3_r, axis=1)
            else:
                top3_mean = norm_ranks

            # Assign to result arrays
            anchor_ranks[group_indices] = norm_ranks
            anchor_sims[group_indices] = best_sims
            top3_means[group_indices] = top3_mean

            processed_count += 1
            if processed_count % checkpoint_step == 0:
                print(
                    f"[{split_name.upper()}] Processed {processed_count}/{total_groups} notebooks..."
                )

        # 3. Create DataFrame and Cache
        df_features = pd.DataFrame(
            {
                "anchor_rank": anchor_ranks,
                "anchor_sim": anchor_sims,
                "top3_anchor_rank_mean": top3_means,
            },
            index=df_md.index,
        )

        df_features.to_parquet(cache_path, index=True)
        print(f"[{split_name.upper()}] Anchor features saved to {cache_path}")

        return df_features


class FeatureEngineer:
    """
    Orchestrates the feature engineering process.
    """

    def __init__(self, load_cached_data=True):
        self.load_cached_data = load_cached_data
        self.pipeline = VectorizationPipeline()
        self.pipeline_dir = Config.WORKING_DIR

    def process_split(self, split="train"):
        """
        Loads data, fits/loads pipeline, computes features, and returns the enriched DataFrame.
        """
        print(f"\n=== Processing Split: {split.upper()} ===")

        # 1. Load Raw Data
        df_md, df_nb = load_dataset(split, load_cached_data=self.load_cached_data)

        # 2. Pipeline Management
        if split == "train":
            # Fit on training data
            # We use a sample of markdown cells + code cells to build vocab if dataset is huge,
            # but here we use all markdown cells as per standard practice.
            print(f"[{split.upper()}] Fitting pipeline on markdown corpus...")
            all_text = df_md["source"].fillna("").astype(str).tolist()
            self.pipeline.fit(all_text)
            self.pipeline.save(self.pipeline_dir)
        else:
            # Load fitted pipeline
            self.pipeline.load(self.pipeline_dir)

        # 3. Generate LSA Features
        lsa_cache_path = os.path.join(Config.WORKING_DIR, f"{split}_lsa.parquet")
        if self.load_cached_data and os.path.exists(lsa_cache_path):
            print(f"[{split.upper()}] Loading cached LSA features...")
            df_lsa = pd.read_parquet(lsa_cache_path)
        else:
            print(f"[{split.upper()}] Generating LSA features...")
            tfidf = self.pipeline.transform_tfidf(
                df_md["source"].fillna("").astype(str)
            )
            lsa_matrix = self.pipeline.transform_lsa(tfidf)

            lsa_cols = [f"lsa_{i}" for i in range(Config.SVD_COMPONENTS)]
            df_lsa = pd.DataFrame(lsa_matrix, columns=lsa_cols, index=df_md.index)
            # Reduce memory
            df_lsa = df_lsa.astype(np.float32)
            df_lsa.to_parquet(lsa_cache_path, index=True)

        # 4. Generate Anchor Features
        df_anchor = AnchorEngine.compute_features(
            df_md, df_nb, self.pipeline, split, load_cached_data=self.load_cached_data
        )

        # 5. Generate Metadata Features
        # Map total cells from df_nb to df_md
        nb_meta = df_nb.set_index("notebook_id")
        nb_meta["num_code_cells"] = nb_meta["code_sources"].apply(len)

        # We need total cells (md + code). We can count md cells from df_md.
        md_counts = df_md["notebook_id"].value_counts()
        nb_meta["num_md_cells"] = md_counts
        nb_meta["total_cells"] = nb_meta["num_code_cells"] + nb_meta[
            "num_md_cells"
        ].fillna(0)
        nb_meta["md_ratio"] = nb_meta["num_md_cells"] / nb_meta["total_cells"]

        # Map back to df_md
        df_meta_feats = df_md[["notebook_id"]].join(
            nb_meta[["total_cells", "md_ratio"]], on="notebook_id"
        )
        df_meta_feats = df_meta_feats.drop(columns=["notebook_id"])

        # 6. Assemble Final DataFrame
        print(f"[{split.upper()}] Assembling final feature set...")
        # Concatenate horizontally
        df_final = pd.concat([df_md, df_lsa, df_anchor, df_meta_feats], axis=1)

        print(f"[{split.upper()}] Final shape: {df_final.shape}")
        return df_final, df_nb
