import os
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import load_or_save_cache, set_seed


class DualViewFeatureGenerator:
    """
    Generates features for the Dual-View Stacked Ranking model.
    Includes Lexical (TF-IDF) and Semantic (Dense Embedding) anchoring, plus LSA context.
    """

    def __init__(self):
        self.tfidf_vectorizer = None
        self.lsa_model = None
        self.sentence_transformer = None
        self.device = Config.DEVICE
        set_seed(Config.SEED)

    def _get_tfidf_model(self, corpus=None, load_cached=True):
        """
        Loads or fits the TfidfVectorizer.
        """
        if load_cached and os.path.exists(Config.TFIDF_VECTORIZER_PATH):
            return joblib.load(Config.TFIDF_VECTORIZER_PATH)

        if corpus is None:
            raise ValueError(
                "Corpus required to fit TfidfVectorizer if cache not found."
            )

        print("Fitting TfidfVectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            sublinear_tf=Config.TFIDF_SUBLINEAR_TF,
            use_idf=Config.TFIDF_USE_IDF,
            strip_accents=Config.TFIDF_STRIP_ACCENTS,
            min_df=2,
        )
        vectorizer.fit(corpus)

        # Save model
        os.makedirs(os.path.dirname(Config.TFIDF_VECTORIZER_PATH), exist_ok=True)
        joblib.dump(vectorizer, Config.TFIDF_VECTORIZER_PATH)
        return vectorizer

    def _get_lsa_model(self, tfidf_matrix=None, load_cached=True):
        """
        Loads or fits the TruncatedSVD model.
        """
        if load_cached and os.path.exists(Config.LSA_MODEL_PATH):
            return joblib.load(Config.LSA_MODEL_PATH)

        if tfidf_matrix is None:
            raise ValueError("TF-IDF matrix required to fit LSA if cache not found.")

        print("Fitting LSA (TruncatedSVD)...")
        lsa = TruncatedSVD(n_components=Config.LSA_COMPONENTS, random_state=Config.SEED)
        lsa.fit(tfidf_matrix)

        # Save model
        os.makedirs(os.path.dirname(Config.LSA_MODEL_PATH), exist_ok=True)
        joblib.dump(lsa, Config.LSA_MODEL_PATH)
        return lsa

    def _get_sentence_transformer(self):
        """
        Loads the SentenceTransformer model.
        """
        if self.sentence_transformer is None:
            print(f"Loading SentenceTransformer: {Config.SENTENCE_TRANSFORMER_MODEL}")
            self.sentence_transformer = SentenceTransformer(
                Config.SENTENCE_TRANSFORMER_MODEL
            )
            self.sentence_transformer.to(self.device)
        return self.sentence_transformer

    def _compute_dense_embeddings(self, text_list, cache_name, load_cached_data):
        """
        Generates or loads dense embeddings using SentenceTransformer.
        """

        def _produce_embeddings():
            model = self._get_sentence_transformer()
            print(f"Encoding {len(text_list)} texts with SentenceTransformer...")
            embeddings = model.encode(
                text_list,
                batch_size=Config.SENTENCE_TRANSFORMER_BATCH_SIZE,
                show_progress_bar=True,
                device=self.device,
                convert_to_numpy=True,
            )
            return embeddings

        return load_or_save_cache(
            file_name=cache_name,
            data_producer_fn=_produce_embeddings,
            load_cached_data=load_cached_data,
        )

    def _compute_anchor_features_core(
        self, df_md, df_code, tfidf_md, tfidf_code, dense_md, dense_code
    ):
        """
        Core logic to compute Dual-View Anchors (Lexical & Semantic).
        Iterates through notebooks and performs matrix operations.
        """
        # Ensure dataframes are sorted by notebook_id to allow grouping/slicing
        # Note: We assume the input matrices align with the dataframe indices.
        # We will use notebook_id grouping.

        # Pre-compute notebook groups
        md_groups = df_md.groupby("notebook_id").indices
        code_groups = df_code.groupby("notebook_id").indices

        # Results containers
        lex_ranks = np.zeros(len(df_md), dtype=np.float32)
        lex_sims = np.zeros(len(df_md), dtype=np.float32)
        sem_ranks = np.zeros(len(df_md), dtype=np.float32)
        sem_sims = np.zeros(len(df_md), dtype=np.float32)

        # List of notebook IDs present in markdown dataframe
        notebook_ids = df_md["notebook_id"].unique()

        print("Computing Dual-View Anchor features...")
        for nb_id in tqdm(notebook_ids, desc="Anchoring"):
            if nb_id not in code_groups:
                # No code cells in this notebook. Assign defaults.
                # Default rank: 0.5 (middle), Default sim: 0.0
                md_idxs = md_groups[nb_id]
                lex_ranks[md_idxs] = 0.5
                lex_sims[md_idxs] = 0.0
                sem_ranks[md_idxs] = 0.5
                sem_sims[md_idxs] = 0.0
                continue

            md_idxs = md_groups[nb_id]
            code_idxs = code_groups[nb_id]

            # --- Lexical View (TF-IDF) ---
            # Slice matrices
            curr_tfidf_md = tfidf_md[md_idxs]
            curr_tfidf_code = tfidf_code[code_idxs]

            # Compute Cosine Similarity (Sparse dot product)
            # Result shape: (n_md, n_code)
            sim_matrix_lex = cosine_similarity(curr_tfidf_md, curr_tfidf_code)

            # Find best match
            best_code_idx_lex = np.argmax(sim_matrix_lex, axis=1)
            best_sim_lex = np.max(sim_matrix_lex, axis=1)

            # --- Semantic View (Dense) ---
            curr_dense_md = dense_md[md_idxs]
            curr_dense_code = dense_code[code_idxs]

            # Compute Cosine Similarity (Dense dot product)
            sim_matrix_sem = cosine_similarity(curr_dense_md, curr_dense_code)

            # Find best match
            best_code_idx_sem = np.argmax(sim_matrix_sem, axis=1)
            best_sim_sem = np.max(sim_matrix_sem, axis=1)

            # --- Map to Ranks ---
            # The 'rank' column in df_code gives the integer position (0, 1, 2...)
            # We need to retrieve the rank of the matched code cell.
            # code_idxs is a list of indices in df_code.
            # best_code_idx_* is the index relative to this notebook's code list (0..n_code-1).
            # So we look up the rank from df_code.

            # Get the actual ranks of code cells in this notebook
            nb_code_ranks = df_code.iloc[code_idxs]["rank"].values
            n_code = len(nb_code_ranks)

            # Map relative index to rank
            matched_ranks_lex = nb_code_ranks[best_code_idx_lex]
            matched_ranks_sem = nb_code_ranks[best_code_idx_sem]

            # Normalize Ranks: rank / (n_code - 1) if n_code > 1 else 0.5
            if n_code > 1:
                norm_ranks_lex = matched_ranks_lex / (n_code - 1)
                norm_ranks_sem = matched_ranks_sem / (n_code - 1)
            else:
                norm_ranks_lex = np.full_like(matched_ranks_lex, 0.5, dtype=np.float32)
                norm_ranks_sem = np.full_like(matched_ranks_sem, 0.5, dtype=np.float32)

            # Store
            lex_ranks[md_idxs] = norm_ranks_lex
            lex_sims[md_idxs] = best_sim_lex
            sem_ranks[md_idxs] = norm_ranks_sem
            sem_sims[md_idxs] = best_sim_sem

        return lex_ranks, lex_sims, sem_ranks, sem_sims

    def process_data(self, df_markdown, df_code, split="train", load_cached_data=True):
        """
        Main entry point to generate all features for a given split.
        Uses caching to avoid re-computation.
        """

        # Define the producer function for the entire feature set
        def _produce_final_features(df_md=df_markdown, df_cd=df_code, mode=split):
            print(f"Generating features for split: {mode}")

            # 1. Text Extraction
            md_text = df_md["source"].fillna("").astype(str).tolist()
            code_text = df_cd["source"].fillna("").astype(str).tolist()

            # 2. TF-IDF Processing
            if mode == "train":
                self.tfidf_vectorizer = self._get_tfidf_model(
                    corpus=md_text, load_cached=load_cached_data
                )
            else:
                self.tfidf_vectorizer = self._get_tfidf_model(
                    corpus=None, load_cached=True
                )

            print("Transforming text to TF-IDF...")
            tfidf_md = self.tfidf_vectorizer.transform(md_text)
            tfidf_code = self.tfidf_vectorizer.transform(code_text)

            # 3. LSA Processing (Markdown Only)
            if mode == "train":
                self.lsa_model = self._get_lsa_model(
                    tfidf_matrix=tfidf_md, load_cached=load_cached_data
                )
            else:
                self.lsa_model = self._get_lsa_model(
                    tfidf_matrix=None, load_cached=True
                )

            print("Transforming TF-IDF to LSA...")
            lsa_features = self.lsa_model.transform(tfidf_md)

            # 4. Dense Embeddings
            # Cache keys need to be specific to the split
            dense_md = self._compute_dense_embeddings(
                md_text, f"{mode}_dense_md.npy", load_cached_data
            )
            dense_code = self._compute_dense_embeddings(
                code_text, f"{mode}_dense_code.npy", load_cached_data
            )

            # 5. Dual-View Anchoring
            lex_rank, lex_sim, sem_rank, sem_sim = self._compute_anchor_features_core(
                df_md, df_cd, tfidf_md, tfidf_code, dense_md, dense_code
            )

            # 6. Assemble DataFrame
            # Start with original metadata/targets
            df_features = df_md.copy()

            # Add Anchor Features
            df_features["lexical_anchor_rank"] = lex_rank
            df_features["lexical_anchor_sim"] = lex_sim
            df_features["semantic_anchor_rank"] = sem_rank
            df_features["semantic_anchor_sim"] = sem_sim

            # Add LSA Features
            lsa_cols = [f"lsa_{i}" for i in range(Config.LSA_COMPONENTS)]
            df_lsa = pd.DataFrame(
                lsa_features, columns=lsa_cols, index=df_features.index
            )
            df_features = pd.concat([df_features, df_lsa], axis=1)

            # Add basic metadata features
            # (e.g., relative position if not using rank directly, or length)
            df_features["char_len"] = df_features["source"].apply(len)
            df_features["word_len"] = df_features["source"].apply(
                lambda x: len(x.split())
            )

            return df_features

        # Determine cache filename
        cache_file = f"{split}_features.parquet"

        # Execute load or generate
        df_final = load_or_save_cache(
            file_name=cache_file,
            data_producer_fn=_produce_final_features,
            load_cached_data=load_cached_data,
        )

        return df_final
