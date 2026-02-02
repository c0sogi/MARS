import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, set_seed


class FeatureEngineeringPipeline:
    def __init__(self, load_cached_data: bool = True):
        self.load_cached_data = load_cached_data
        self.logger = setup_logger("FeatureEngineering")
        self.cache_dir = Config.WORKING_DIR

        # Define cache filenames
        self.splits = ["train", "val", "test"]
        self.feature_types = ["lexical", "behavioral", "semantic", "metadata"]

        set_seed(Config.RANDOM_SEED)

    def _get_cache_path(self, split, feature_type):
        ext = "npz" if feature_type in ["lexical", "behavioral"] else "npy"
        return os.path.join(self.cache_dir, f"X_{split}_{feature_type}.{ext}")

    def _get_target_path(self, split):
        return os.path.join(self.cache_dir, f"y_{split}.npy")

    def _check_cache_exists(self):
        for split in self.splits:
            # Check features
            for ft in self.feature_types:
                if not os.path.exists(self._get_cache_path(split, ft)):
                    return False
            # Check targets (only for train/val)
            if split != "test":
                if not os.path.exists(self._get_target_path(split)):
                    return False
        return True

    def _save_cache(self, data_dict):
        self.logger.info("Saving features to cache...")
        for split in self.splits:
            # Save features
            split_data = data_dict[split]

            # Sparse matrices
            sp.save_npz(self._get_cache_path(split, "lexical"), split_data["lexical"])
            sp.save_npz(
                self._get_cache_path(split, "behavioral"), split_data["behavioral"]
            )

            # Dense arrays
            np.save(self._get_cache_path(split, "semantic"), split_data["semantic"])
            np.save(self._get_cache_path(split, "metadata"), split_data["metadata"])

            # Save target if available
            if "y" in split_data:
                np.save(self._get_target_path(split), split_data["y"])

    def _load_cache(self):
        self.logger.info("Loading features from cache...")
        data_dict = {}
        for split in self.splits:
            split_data = {}
            split_data["lexical"] = sp.load_npz(self._get_cache_path(split, "lexical"))
            split_data["behavioral"] = sp.load_npz(
                self._get_cache_path(split, "behavioral")
            )
            split_data["semantic"] = np.load(self._get_cache_path(split, "semantic"))
            split_data["metadata"] = np.load(self._get_cache_path(split, "metadata"))

            if split != "test":
                split_data["y"] = np.load(self._get_target_path(split))

            data_dict[split] = split_data
        return data_dict

    def _prepare_text(self, df):
        # Combine title and edit-aware text
        title = df[Config.TITLE_COL].fillna("").astype(str)
        body = df[Config.TEXT_COL].fillna("").astype(str)
        return title + " " + body

    def _prepare_subreddits(self, df):
        # Convert list of subreddits to space-separated string
        def join_subs(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join(x)
            return ""

        return df["requester_subreddits_at_request"].apply(join_subs)

    def run(self):
        # 1. Check Cache
        if self.load_cached_data and self._check_cache_exists():
            return self._load_cache()

        self.logger.info(
            "Cache not found or ignored. generating features from scratch..."
        )

        # 2. Load Data
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        df_val = pd.read_parquet(Config.VAL_PATH)
        df_test = pd.read_parquet(Config.TEST_PATH)

        # Extract Targets
        y_train = df_train[Config.TARGET_COL].values
        y_val = df_val[Config.TARGET_COL].values

        # 3. Text Processing (Lexical)
        self.logger.info("Generating Lexical Features (TF-IDF)...")
        text_train = self._prepare_text(df_train)
        text_val = self._prepare_text(df_val)
        text_test = self._prepare_text(df_test)

        tfidf_text = TfidfVectorizer(**Config.TEXT_PARAMS)
        X_train_lexical = tfidf_text.fit_transform(text_train)
        X_val_lexical = tfidf_text.transform(text_val)
        X_test_lexical = tfidf_text.transform(text_test)

        # 4. Behavioral Processing (Subreddit History)
        self.logger.info("Generating Behavioral Features (Subreddit TF-IDF)...")
        subs_train = self._prepare_subreddits(df_train)
        subs_val = self._prepare_subreddits(df_val)
        subs_test = self._prepare_subreddits(df_test)

        # Reuse text params but ensure we capture subreddit tokens
        # We use the same params as text (unigrams/bigrams) to capture patterns like "random acts"
        # Disable stop words for subreddits as they are identifiers
        subs_params = Config.TEXT_PARAMS.copy()
        subs_params["stop_words"] = None
        tfidf_subs = TfidfVectorizer(**subs_params)
        X_train_behavioral = tfidf_subs.fit_transform(subs_train)
        X_val_behavioral = tfidf_subs.transform(subs_val)
        X_test_behavioral = tfidf_subs.transform(subs_test)

        # 5. Semantic Processing (Embeddings)
        self.logger.info("Generating Semantic Features (Embeddings)...")
        # Load model once
        model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode Text
        X_train_semantic = model.encode(text_train.tolist(), show_progress_bar=False)
        X_val_semantic = model.encode(text_val.tolist(), show_progress_bar=False)
        X_test_semantic = model.encode(text_test.tolist(), show_progress_bar=False)

        # Encode History (for Cross-Modal Interaction)
        # Handle empty history by encoding empty string (results in vector, usually not zero)
        hist_train_emb = model.encode(subs_train.tolist(), show_progress_bar=False)
        hist_val_emb = model.encode(subs_val.tolist(), show_progress_bar=False)
        hist_test_emb = model.encode(subs_test.tolist(), show_progress_bar=False)

        # 6. Metadata & Interaction Processing
        self.logger.info("Generating Metadata & Interaction Features...")

        def process_metadata(
            df, text_emb, hist_emb, is_train=False, imputer=None, scaler=None
        ):
            # Select numerical columns
            meta_df = df[Config.NUMERICAL_ALLOW_LIST].copy()

            # Compute Cosine Similarity between Text and History
            # paired_cosine_distances returns 1 - cos_sim. We want cos_sim.
            # We add a small epsilon to avoid division by zero if vectors are zero (unlikely with this model)
            cos_sim = 1 - paired_cosine_distances(text_emb, hist_emb)
            meta_df["text_history_similarity"] = cos_sim

            # Convert to numpy
            X_meta = meta_df.values

            # Impute
            if is_train:
                imputer = SimpleImputer(strategy="median")
                X_meta = imputer.fit_transform(X_meta)
            else:
                X_meta = imputer.transform(X_meta)

            # Scale
            if is_train:
                scaler = StandardScaler()
                X_meta = scaler.fit_transform(X_meta)
            else:
                X_meta = scaler.transform(X_meta)

            return X_meta, imputer, scaler

        # Process Train
        X_train_meta, imputer, scaler = process_metadata(
            df_train, X_train_semantic, hist_train_emb, is_train=True
        )
        # Process Val
        X_val_meta, _, _ = process_metadata(
            df_val,
            X_val_semantic,
            hist_val_emb,
            is_train=False,
            imputer=imputer,
            scaler=scaler,
        )
        # Process Test
        X_test_meta, _, _ = process_metadata(
            df_test,
            X_test_semantic,
            hist_test_emb,
            is_train=False,
            imputer=imputer,
            scaler=scaler,
        )

        # 7. Construct Result
        data_dict = {
            "train": {
                "lexical": X_train_lexical,
                "behavioral": X_train_behavioral,
                "semantic": X_train_semantic,
                "metadata": X_train_meta,
                "y": y_train,
            },
            "val": {
                "lexical": X_val_lexical,
                "behavioral": X_val_behavioral,
                "semantic": X_val_semantic,
                "metadata": X_val_meta,
                "y": y_val,
            },
            "test": {
                "lexical": X_test_lexical,
                "behavioral": X_test_behavioral,
                "semantic": X_test_semantic,
                "metadata": X_test_meta,
            },
        }

        # 8. Save to Cache
        self._save_cache(data_dict)

        self.logger.info("Feature engineering complete.")
        return data_dict
