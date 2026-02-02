import os
import numpy as np
import pandas as pd
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config, set_seed
from library.data_loader import load_raw_data


class MetadataExtractor:
    """
    Handles numerical feature engineering: imputation, ratio generation,
    Arcsinh transformation, and scaling.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.num_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

    def fit(self, df):
        # Fit imputer on raw numerical columns
        self.imputer.fit(df[self.num_cols])

        # Fit scaler on imputed + arcsinh transformed data
        imputed = self.imputer.transform(df[self.num_cols])
        self.scaler.fit(np.arcsinh(imputed))
        return self

    def transform(self, df):
        # Impute
        imputed = self.imputer.transform(df[self.num_cols])

        # Calculate Ratios (Upvotes Minus / Upvotes Plus)
        # Index 4 is minus, Index 5 is plus based on num_cols list
        plus = imputed[:, 5]
        minus = imputed[:, 4]
        ratio = np.divide(
            minus, plus, out=np.zeros_like(minus), where=plus != 0
        ).reshape(-1, 1)

        # Arcsinh + Scale for MLP
        dense_scaled = self.scaler.transform(np.arcsinh(imputed))

        return dense_scaled, ratio, imputed


class TextEmbedder:
    """
    Generates SBERT embeddings for request text and user history.
    """

    def __init__(self):
        self.model = SentenceTransformer(Config.SBERT_MODEL)

    def encode(self, texts):
        return self.model.encode(
            texts.fillna("").astype(str).tolist(),
            show_progress_bar=False,
            batch_size=64,
        )

    def process_history(self, df):
        """
        Parses subreddit lists, generates embedding sequences, masks, and centroids.
        """
        hist_embs_list = []
        masks_list = []
        centroids = []
        max_len = 20

        for sub_str in df["requester_subreddits_at_request"]:
            try:
                subs = ast.literal_eval(sub_str) if isinstance(sub_str, str) else []
            except:
                subs = []

            if not isinstance(subs, list):
                subs = []

            current_subs = subs[:max_len]

            if not current_subs:
                emb_matrix = np.zeros((max_len, 384))
                mask = np.zeros(max_len)
                centroid = np.zeros(384)
            else:
                embs = self.model.encode(current_subs, show_progress_bar=False)
                n = len(embs)
                emb_matrix = np.zeros((max_len, 384))
                emb_matrix[:n, :] = embs
                mask = np.zeros(max_len)
                mask[:n] = 1
                centroid = np.mean(embs, axis=0)

            hist_embs_list.append(emb_matrix)
            masks_list.append(mask)
            centroids.append(centroid)

        return np.array(hist_embs_list), np.array(masks_list), np.array(centroids)


class TfidfProcessor:
    """
    Generates TF-IDF features for the Random Forest model.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE_TFIDF, stop_words="english"
        )

    def _prepare_text(self, df):
        return (df["request_title"] + " " + df["request_text_edit_aware"]).fillna("")

    def fit_transform(self, df):
        text = self._prepare_text(df)
        return self.vectorizer.fit_transform(text).toarray()

    def transform(self, df):
        text = self._prepare_text(df)
        return self.vectorizer.transform(text).toarray()


class InteractionProcessor:
    """
    Handles Top-K community indicators, Consistency Scalars, and Interaction Terms.
    """

    def __init__(self):
        self.top_k_subs = []

    def fit(self, df):
        all_subs = []
        for s_str in df["requester_subreddits_at_request"]:
            try:
                subs = ast.literal_eval(s_str)
                if isinstance(subs, list):
                    all_subs.extend(subs)
            except:
                pass

        # Identify top K subreddits
        self.top_k_subs = [
            x[0] for x in Counter(all_subs).most_common(Config.TOP_K_SUBREDDITS)
        ]
        return self

    def get_top_k_features(self, df):
        feats = np.zeros((len(df), len(self.top_k_subs)))
        for i, s_str in enumerate(df["requester_subreddits_at_request"]):
            try:
                subs = set(ast.literal_eval(s_str))
                for j, sub in enumerate(self.top_k_subs):
                    if sub in subs:
                        feats[i, j] = 1
            except:
                pass
        return feats

    def compute_consistency(self, title_emb, body_emb, centroid):
        # Cosine similarity with safety epsilon
        norm_t = np.linalg.norm(title_emb, axis=1, keepdims=True) + 1e-9
        norm_b = np.linalg.norm(body_emb, axis=1, keepdims=True) + 1e-9
        norm_c = np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9

        sim_t = np.sum(title_emb * centroid, axis=1, keepdims=True) / (norm_t * norm_c)
        sim_b = np.sum(body_emb * centroid, axis=1, keepdims=True) / (norm_b * norm_c)
        return sim_t, sim_b

    def get_interactions(self, cons_t, cons_b, num_arr, ratio_arr):
        # I1 = Title_Consistency * log(1 + Account_Age)
        # Account age is index 0 in num_arr
        age = num_arr[:, 0].reshape(-1, 1)
        i1 = cons_t * np.log1p(age)

        # I2 = Body_Consistency * Upvote_Ratio
        i2 = cons_b * ratio_arr

        return np.hstack([i1, i2])


def process_features(load_cached_data=True):
    """
    Orchestrates the feature generation pipeline.
    Checks cache, processes data using the classes above if needed, and returns formatted datasets.
    """
    set_seed()
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_files = {
        "rf_train": os.path.join(Config.CACHE_DIR, "rf_train.npz"),
        "rf_val": os.path.join(Config.CACHE_DIR, "rf_val.npz"),
        "rf_test": os.path.join(Config.CACHE_DIR, "rf_test.npz"),
        "mlp_train": os.path.join(Config.CACHE_DIR, "mlp_train.npz"),
        "mlp_val": os.path.join(Config.CACHE_DIR, "mlp_val.npz"),
        "mlp_test": os.path.join(Config.CACHE_DIR, "mlp_test.npz"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading processed data from cache...")
        data = {}
        for k, v in cache_files.items():
            if k.startswith("rf") or k.startswith("mlp"):
                data[k] = np.load(v, allow_pickle=True)
            else:
                data[k] = np.load(v, allow_pickle=True)
        return data

    print("Processing data from scratch...")

    # Load Raw Data
    df_train = load_raw_data(Config.TRAIN_PATH)
    df_val = load_raw_data(Config.VAL_PATH)
    df_test = load_raw_data(Config.TEST_PATH)

    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    ids_test = df_test["request_id"].values

    # Initialize Processors
    meta_extractor = MetadataExtractor()
    text_embedder = TextEmbedder()
    tfidf_processor = TfidfProcessor()
    interaction_processor = InteractionProcessor()

    # 1. Metadata Processing
    print("Processing metadata...")
    meta_extractor.fit(df_train)
    train_dense_scaled, train_ratio, train_num_raw = meta_extractor.transform(df_train)
    val_dense_scaled, val_ratio, val_num_raw = meta_extractor.transform(df_val)
    test_dense_scaled, test_ratio, test_num_raw = meta_extractor.transform(df_test)

    # 2. Text Embedding (SBERT)
    print("Generating SBERT embeddings...")
    train_title_emb = text_embedder.encode(df_train["request_title"])
    train_body_emb = text_embedder.encode(df_train["request_text_edit_aware"])
    val_title_emb = text_embedder.encode(df_val["request_title"])
    val_body_emb = text_embedder.encode(df_val["request_text_edit_aware"])
    test_title_emb = text_embedder.encode(df_test["request_title"])
    test_body_emb = text_embedder.encode(df_test["request_text_edit_aware"])

    # 3. User History & Persona
    print("Processing user history...")
    train_hist, train_mask, train_cent = text_embedder.process_history(df_train)
    val_hist, val_mask, val_cent = text_embedder.process_history(df_val)
    test_hist, test_mask, test_cent = text_embedder.process_history(df_test)

    # 4. Interactions & Consistency
    print("Computing interactions...")
    interaction_processor.fit(df_train)

    # Top-K
    train_topk = interaction_processor.get_top_k_features(df_train)
    val_topk = interaction_processor.get_top_k_features(df_val)
    test_topk = interaction_processor.get_top_k_features(df_test)

    # Consistency
    train_cons_t, train_cons_b = interaction_processor.compute_consistency(
        train_title_emb, train_body_emb, train_cent
    )
    val_cons_t, val_cons_b = interaction_processor.compute_consistency(
        val_title_emb, val_body_emb, val_cent
    )
    test_cons_t, test_cons_b = interaction_processor.compute_consistency(
        test_title_emb, test_body_emb, test_cent
    )

    # Interaction Terms
    train_inter = interaction_processor.get_interactions(
        train_cons_t, train_cons_b, train_num_raw, train_ratio
    )
    val_inter = interaction_processor.get_interactions(
        val_cons_t, val_cons_b, val_num_raw, val_ratio
    )
    test_inter = interaction_processor.get_interactions(
        test_cons_t, test_cons_b, test_num_raw, test_ratio
    )

    # 5. TF-IDF
    print("Generating TF-IDF features...")
    train_tfidf = tfidf_processor.fit_transform(df_train)
    val_tfidf = tfidf_processor.transform(df_val)
    test_tfidf = tfidf_processor.transform(df_test)

    # --- Assemble Final Datasets ---

    # MLP Data Construction
    # meta_skip = [Dense_Scaled, Ratio, TopK]
    mlp_train = {
        "title": train_title_emb,
        "body": train_body_emb,
        "hist": train_hist,
        "mask": train_mask,
        "cent": train_cent,
        "meta_dense": train_dense_scaled,
        "meta_skip": np.hstack([train_dense_scaled, train_ratio, train_topk]),
    }
    mlp_val = {
        "title": val_title_emb,
        "body": val_body_emb,
        "hist": val_hist,
        "mask": val_mask,
        "cent": val_cent,
        "meta_dense": val_dense_scaled,
        "meta_skip": np.hstack([val_dense_scaled, val_ratio, val_topk]),
    }
    mlp_test = {
        "title": test_title_emb,
        "body": test_body_emb,
        "hist": test_hist,
        "mask": test_mask,
        "cent": test_cent,
        "meta_dense": test_dense_scaled,
        "meta_skip": np.hstack([test_dense_scaled, test_ratio, test_topk]),
    }

    # RF Data Construction
    # [TFIDF, Num_Raw, Ratio, TopK, Consistency, Interactions]
    rf_train_X = np.hstack(
        [
            train_tfidf,
            train_num_raw,
            train_ratio,
            train_topk,
            train_cons_t,
            train_cons_b,
            train_inter,
        ]
    )
    rf_val_X = np.hstack(
        [val_tfidf, val_num_raw, val_ratio, val_topk, val_cons_t, val_cons_b, val_inter]
    )
    rf_test_X = np.hstack(
        [
            test_tfidf,
            test_num_raw,
            test_ratio,
            test_topk,
            test_cons_t,
            test_cons_b,
            test_inter,
        ]
    )

    # Save to Cache
    np.savez(cache_files["rf_train"], X=rf_train_X)
    np.savez(cache_files["rf_val"], X=rf_val_X)
    np.savez(cache_files["rf_test"], X=rf_test_X)
    np.savez(cache_files["mlp_train"], **mlp_train)
    np.savez(cache_files["mlp_val"], **mlp_val)
    np.savez(cache_files["mlp_test"], **mlp_test)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["ids_test"], ids_test)

    print("Data processing complete and cached.")

    return {
        "rf_train": np.load(cache_files["rf_train"], allow_pickle=True),
        "rf_val": np.load(cache_files["rf_val"], allow_pickle=True),
        "rf_test": np.load(cache_files["rf_test"], allow_pickle=True),
        "mlp_train": np.load(cache_files["mlp_train"], allow_pickle=True),
        "mlp_val": np.load(cache_files["mlp_val"], allow_pickle=True),
        "mlp_test": np.load(cache_files["mlp_test"], allow_pickle=True),
        "y_train": np.load(cache_files["y_train"], allow_pickle=True),
        "y_val": np.load(cache_files["y_val"], allow_pickle=True),
        "ids_test": np.load(cache_files["ids_test"], allow_pickle=True),
    }
