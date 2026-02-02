import os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import load_csv_data, clean_text, parse_list_column, set_seed

# Ensure reproducibility
set_seed(Config.RANDOM_SEED)


class SBERTEncoder:
    """
    Wrapper for SentenceTransformer to handle batch encoding.
    """

    def __init__(self, model_name=Config.SBERT_MODEL_NAME, device=Config.DEVICE):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        # Handle empty lists or non-strings
        cleaned_texts = [str(t) if pd.notna(t) else "" for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings


class CommunityProfiler:
    """
    Fits a GMM on subreddit embeddings to generate soft-cluster profiles for users.
    """

    def __init__(
        self,
        n_components=Config.GMM_N_COMPONENTS,
        covariance_type=Config.GMM_COVARIANCE_TYPE,
        random_state=Config.RANDOM_SEED,
    ):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        self.subreddit_to_embedding = {}
        self.fitted = False

    def fit(self, unique_subreddits, embeddings):
        """
        Fits the GMM on the embeddings of unique subreddits.
        """
        self.gmm.fit(embeddings)
        self.subreddit_to_embedding = dict(zip(unique_subreddits, embeddings))
        self.fitted = True

    def get_user_profiles(self, user_subreddits_series):
        """
        Generates a probability distribution (profile) for each user based on their subreddit history.
        Returns: np.array of shape (n_samples, n_components)
        """
        if not self.fitted:
            raise ValueError("CommunityProfiler must be fitted before transformation.")

        n_samples = len(user_subreddits_series)
        n_components = self.gmm.n_components
        profiles = np.zeros((n_samples, n_components))

        for idx, sub_list in enumerate(user_subreddits_series):
            if not sub_list or len(sub_list) == 0:
                # Uniform distribution or zero vector?
                # Zero vector implies no history, which is distinct from "active in all clusters".
                # We leave it as zeros.
                continue

            # Get embeddings for user's subreddits
            user_embs = []
            for sub in sub_list:
                if sub in self.subreddit_to_embedding:
                    user_embs.append(self.subreddit_to_embedding[sub])

            if len(user_embs) > 0:
                user_embs = np.array(user_embs)
                # Predict posterior probabilities for each subreddit
                # shape: (n_subreddits, n_components)
                probs = self.gmm.predict_proba(user_embs)
                # Average to get user profile
                profiles[idx] = np.mean(probs, axis=0)

        return profiles


class ConsistencyScorer:
    """
    Calculates semantic consistency between the request and the user's history.
    """

    def __init__(self):
        pass

    def compute_scores(self, request_embeddings, user_subreddits_series, subreddit_map):
        """
        Computes Cosine Similarity between Request Embedding and History Centroid.
        """
        scores = np.zeros(len(request_embeddings))

        for idx, (req_emb, sub_list) in enumerate(
            zip(request_embeddings, user_subreddits_series)
        ):
            if not sub_list or len(sub_list) == 0:
                scores[idx] = 0.0  # Neutral/No history
                continue

            hist_embs = []
            for sub in sub_list:
                if sub in subreddit_map:
                    hist_embs.append(subreddit_map[sub])

            if len(hist_embs) > 0:
                centroid = np.mean(hist_embs, axis=0)
                # Cosine similarity
                # Reshape for sklearn: (1, dim)
                sim = cosine_similarity(req_emb.reshape(1, -1), centroid.reshape(1, -1))
                scores[idx] = sim[0][0]
            else:
                scores[idx] = 0.0

        return scores.reshape(-1, 1)


class TFIDFProcessor:
    """
    Handles TF-IDF vectorization for the Random Forest stream.
    """

    def __init__(
        self,
        max_features=Config.TFIDF_MAX_FEATURES,
        ngram_range=Config.TFIDF_NGRAM_RANGE,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()


class MetadataProcessor:
    """
    Handles numerical feature engineering, imputation, and scaling.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.numeric_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_on_raop_at_request",
        ]
        self.fitted = False

    def engineer_features(self, df):
        """
        Creates derived features (ratios, text stats).
        """
        df_eng = df[self.numeric_cols].copy()

        # Text Meta-Features
        # Use 'request_text_edit_aware' if available, else 'request_text'
        txt_col = (
            "request_text_edit_aware"
            if "request_text_edit_aware" in df.columns
            else "request_text"
        )
        texts = df[txt_col].fillna("").astype(str)

        df_eng["text_len_char"] = texts.apply(len)
        df_eng["text_len_word"] = texts.apply(lambda x: len(x.split()))
        df_eng["caps_ratio"] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )

        # Ratios
        # Upvote Ratio: (up - down) is net, (up + down) is total.
        # We don't have raw up/down, only net and sum.
        # Let Net = U - D, Sum = U + D.
        # U = (Sum + Net) / 2. Ratio = U / Sum = (Sum + Net) / (2 * Sum)
        # Simplified: Net / Sum is a proxy for "controversy" or "approval".
        # We'll use Net / (Sum + epsilon)
        df_eng["upvote_ratio"] = df["requester_upvotes_minus_downvotes_at_request"] / (
            df["requester_upvotes_plus_downvotes_at_request"] + 1e-5
        )

        # Activity Ratio (RAOP vs Global)
        total_activity = (
            df["requester_number_of_comments_at_request"]
            + df["requester_number_of_posts_at_request"]
        )
        raop_activity = (
            df["requester_number_of_comments_in_raop_at_request"]
            + df["requester_number_of_posts_on_raop_at_request"]
        )
        df_eng["raop_activity_ratio"] = raop_activity / (total_activity + 1e-5)

        return df_eng

    def fit(self, df):
        features = self.engineer_features(df)
        self.imputer.fit(features)

        # Fit scaler on Arcsinh transformed data for MLP
        features_imputed = self.imputer.transform(features)
        features_arcsinh = np.arcsinh(features_imputed)
        self.scaler.fit(features_arcsinh)
        self.fitted = True

    def transform(self, df):
        if not self.fitted:
            raise ValueError("MetadataProcessor must be fitted first.")

        features = self.engineer_features(df)
        features_imputed = self.imputer.transform(features)

        # RF Features: Raw imputed (trees handle scale well)
        rf_features = features_imputed

        # MLP Features: Arcsinh + Scaled
        mlp_features = self.scaler.transform(np.arcsinh(features_imputed))

        return rf_features, mlp_features


def prepare_history_sequences(
    user_subreddits_series, subreddit_map, max_len=50, embedding_dim=Config.SBERT_DIM
):
    """
    Prepares padded sequences of history embeddings for the MLP.
    Returns: np.array of shape (n_samples, max_len, embedding_dim)
    """
    n_samples = len(user_subreddits_series)
    sequences = np.zeros((n_samples, max_len, embedding_dim), dtype=np.float32)

    for i, sub_list in enumerate(user_subreddits_series):
        if not sub_list:
            continue

        # Get embeddings
        embs = [subreddit_map[sub] for sub in sub_list if sub in subreddit_map]

        # Truncate if too long
        if len(embs) > max_len:
            embs = embs[:max_len]

        # Fill sequence (padding is already zeros)
        for t, emb in enumerate(embs):
            sequences[i, t, :] = emb

    return sequences


def prepare_data(load_cached_data=True):
    """
    Main orchestration function.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["train_data"].item(),
            data["val_data"].item(),
            data["test_data"].item(),
        )

    print("Starting feature engineering pipeline...")

    # 1. Load Data
    df_train = load_csv_data("train", debug=Config.DEBUG)
    df_val = load_csv_data("val", debug=Config.DEBUG)
    df_test = load_csv_data("test", debug=Config.DEBUG)

    # 2. Pre-processing
    # Parse lists
    for df in [df_train, df_val, df_test]:
        df["requester_subreddits_at_request"] = parse_list_column(
            df["requester_subreddits_at_request"]
        )
        # Use edit aware text if possible
        df["text_content"] = (
            df["request_text_edit_aware"].fillna("").apply(clean_text)
            + " "
            + df["request_title"].fillna("").apply(clean_text)
        )

    # 3. SBERT Encoding (Request Texts)
    print("Encoding request texts...")
    sbert = SBERTEncoder()
    train_req_emb = sbert.encode(df_train["text_content"].tolist())
    val_req_emb = sbert.encode(df_val["text_content"].tolist())
    test_req_emb = sbert.encode(df_test["text_content"].tolist())

    # 4. Subreddit & History Processing
    print("Processing subreddit history...")
    # Get all unique subreddits from train set to build the universe
    train_subreddits = set()
    for sub_list in df_train["requester_subreddits_at_request"]:
        train_subreddits.update(sub_list)
    unique_subreddits = list(train_subreddits)

    # Encode unique subreddits
    print(f"Encoding {len(unique_subreddits)} unique subreddits...")
    sub_embeddings = sbert.encode(
        unique_subreddits, batch_size=64, show_progress_bar=False
    )
    subreddit_map = dict(zip(unique_subreddits, sub_embeddings))

    # Fit GMM Profiler
    print("Fitting GMM Community Profiler...")
    profiler = CommunityProfiler()
    profiler.fit(unique_subreddits, sub_embeddings)

    # Generate Profiles
    train_profiles = profiler.get_user_profiles(
        df_train["requester_subreddits_at_request"]
    )
    val_profiles = profiler.get_user_profiles(df_val["requester_subreddits_at_request"])
    test_profiles = profiler.get_user_profiles(
        df_test["requester_subreddits_at_request"]
    )

    # Consistency Scores
    print("Calculating Consistency Scores...")
    scorer = ConsistencyScorer()
    train_consistency = scorer.compute_scores(
        train_req_emb, df_train["requester_subreddits_at_request"], subreddit_map
    )
    val_consistency = scorer.compute_scores(
        val_req_emb, df_val["requester_subreddits_at_request"], subreddit_map
    )
    test_consistency = scorer.compute_scores(
        test_req_emb, df_test["requester_subreddits_at_request"], subreddit_map
    )

    # Prepare History Sequences for MLP
    print("Preparing history sequences for MLP...")
    train_hist_seq = prepare_history_sequences(
        df_train["requester_subreddits_at_request"], subreddit_map
    )
    val_hist_seq = prepare_history_sequences(
        df_val["requester_subreddits_at_request"], subreddit_map
    )
    test_hist_seq = prepare_history_sequences(
        df_test["requester_subreddits_at_request"], subreddit_map
    )

    # 5. TF-IDF (RF Stream)
    print("Vectorizing text with TF-IDF...")
    tfidf = TFIDFProcessor()
    train_tfidf = tfidf.fit_transform(df_train["text_content"])
    val_tfidf = tfidf.transform(df_val["text_content"])
    test_tfidf = tfidf.transform(df_test["text_content"])

    # 6. Metadata Engineering
    print("Processing metadata...")
    meta_proc = MetadataProcessor()
    meta_proc.fit(df_train)

    train_meta_rf, train_meta_mlp = meta_proc.transform(df_train)
    val_meta_rf, val_meta_mlp = meta_proc.transform(df_val)
    test_meta_rf, test_meta_mlp = meta_proc.transform(df_test)

    # 7. Assemble Datasets
    print("Assembling final datasets...")

    # Targets
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    # Test has no target

    # RF Features: Concatenate [TFIDF, Meta, Profiles, Consistency]
    X_train_rf = np.hstack(
        [train_tfidf, train_meta_rf, train_profiles, train_consistency]
    )
    X_val_rf = np.hstack([val_tfidf, val_meta_rf, val_profiles, val_consistency])
    X_test_rf = np.hstack([test_tfidf, test_meta_rf, test_profiles, test_consistency])

    # MLP Features: Dict of inputs
    train_data = {
        "rf_features": X_train_rf,
        "mlp_features": {
            "request_emb": train_req_emb,
            "history_seq": train_hist_seq,
            "metadata": train_meta_mlp,
        },
        "y": y_train,
        "ids": df_train["request_id"].values,
    }

    val_data = {
        "rf_features": X_val_rf,
        "mlp_features": {
            "request_emb": val_req_emb,
            "history_seq": val_hist_seq,
            "metadata": val_meta_mlp,
        },
        "y": y_val,
        "ids": df_val["request_id"].values,
    }

    test_data = {
        "rf_features": X_test_rf,
        "mlp_features": {
            "request_emb": test_req_emb,
            "history_seq": test_hist_seq,
            "metadata": test_meta_mlp,
        },
        "ids": df_test["request_id"].values,
    }

    # 8. Cache
    print(f"Caching data to {cache_file}...")
    np.savez(cache_file, train_data=train_data, val_data=val_data, test_data=test_data)

    return train_data, val_data, test_data
