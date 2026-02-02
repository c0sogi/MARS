import os
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_utils import load_dataset

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SentimentExtractor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def get_sentiment(self, text):
        if pd.isna(text) or text == "":
            return 0.0, 0.0

        # VADER scores: neg, neu, pos, compound
        scores = self.sia.polarity_scores(str(text))

        # Polarity: compound score (-1 to 1)
        polarity = scores["compound"]

        # Subjectivity Proxy: 1.0 - neutral score.
        # If text is purely neutral, subjectivity is 0.
        # If text has high pos or neg, subjectivity is high.
        subjectivity = 1.0 - scores["neu"]

        return polarity, subjectivity

    def transform(self, df):
        # Initialize columns
        for col in Config.SENTIMENT_COLS:
            df[col] = 0.0

        # Process Title
        print("    Extracting sentiment for Titles...")
        title_sent = df["request_title"].apply(self.get_sentiment)
        df["title_polarity"] = title_sent.apply(lambda x: x[0])
        df["title_subjectivity"] = title_sent.apply(lambda x: x[1])

        # Process Body
        print("    Extracting sentiment for Bodies...")
        body_sent = df["request_text_edit_aware"].apply(self.get_sentiment)
        df["body_polarity"] = body_sent.apply(lambda x: x[0])
        df["body_subjectivity"] = body_sent.apply(lambda x: x[1])

        return df[Config.SENTIMENT_COLS]


class MetadataEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols = Config.NUMERIC_COLS
        self.sentiment_cols = Config.SENTIMENT_COLS

    def engineer_ratios(self, df):
        # Avoid division by zero
        epsilon = 1e-6

        # Upvote Ratio
        df["upvote_ratio"] = df["requester_upvotes_plus_downvotes_at_request"] / (
            df["requester_upvotes_plus_downvotes_at_request"] + epsilon
        )

        # Comment/Post Ratio
        df["comment_post_ratio"] = df["requester_number_of_comments_at_request"] / (
            df["requester_number_of_posts_at_request"] + epsilon
        )

        # RAOP Activity Ratio
        df["raop_activity_ratio"] = (
            df["requester_number_of_comments_in_raop_at_request"]
            + df["requester_number_of_posts_on_raop_at_request"]
        ) / (
            df["requester_number_of_comments_at_request"]
            + df["requester_number_of_posts_at_request"]
            + epsilon
        )

        return df

    def process(self, train_df, val_df, test_df):
        print("  Processing Metadata...")

        # 1. Add Sentiment Features
        extractor = SentimentExtractor()
        for df in [train_df, val_df, test_df]:
            sent_df = extractor.transform(df)
            # Columns are already assigned in transform via reference,
            # but let's ensure they are part of the dataframe if transform returned a copy
            for col in self.sentiment_cols:
                if col not in df.columns:
                    df[col] = sent_df[col]

        # 2. Add Ratio Features
        train_df = self.engineer_ratios(train_df)
        val_df = self.engineer_ratios(val_df)
        test_df = self.engineer_ratios(test_df)

        # Define all meta columns
        ratio_cols = ["upvote_ratio", "comment_post_ratio", "raop_activity_ratio"]
        all_meta_cols = self.numeric_cols + self.sentiment_cols + ratio_cols

        # 3. Prepare RF Metadata (Raw + Imputed)
        # RF handles raw magnitudes well, but needs no NaNs
        print("    Preparing RF Metadata...")
        rf_train = train_df[all_meta_cols].copy()
        rf_val = val_df[all_meta_cols].copy()
        rf_test = test_df[all_meta_cols].copy()

        self.imputer.fit(rf_train)
        rf_train_arr = self.imputer.transform(rf_train)
        rf_val_arr = self.imputer.transform(rf_val)
        rf_test_arr = self.imputer.transform(rf_test)

        # 4. Prepare MLP Metadata (Arcsinh + Scaled)
        print("    Preparing MLP Metadata...")
        # Fill NaNs with 0 before arcsinh (assuming NaN implies 0 counts/missing)
        # For ratios, median is safer, but for counts, 0 is often correct.
        # We will use the imputed values from step 3 for consistency.

        # Apply Arcsinh to handle skewed distributions (counts)
        mlp_train_arr = np.arcsinh(rf_train_arr)
        mlp_val_arr = np.arcsinh(rf_val_arr)
        mlp_test_arr = np.arcsinh(rf_test_arr)

        # Scale
        self.scaler.fit(mlp_train_arr)
        mlp_train_arr = self.scaler.transform(mlp_train_arr)
        mlp_val_arr = self.scaler.transform(mlp_val_arr)
        mlp_test_arr = self.scaler.transform(mlp_test_arr)

        return (rf_train_arr, rf_val_arr, rf_test_arr), (
            mlp_train_arr,
            mlp_val_arr,
            mlp_test_arr,
        )


class TfidfPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

    def process(self, train_df, val_df, test_df):
        print("  Processing TF-IDF...")

        def combine_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).tolist()

        train_text = combine_text(train_df)
        val_text = combine_text(val_df)
        test_text = combine_text(test_df)

        # Fit on Train
        print("    Fitting TfidfVectorizer...")
        self.vectorizer.fit(train_text)

        # Transform
        train_tfidf = self.vectorizer.transform(train_text)
        val_tfidf = self.vectorizer.transform(val_text)
        test_tfidf = self.vectorizer.transform(test_text)

        return train_tfidf, val_tfidf, test_tfidf


class SBERTPipeline:
    def __init__(self):
        self.model_name = Config.MLP_PARAMS["sbert_model"]
        self.max_history = Config.MLP_PARAMS["max_history_len"]

    def process(self, train_df, val_df, test_df):
        print(f"  Processing SBERT Embeddings ({self.model_name})...")
        model = SentenceTransformer(self.model_name)

        # Helper to encode list of texts
        def encode_texts(texts, desc):
            print(f"    Encoding {desc}...")
            # Fill NaNs
            texts = [str(t) if not pd.isna(t) else "" for t in texts]
            return model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )

        # 1. Encode Titles and Bodies
        data = {}
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            data[f"{split_name}_title"] = encode_texts(
                df["request_title"], f"{split_name} titles"
            )
            data[f"{split_name}_body"] = encode_texts(
                df["request_text_edit_aware"], f"{split_name} bodies"
            )

        # 2. Encode History
        # Strategy: Identify all unique subreddits, encode them once, then map.
        print("    Encoding History Sequences...")

        # Collect all unique subreddits
        all_subreddits = set()
        for df in [train_df, val_df, test_df]:
            for hist_list in df[Config.HISTORY_COL]:
                # hist_list is already a list due to load_dataset
                if isinstance(hist_list, list):
                    all_subreddits.update(hist_list)

        sorted_subs = sorted(list(all_subreddits))
        sub_to_idx = {sub: i for i, sub in enumerate(sorted_subs)}

        print(f"    Unique subreddits found: {len(sorted_subs)}")
        sub_embeddings = model.encode(
            sorted_subs, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

        # Create Zero embedding for padding
        embedding_dim = sub_embeddings.shape[1]
        # We can just use 0-padding in the final tensor.

        def process_history(df):
            num_samples = len(df)
            # Shape: (N, MaxLen, D)
            history_tensor = np.zeros(
                (num_samples, self.max_history, embedding_dim), dtype=np.float32
            )

            for i, row_hist in enumerate(df[Config.HISTORY_COL]):
                if not isinstance(row_hist, list) or not row_hist:
                    continue

                # Truncate or take last k? Usually recent history is more relevant.
                # Let's take the most recent ones (assuming list is chronological or random).
                # The dataset description doesn't specify order, but usually lists are chronological.
                # We'll take the first K (assuming they are relevant) or last K.
                # Let's take the first K provided in the list.
                current_subs = row_hist[: self.max_history]

                for j, sub in enumerate(current_subs):
                    if sub in sub_to_idx:
                        idx = sub_to_idx[sub]
                        history_tensor[i, j, :] = sub_embeddings[idx]

            return history_tensor

        data["train_history"] = process_history(train_df)
        data["val_history"] = process_history(val_df)
        data["test_history"] = process_history(test_df)

        return data


def get_features(load_cached_data=True):
    """
    Main entry point for feature engineering.
    Orchestrates loading, processing, and caching.
    """
    # Cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    rf_cache_path = os.path.join(cache_dir, "rf_features.npz")
    mlp_cache_path = os.path.join(cache_dir, "mlp_features.npz")
    labels_cache_path = os.path.join(cache_dir, "labels.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(rf_cache_path)
        and os.path.exists(mlp_cache_path)
    ):
        print("Loading features from cache...")
        try:
            rf_data = np.load(rf_cache_path, allow_pickle=True)
            mlp_data = np.load(mlp_cache_path, allow_pickle=True)
            labels = np.load(labels_cache_path, allow_pickle=True).item()

            # Reconstruct dictionaries/tuples
            rf_out = {k: rf_data[k] for k in rf_data.files}
            mlp_out = {k: mlp_data[k] for k in mlp_data.files}

            return rf_out, mlp_out, labels
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    # Load Raw Data
    print("Loading raw datasets...")
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)

    # Extract Labels
    y_train = train_df[Config.TARGET_COL].astype(int).values
    y_val = val_df[Config.TARGET_COL].astype(int).values
    labels = {"y_train": y_train, "y_val": y_val}

    # --- Pipeline Execution ---

    # 1. Metadata & Sentiment
    meta_engineer = MetadataEngineer()
    (rf_meta_train, rf_meta_val, rf_meta_test), (
        mlp_meta_train,
        mlp_meta_val,
        mlp_meta_test,
    ) = meta_engineer.process(train_df, val_df, test_df)

    # 2. TF-IDF (Stream A)
    tfidf_pipeline = TfidfPipeline()
    tfidf_train, tfidf_val, tfidf_test = tfidf_pipeline.process(
        train_df, val_df, test_df
    )

    # Combine for RF (Sparse TFIDF + Dense Metadata)
    # Note: RF in sklearn handles sparse input efficiently.
    # We concatenate sparse TFIDF and dense Metadata.
    from scipy.sparse import hstack

    print("  Combining RF Features...")
    rf_train_final = hstack([tfidf_train, rf_meta_train]).tocsr()
    rf_val_final = hstack([tfidf_val, rf_meta_val]).tocsr()
    rf_test_final = hstack([tfidf_test, rf_meta_test]).tocsr()

    # 3. SBERT (Stream B)
    sbert_pipeline = SBERTPipeline()
    sbert_data = sbert_pipeline.process(train_df, val_df, test_df)

    # --- Saving to Cache ---
    print("Saving features to cache...")

    # Save RF data (Sparse matrices need careful handling with npz, usually better to save components or pickles)
    # Ideally we save components and reconstruct. But for simplicity in this script structure:
    # We will save the scipy sparse matrices using a helper or just save components.
    # To keep it simple and robust without pickle reliance for the main object structure:
    # We save indices/data/indptr for sparse, or just save them as object arrays if small enough.
    # Actually, scipy.sparse.save_npz is best, but we want one file.
    # Let's save components: TFIDF and Metadata separately in the RF dict.

    # Actually, let's just save the final combined sparse matrices using scipy.sparse.save_npz in separate files
    # to avoid complexity, OR just return them and let the training script handle it.
    # But the requirement says "Save the result to the cache directory".

    # We will save the components for RF to allow reconstruction.
    # However, saving sparse matrices inside .npz is tricky.
    # Strategy: Save dense metadata and sparse TFIDF separately.

    import scipy.sparse

    scipy.sparse.save_npz(os.path.join(cache_dir, "rf_train_tfidf.npz"), tfidf_train)
    scipy.sparse.save_npz(os.path.join(cache_dir, "rf_val_tfidf.npz"), tfidf_val)
    scipy.sparse.save_npz(os.path.join(cache_dir, "rf_test_tfidf.npz"), tfidf_test)

    np.savez(
        rf_cache_path,
        train_meta=rf_meta_train,
        val_meta=rf_meta_val,
        test_meta=rf_meta_test,
    )

    np.savez(
        mlp_cache_path,
        train_title=sbert_data["train_title"],
        train_body=sbert_data["train_body"],
        train_history=sbert_data["train_history"],
        train_meta=mlp_meta_train,
        val_title=sbert_data["val_title"],
        val_body=sbert_data["val_body"],
        val_history=sbert_data["val_history"],
        val_meta=mlp_meta_val,
        test_title=sbert_data["test_title"],
        test_body=sbert_data["test_body"],
        test_history=sbert_data["test_history"],
        test_meta=mlp_meta_test,
    )

    np.save(labels_cache_path, labels)

    # Construct Return Objects
    rf_out = {
        "train_tfidf": tfidf_train,
        "train_meta": rf_meta_train,
        "val_tfidf": tfidf_val,
        "val_meta": rf_meta_val,
        "test_tfidf": tfidf_test,
        "test_meta": rf_meta_test,
    }

    mlp_out = {
        "train_title": sbert_data["train_title"],
        "train_body": sbert_data["train_body"],
        "train_history": sbert_data["train_history"],
        "train_meta": mlp_meta_train,
        "val_title": sbert_data["val_title"],
        "val_body": sbert_data["val_body"],
        "val_history": sbert_data["val_history"],
        "val_meta": mlp_meta_val,
        "test_title": sbert_data["test_title"],
        "test_body": sbert_data["test_body"],
        "test_history": sbert_data["test_history"],
        "test_meta": mlp_meta_test,
    }

    return rf_out, mlp_out, labels
