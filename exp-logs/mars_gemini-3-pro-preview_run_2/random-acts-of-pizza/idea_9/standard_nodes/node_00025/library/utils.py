import os
import json
import logging
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# ==========================================
# Utility Functions
# ==========================================


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: PyTorch seeding is handled implicitly if imported,
    # but we primarily use sklearn/numpy here.


def setup_logger():
    """Configures a simple logger."""
    logger = logging.getLogger("solution")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ==========================================
# Data Processing & Feature Engineering
# ==========================================


def load_raw_data(metadata_dir="./metadata", input_dir="./input"):
    """Loads metadata and merges with raw JSON data."""

    # Load Metadata
    df_train_meta = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    df_val_meta = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    df_test_meta = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Load Raw JSON
    with open(os.path.join(input_dir, "train.json"), "r") as f:
        train_json = json.load(f)
    with open(os.path.join(input_dir, "test.json"), "r") as f:
        test_json = json.load(f)

    # Convert JSON to DF for easier merging
    # Note: We need to map by request_id.
    # The raw json is a list of dicts.
    df_raw_train = pd.DataFrame(train_json)
    df_raw_test = pd.DataFrame(test_json)

    # Merge
    # We use inner join on request_id to attach features to the split definitions
    # Note: df_raw_train contains both train and val samples

    df_train = df_train_meta.merge(df_raw_train, on="request_id", how="left")
    df_val = df_val_meta.merge(df_raw_train, on="request_id", how="left")
    df_test = df_test_meta.merge(df_raw_test, on="request_id", how="left")

    # Handle label column conflicts if any (though merge on request_id should be clean)
    if "requester_received_pizza_x" in df_train.columns:
        df_train["requester_received_pizza"] = df_train["requester_received_pizza_x"]
        df_val["requester_received_pizza"] = df_val["requester_received_pizza_x"]

    return df_train, df_val, df_test


def extract_text_embeddings(
    texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32
):
    """Generates L2-normalized embeddings."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    # L2 Normalize
    embeddings = normalize(embeddings, norm="l2")
    return embeddings


def get_knn_feature(X_train, y_train, X_query, k=50, cv=False):
    """
    Generates the 'Local Success Probability' feature.
    If cv=True, uses Stratified K-Fold to generate OOF predictions for X_train.
    If cv=False, fits on X_train and queries for X_query.
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")

    if cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # We want the probability of class 1
        y_prob = cross_val_predict(
            knn, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
        )
        return y_prob[:, 1].reshape(-1, 1)
    else:
        knn.fit(X_train, y_train)
        y_prob = knn.predict_proba(X_query)
        return y_prob[:, 1].reshape(-1, 1)


def process_metadata(df_train, df_val, df_test):
    """Extracts and scales metadata using RankGauss."""
    numeric_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # Fill NaNs with 0 (though data analysis showed none, safety first)
    X_train = df_train[numeric_cols].fillna(0).values
    X_val = df_val[numeric_cols].fillna(0).values
    X_test = df_test[numeric_cols].fillna(0).values

    # RankGauss (QuantileTransformer to Normal)
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_scaled = qt.fit_transform(X_train)
    X_val_scaled = qt.transform(X_val)
    X_test_scaled = qt.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def get_features(load_cached_data=True, debug_sample_size=None):
    """
    Main feature engineering pipeline with caching.
    Returns feature matrices and labels.
    """
    logger = logging.getLogger("solution")
    cache_dir = "./working/idea_9/"
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        logger.info("Loading features from cache...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_val = np.load(files["X_val"])
        y_val = np.load(files["y_val"])
        X_test = np.load(files["X_test"])
        test_ids = np.load(files["test_ids"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_ids

    logger.info("Computing features from scratch...")

    # Load Data
    df_train, df_val, df_test = load_raw_data()

    # Debug Subsampling
    if debug_sample_size:
        logger.info(f"Subsampling data to {debug_sample_size} for debugging.")
        df_train = df_train.iloc[:debug_sample_size]
        df_val = df_val.iloc[:debug_sample_size]
        df_test = df_test.iloc[:debug_sample_size]

    # Labels
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    test_ids = df_test["request_id"].values

    # Text Processing
    text_col = "request_text_edit_aware"  # Prefer edit aware
    # Fallback if column missing (unlikely based on dataset desc)
    if text_col not in df_train.columns:
        text_col = "request_text"

    train_texts = df_train[text_col].fillna("").astype(str).tolist()
    val_texts = df_val[text_col].fillna("").astype(str).tolist()
    test_texts = df_test[text_col].fillna("").astype(str).tolist()

    # 1. Embeddings
    logger.info("Generating Embeddings...")
    emb_train = extract_text_embeddings(train_texts)
    emb_val = extract_text_embeddings(val_texts)
    emb_test = extract_text_embeddings(test_texts)

    # 2. k-NN Feature (Local Success Prior)
    logger.info("Computing k-NN Features...")
    # Train: OOF
    knn_train = get_knn_feature(emb_train, y_train, None, k=50, cv=True)
    # Val: Fit on Train, Query Val
    knn_val = get_knn_feature(emb_train, y_train, emb_val, k=50, cv=False)
    # Test: Fit on Train, Query Test
    knn_test = get_knn_feature(emb_train, y_train, emb_test, k=50, cv=False)

    # 3. Metadata
    logger.info("Processing Metadata...")
    meta_train, meta_val, meta_test = process_metadata(df_train, df_val, df_test)

    # Combine Features
    X_train = np.hstack([emb_train, knn_train, meta_train])
    X_val = np.hstack([emb_val, knn_val, meta_val])
    X_test = np.hstack([emb_test, knn_test, meta_test])

    # Cache
    logger.info("Saving features to cache...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_test"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


# ==========================================
# Main Execution
# ==========================================


def main():
    set_seed(42)
    logger = setup_logger()

    # 1. Get Features
    # Set debug_sample_size=None for full run
    X_train, y_train, X_val, y_val, X_test, test_ids = get_features(
        load_cached_data=True
    )

    logger.info(f"Training Data Shape: {X_train.shape}")

    # 2. Model Training
    # Neighborhood-Augmented Linear Ensemble
    # Base: Logistic Regression with balanced weights and strong regularization
    base_lr = LogisticRegression(
        class_weight="balanced",
        C=0.1,
        solver="liblinear",
        random_state=42,
        max_iter=1000,
    )

    # Ensemble: Bagging to reduce variance
    model = BaggingClassifier(
        estimator=base_lr,
        n_estimators=20,
        max_samples=0.8,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # 3. Validation
    logger.info("Validating...")
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)

    # Print metric with full precision
    print(f"Validation AUC: {auc}")

    # 4. Submission
    logger.info("Generating submission...")
    y_test_pred = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": y_test_pred}
    )

    # Ensure output directory exists
    os.makedirs("submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")


# Run the main function
main()
