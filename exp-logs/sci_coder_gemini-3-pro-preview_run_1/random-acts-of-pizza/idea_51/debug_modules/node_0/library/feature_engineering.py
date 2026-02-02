import os
import ast
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from library.config import (
    WORKING_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    EXCLUDE_COLS,
    TEXT_COLS,
    TFIDF_VOCAB_SIZE,
    TFIDF_NGRAM_RANGE,
    TOP_K_SUBREDDITS,
    RANDOM_STATE,
)
from library.utils import (
    save_to_cache,
    load_from_cache,
    get_feature_intersection,
    arcsinh_transform,
    seed_everything,
)
from library.text_processing import process_text_data

# Cache filenames
CACHE_FILE_RF = "features_rf.npz"
CACHE_FILE_MLP = "features_mlp.npz"


def parse_subreddits(x):
    """Safely parses the stringified list of subreddits."""
    try:
        if pd.isna(x) or x == "":
            return []
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def clean_data(df):
    """
    Removes columns that are explicitly excluded to prevent leakage.
    """
    cols_to_drop = [c for c in EXCLUDE_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def generate_metadata_features(df):
    """
    Generates scalar metadata features including ratios, log-transforms,
    and text statistics.
    """
    df = df.copy()

    # 1. Text Statistics (using edit_aware text if available)
    # Fill NaNs with empty string for length calc
    txt = df["request_text_edit_aware"].fillna("").astype(str)
    title = df["request_title"].fillna("").astype(str)

    df["meta_text_len_char"] = txt.apply(len)
    df["meta_text_len_word"] = txt.apply(lambda x: len(x.split()))
    df["meta_title_len_char"] = title.apply(len)
    df["meta_title_len_word"] = title.apply(lambda x: len(x.split()))

    # Caps ratio (shouting indicator)
    def get_caps_ratio(s):
        if len(s) == 0:
            return 0.0
        return sum(1 for c in s if c.isupper()) / len(s)

    df["meta_text_caps_ratio"] = txt.apply(get_caps_ratio)
    df["meta_title_caps_ratio"] = title.apply(get_caps_ratio)

    # 2. Account & Activity Ratios
    # Handle division by zero with epsilon
    eps = 1e-6

    # Upvote Ratio: Up / (Up + Down)
    # Note: We use 'plus_downvotes' which is sum, and 'minus' which is diff.
    # Up + Down = Total
    # Up - Down = Diff
    # 2*Up = Total + Diff => Up = (Total + Diff) / 2
    total_votes = df["requester_upvotes_plus_downvotes_at_request"].fillna(0)
    diff_votes = df["requester_upvotes_minus_downvotes_at_request"].fillna(0)

    upvotes = (total_votes + diff_votes) / 2
    downvotes = total_votes - upvotes

    df["meta_upvote_ratio"] = upvotes / (total_votes + eps)

    # Activity Ratios
    # Comments / Posts (on RAOP vs Global)
    raop_posts = df["requester_number_of_posts_on_raop_at_request"].fillna(0)
    total_posts = df["requester_number_of_posts_at_request"].fillna(0)

    df["meta_raop_post_ratio"] = raop_posts / (total_posts + eps)

    # 3. Log/Arcsinh Transforms for Skewed Raw Metrics
    # We create specific transformed columns for the RF to use directly
    # (The MLP pipeline does a global transform later, but RF might benefit from explicit features)
    skewed_cols = [
        "requester_account_age_in_days_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_upvotes_minus_downvotes_at_request",
    ]

    for col in skewed_cols:
        if col in df.columns:
            df[f"meta_log_{col}"] = np.log1p(np.maximum(0, df[col].fillna(0)))

    return df


def get_top_k_subreddits(train_df, val_df, test_df, k=TOP_K_SUBREDDITS):
    """
    Generates binary indicator features for the top K most frequent subreddits
    found in the training set.
    """
    # 1. Count frequencies in Train
    all_subreddits = []

    # Parse lists
    train_subs = train_df["requester_subreddits_at_request"].apply(parse_subreddits)
    val_subs = val_df["requester_subreddits_at_request"].apply(parse_subreddits)
    test_subs = test_df["requester_subreddits_at_request"].apply(parse_subreddits)

    for sub_list in train_subs:
        all_subreddits.extend(sub_list)

    counts = pd.Series(all_subreddits).value_counts()
    top_k_subs = counts.head(k).index.tolist()

    # 2. Create Binary Features
    def create_indicators(subs_series):
        # Create a matrix of zeros
        matrix = np.zeros((len(subs_series), len(top_k_subs)), dtype=np.int32)

        # Map subreddits to column indices
        sub_to_idx = {sub: i for i, sub in enumerate(top_k_subs)}

        for row_idx, sub_list in enumerate(subs_series):
            for sub in sub_list:
                if sub in sub_to_idx:
                    matrix[row_idx, sub_to_idx[sub]] = 1
        return matrix

    train_k = create_indicators(train_subs)
    val_k = create_indicators(val_subs)
    test_k = create_indicators(test_subs)

    return train_k, val_k, test_k, top_k_subs


def create_interaction_features(metadata_df, sim_title, sim_body):
    """
    Creates explicit interaction features for the Random Forest:
    Consistency * Credibility.

    Args:
        metadata_df: DataFrame containing credibility metrics.
        sim_title: Numpy array of Topic Consistency scores.
        sim_body: Numpy array of Narrative Consistency scores.
    """
    # Ensure inputs are aligned 1D arrays
    sim_title = sim_title.flatten()
    sim_body = sim_body.flatten()

    # Credibility metrics to interact with
    # We use the log-transformed versions or raw versions if robust
    cred_cols = [
        "requester_account_age_in_days_at_request",
        "meta_upvote_ratio",
        "requester_number_of_posts_at_request",
    ]

    interactions = {}

    for col in cred_cols:
        if col not in metadata_df.columns:
            continue

        # Fill NaNs for interaction calculation
        feat_vec = metadata_df[col].fillna(0).values

        # Interaction 1: Topic Consistency * Feature
        interactions[f"inter_topic_x_{col}"] = sim_title * feat_vec

        # Interaction 2: Narrative Consistency * Feature
        interactions[f"inter_narrative_x_{col}"] = sim_body * feat_vec

    return pd.DataFrame(interactions)


def run_feature_engineering(load_cached_data=True):
    """
    Main driver function to generate features for RF and MLP.
    """
    seed_everything(RANDOM_STATE)

    # 1. Check Cache
    rf_cache = load_from_cache(CACHE_FILE_RF)
    mlp_cache = load_from_cache(CACHE_FILE_MLP)

    if load_cached_data and rf_cache is not None and mlp_cache is not None:
        print("Loaded tabular features from cache.")
        return dict(rf_cache), dict(mlp_cache)

    print("Generating tabular features from scratch...")

    # 2. Load Raw Metadata
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 3. Load Text Features (Needed for interactions)
    # We call process_text_data to ensure we have the consistency scalars
    text_data = process_text_data(train_df, val_df, test_df, load_cached_data=True)

    # 4. Generate Base Metadata Features
    print("Generating metadata features...")
    train_meta = generate_metadata_features(train_df)
    val_meta = generate_metadata_features(val_df)
    test_meta = generate_metadata_features(test_df)

    # 5. Top-K Subreddit Indicators
    print(f"Generating Top-{TOP_K_SUBREDDITS} subreddit indicators...")
    train_topk, val_topk, test_topk, topk_names = get_top_k_subreddits(
        train_df, val_df, test_df
    )

    # 6. Prepare Feature Sets

    # --- STREAM A: Random Forest Features ---
    print("Constructing Random Forest features...")

    # A. TF-IDF (Title + Body)
    print("Vectorizing text for RF...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_VOCAB_SIZE,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
    )

    # Combine title and body for TF-IDF
    def combine_text(df):
        return (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        )

    train_text = combine_text(train_df)
    val_text = combine_text(val_df)
    test_text = combine_text(test_df)

    train_tfidf = tfidf.fit_transform(train_text)
    val_tfidf = tfidf.transform(val_text)
    test_tfidf = tfidf.transform(test_text)

    # B. Interaction Features
    print("Creating interaction features...")
    train_inter = create_interaction_features(
        train_meta, text_data["train_sim_title"], text_data["train_sim_body"]
    )
    val_inter = create_interaction_features(
        val_meta, text_data["val_sim_title"], text_data["val_sim_body"]
    )
    test_inter = create_interaction_features(
        test_meta, text_data["test_sim_title"], text_data["test_sim_body"]
    )

    # C. Select Tabular Columns for RF
    # We use intersection to ensure columns exist in all sets, removing exclusions
    # We want the generated metadata + original numeric columns
    # First, clean the dfs
    train_clean = clean_data(train_meta)
    test_clean = clean_data(test_meta)  # used for intersection check

    # Identify numeric columns common to train and test
    numeric_cols = get_feature_intersection(
        train_clean.select_dtypes(include=np.number),
        test_clean.select_dtypes(include=np.number),
    )

    # Extract numeric matrices
    X_train_num = train_meta[numeric_cols].values
    X_val_num = val_meta[numeric_cols].values
    X_test_num = test_meta[numeric_cols].values

    # Impute NaNs for RF (Median)
    imputer = SimpleImputer(strategy="median")
    X_train_num = imputer.fit_transform(X_train_num)
    X_val_num = imputer.transform(X_val_num)
    X_test_num = imputer.transform(X_test_num)

    # D. Assemble RF Features (Sparse Stack)
    # Structure: [Numeric Metadata, Interactions, TopK, TF-IDF]
    # Convert dense to sparse for stacking
    def assemble_rf(num, inter, topk, tfidf_mat):
        return sparse.hstack(
            [
                sparse.csr_matrix(num),
                sparse.csr_matrix(inter.values),
                sparse.csr_matrix(topk),
                tfidf_mat,
            ],
            format="csr",
        )

    X_train_rf = assemble_rf(X_train_num, train_inter, train_topk, train_tfidf)
    X_val_rf = assemble_rf(X_val_num, val_inter, val_topk, val_tfidf)
    X_test_rf = assemble_rf(X_test_num, test_inter, test_topk, test_tfidf)

    # --- STREAM B: MLP Features ---
    print("Constructing MLP features...")

    # MLP uses:
    # 1. Numeric Metadata (Arcsinh + Scaled)
    # 2. Top-K Indicators (Binary, kept as is)
    # (Embeddings are handled in the Dataset class using text_data)

    # Apply Arcsinh to numeric columns
    X_train_mlp_num = arcsinh_transform(
        X_train_num
    )  # Re-using imputed data is fine/better
    X_val_mlp_num = arcsinh_transform(X_val_num)
    X_test_mlp_num = arcsinh_transform(X_test_num)

    # Standard Scale
    scaler = StandardScaler()
    X_train_mlp_num = scaler.fit_transform(X_train_mlp_num)
    X_val_mlp_num = scaler.transform(X_val_mlp_num)
    X_test_mlp_num = scaler.transform(X_test_mlp_num)

    # Concatenate [Scaled Numeric, TopK]
    # Keep dense for MLP usually, unless TopK is huge. 50 is small.
    X_train_mlp = np.hstack([X_train_mlp_num, train_topk])
    X_val_mlp = np.hstack([X_val_mlp_num, val_topk])
    X_test_mlp = np.hstack([X_test_mlp_num, test_topk])

    # 7. Targets
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values
    # Test has no target

    # 8. Save
    rf_data = {
        "X_train": X_train_rf,
        "y_train": y_train,
        "X_val": X_val_rf,
        "y_val": y_val,
        "X_test": X_test_rf,
    }

    mlp_data = {
        "X_train": X_train_mlp,
        "y_train": y_train,
        "X_val": X_val_mlp,
        "y_val": y_val,
        "X_test": X_test_mlp,
    }

    save_to_cache(rf_data, CACHE_FILE_RF)
    save_to_cache(mlp_data, CACHE_FILE_MLP)

    print("Feature engineering complete.")
    return rf_data, mlp_data
