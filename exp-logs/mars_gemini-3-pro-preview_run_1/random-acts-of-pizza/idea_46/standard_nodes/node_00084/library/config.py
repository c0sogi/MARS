import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Global Configuration & Constants
# ==========================================


class Config:
    # Paths
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    CACHE_DIR = "./working/idea_46_v2/"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Random Seed
    SEED = 42

    # Feature Engineering Params
    TOP_K_SUBREDDITS = 50
    SBERT_MODEL = "all-MiniLM-L6-v2"
    VOCAB_SIZE_TFIDF = 5000

    # MLP Hyperparameters
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT_EMB = 0.5
    MLP_DROPOUT_DENSE = 0.2
    MLP_LR = 1e-4
    MLP_WEIGHT_DECAY = 1e-4
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15
    MLP_BATCH_SIZE = 32

    # Random Forest Hyperparameters
    RF_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
    }


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. Model Architecture (Orthogonal Skip-Gated MLP)
# ==========================================


class OrthogonalSkipGatedMLP(nn.Module):
    """
    Implements the Orthogonal Skip-Gated Dual-Query MLP described in Idea 46.

    Features:
    - Dual-Query Attention (Title/Body querying History).
    - Global Persona Injection (Centroid).
    - Orthogonal Gating: Control signal comes ONLY from low-dim metadata.
    - Skip Connection: Carries metadata + community indicators.
    """

    def __init__(self, metadata_dim, skip_dim, embedding_dim=384, hidden_dim=256):
        super().__init__()

        # Branch 1 & 2: Title and Body Semantics
        self.title_proj = nn.Linear(embedding_dim, hidden_dim)
        self.body_proj = nn.Linear(embedding_dim, hidden_dim)

        # Branch 3: Dual-Query History Attention
        # Query = Title/Body (hidden_dim), Key/Value = History (embedding_dim)
        # We project history to hidden_dim first for attention
        self.history_proj = nn.Linear(embedding_dim, hidden_dim)
        self.attention_title = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.attention_body = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )

        # Branch 4: Global Persona Injection
        self.persona_proj = nn.Linear(embedding_dim, hidden_dim)

        # Branch 5: Metadata Control Gate (Orthogonal)
        # Input: Pure numerical metadata (metadata_dim)
        # Output: Gate scalar (or vector matching semantic size)
        self.gate_control = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Scalar gate for simplicity, or vector
            nn.Sigmoid(),
        )

        # Fusion Layer
        # Concatenated Semantic Vector: Title + Body + Ctx1 + Ctx2 + Persona
        self.semantic_fusion_dim = hidden_dim * 5
        self.semantic_fusion = nn.Linear(self.semantic_fusion_dim, hidden_dim)

        # Final Classification
        # Input: (Gated Semantic) + Skip (Metadata + Community)
        self.final_input_dim = hidden_dim + skip_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(Config.MLP_DROPOUT_EMB)

    def forward(
        self,
        title_emb,
        body_emb,
        history_emb,
        history_mask,
        persona_centroid,
        metadata_dense,
        metadata_skip,
    ):
        # 1. Process Semantics
        t_feat = F.relu(self.title_proj(self.dropout(title_emb)))
        b_feat = F.relu(self.body_proj(self.dropout(body_emb)))
        p_feat = F.relu(self.persona_proj(self.dropout(persona_centroid)))

        # 2. History Attention
        # history_emb: (B, Seq, 384) -> (B, Seq, H)
        h_feat = F.relu(self.history_proj(self.dropout(history_emb)))

        # Create key_padding_mask (True where padded)
        # history_mask is 1 for valid, 0 for pad. MHA expects True for pad.
        key_padding_mask = history_mask == 0

        # Query 1: Title
        # Unsqueeze title to (B, 1, H)
        q1 = t_feat.unsqueeze(1)
        attn_out1, _ = self.attention_title(
            q1, h_feat, h_feat, key_padding_mask=key_padding_mask
        )
        ctx1 = attn_out1.squeeze(1)

        # Query 2: Body
        q2 = b_feat.unsqueeze(1)
        attn_out2, _ = self.attention_body(
            q2, h_feat, h_feat, key_padding_mask=key_padding_mask
        )
        ctx2 = attn_out2.squeeze(1)

        # 3. Semantic Fusion
        semantic_raw = torch.cat([t_feat, b_feat, ctx1, ctx2, p_feat], dim=1)
        semantic_vec = F.relu(self.semantic_fusion(semantic_raw))

        # 4. Orthogonal Gating
        # Gate derived ONLY from metadata_dense
        gate = self.gate_control(metadata_dense)
        gated_semantic = semantic_vec * gate

        # 5. Skip Connection & Final Classification
        combined = torch.cat([gated_semantic, metadata_skip], dim=1)
        logits = self.classifier(combined)

        return logits


# ==========================================
# 3. Data Processing & Feature Engineering
# ==========================================


def load_and_process_data(load_cached_data=True):
    """
    Loads data, performs feature engineering (SBERT, TF-IDF, Interactions),
    and returns processed datasets for RF and MLP.
    Handles caching to disk.
    """
    set_seed()

    # Cache paths
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

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Target
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    ids_test = df_test["request_id"].values

    # --- Text Processing (SBERT) ---
    print("Generating SBERT embeddings...")
    sbert = SentenceTransformer(Config.SBERT_MODEL)

    def get_embeddings(texts):
        return sbert.encode(
            texts.fillna("").astype(str).tolist(),
            show_progress_bar=False,
            batch_size=64,
        )

    # Title & Body Embeddings
    train_title_emb = get_embeddings(df_train["request_title"])
    train_body_emb = get_embeddings(df_train["request_text_edit_aware"])
    val_title_emb = get_embeddings(df_val["request_title"])
    val_body_emb = get_embeddings(df_val["request_text_edit_aware"])
    test_title_emb = get_embeddings(df_test["request_title"])
    test_body_emb = get_embeddings(df_test["request_text_edit_aware"])

    # --- User History & Persona (Simulated/Extracted) ---
    # Since we don't have raw comment history in the metadata CSV (it's aggregated),
    # we use 'requester_subreddits_at_request' list to approximate history context.
    # We treat each subreddit name as a "history item".

    def process_history(df):
        # Parse string representation of list if necessary
        import ast

        hist_embs_list = []
        masks_list = []
        centroids = []

        # Max history length (subreddits)
        max_len = 20

        for sub_str in df["requester_subreddits_at_request"]:
            try:
                subs = ast.literal_eval(sub_str) if isinstance(sub_str, str) else []
            except:
                subs = []

            if not isinstance(subs, list):
                subs = []

            # Truncate or Pad
            current_subs = subs[:max_len]
            if not current_subs:
                # Empty history handling
                emb_matrix = np.zeros((max_len, 384))
                mask = np.zeros(max_len)
                mask[0] = (
                    1  # Ensure at least one valid token to prevent NaN in Attention
                )
                centroid = np.zeros(384)
            else:
                # Encode subreddits
                embs = sbert.encode(current_subs, show_progress_bar=False)
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

    print("Processing user history...")
    train_hist, train_mask, train_cent = process_history(df_train)
    val_hist, val_mask, val_cent = process_history(df_val)
    test_hist, test_mask, test_cent = process_history(df_test)

    # --- Global Consistency Scalars ---
    def compute_consistency(title_emb, body_emb, centroid):
        # Cosine sim between title/body and history centroid
        # Handle zero vectors
        norm_t = np.linalg.norm(title_emb, axis=1, keepdims=True) + 1e-9
        norm_b = np.linalg.norm(body_emb, axis=1, keepdims=True) + 1e-9
        norm_c = np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9

        sim_t = np.sum(title_emb * centroid, axis=1, keepdims=True) / (norm_t * norm_c)
        sim_b = np.sum(body_emb * centroid, axis=1, keepdims=True) / (norm_b * norm_c)
        return sim_t, sim_b

    train_cons_t, train_cons_b = compute_consistency(
        train_title_emb, train_body_emb, train_cent
    )
    val_cons_t, val_cons_b = compute_consistency(val_title_emb, val_body_emb, val_cent)
    test_cons_t, test_cons_b = compute_consistency(
        test_title_emb, test_body_emb, test_cent
    )

    # --- Numerical Metadata & Interaction Features ---
    num_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # Impute
    imputer = SimpleImputer(strategy="median")
    train_num = imputer.fit_transform(df_train[num_cols])
    val_num = imputer.transform(df_val[num_cols])
    test_num = imputer.transform(df_test[num_cols])

    # Ratios
    def get_ratios(arr):
        # upvotes_minus / upvotes_plus (approx upvote ratio)
        # Avoid div by zero
        plus = arr[:, 5]
        minus = arr[:, 4]
        ratio = np.divide(minus, plus, out=np.zeros_like(minus), where=plus != 0)
        return ratio.reshape(-1, 1)

    train_ratio = get_ratios(train_num)
    val_ratio = get_ratios(val_num)
    test_ratio = get_ratios(test_num)

    # Interaction Features
    # I1 = Title_Consistency * log(1 + Account_Age)
    # I2 = Body_Consistency * Upvote_Ratio
    def get_interactions(cons_t, cons_b, num_arr, ratio_arr):
        age = num_arr[:, 0].reshape(-1, 1)
        i1 = cons_t * np.log1p(age)
        i2 = cons_b * ratio_arr
        return np.hstack([i1, i2])

    train_inter = get_interactions(train_cons_t, train_cons_b, train_num, train_ratio)
    val_inter = get_interactions(val_cons_t, val_cons_b, val_num, val_ratio)
    test_inter = get_interactions(test_cons_t, test_cons_b, test_num, test_ratio)

    # --- Top-K Community Indicators ---
    # Flatten all subreddits in train to find top K
    import ast
    from collections import Counter

    all_subs = []
    for s_str in df_train["requester_subreddits_at_request"]:
        try:
            subs = ast.literal_eval(s_str)
            if isinstance(subs, list):
                all_subs.extend(subs)
        except:
            pass

    top_k = [x[0] for x in Counter(all_subs).most_common(Config.TOP_K_SUBREDDITS)]

    def get_top_k_features(df):
        feats = np.zeros((len(df), len(top_k)))
        for i, s_str in enumerate(df["requester_subreddits_at_request"]):
            try:
                subs = set(ast.literal_eval(s_str))
                for j, sub in enumerate(top_k):
                    if sub in subs:
                        feats[i, j] = 1
            except:
                pass
        return feats

    train_topk = get_top_k_features(df_train)
    val_topk = get_top_k_features(df_val)
    test_topk = get_top_k_features(df_test)

    # --- TF-IDF for RF ---
    tfidf = TfidfVectorizer(max_features=Config.VOCAB_SIZE_TFIDF, stop_words="english")
    train_text = (
        df_train["request_title"] + " " + df_train["request_text_edit_aware"]
    ).fillna("")
    val_text = (
        df_val["request_title"] + " " + df_val["request_text_edit_aware"]
    ).fillna("")
    test_text = (
        df_test["request_title"] + " " + df_test["request_text_edit_aware"]
    ).fillna("")

    train_tfidf = tfidf.fit_transform(train_text).toarray()
    val_tfidf = tfidf.transform(val_text).toarray()
    test_tfidf = tfidf.transform(test_text).toarray()

    # --- Assemble Datasets ---

    # MLP Input
    # Dense Metadata: Arcsinh(Numerical)
    # Skip Metadata: Dense + TopK + Ratios
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(np.arcsinh(train_num))
    val_num_scaled = scaler.transform(np.arcsinh(val_num))
    test_num_scaled = scaler.transform(np.arcsinh(test_num))

    mlp_train = {
        "title": train_title_emb,
        "body": train_body_emb,
        "hist": train_hist,
        "mask": train_mask,
        "cent": train_cent,
        "meta_dense": train_num_scaled,
        "meta_skip": np.hstack([train_num_scaled, train_ratio, train_topk]),
    }
    mlp_val = {
        "title": val_title_emb,
        "body": val_body_emb,
        "hist": val_hist,
        "mask": val_mask,
        "cent": val_cent,
        "meta_dense": val_num_scaled,
        "meta_skip": np.hstack([val_num_scaled, val_ratio, val_topk]),
    }
    mlp_test = {
        "title": test_title_emb,
        "body": test_body_emb,
        "hist": test_hist,
        "mask": test_mask,
        "cent": test_cent,
        "meta_dense": test_num_scaled,
        "meta_skip": np.hstack([test_num_scaled, test_ratio, test_topk]),
    }

    # RF Input
    # TFIDF + Num + Ratio + TopK + Consistency + Interactions
    rf_train_X = np.hstack(
        [
            train_tfidf,
            train_num,
            train_ratio,
            train_topk,
            train_cons_t,
            train_cons_b,
            train_inter,
        ]
    )
    rf_val_X = np.hstack(
        [val_tfidf, val_num, val_ratio, val_topk, val_cons_t, val_cons_b, val_inter]
    )
    rf_test_X = np.hstack(
        [
            test_tfidf,
            test_num,
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
