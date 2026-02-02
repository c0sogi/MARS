import os
import json
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Configuration
# ==========================================
class Config:
    # Paths
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_DIR = "./working/idea_35"

    # Random Seed
    SEED = 42

    # Feature Engineering
    SBERT_MODEL = "all-MiniLM-L6-v2"  # Fast and effective
    TOP_K_SUBREDDITS = 50
    MAX_HISTORY_LEN = 50  # Limit history length for MLP

    # Random Forest Hyperparams
    RF_ESTIMATORS = 500
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"

    # MLP Hyperparams
    MLP_HIDDEN_DIM = 128
    MLP_DROPOUT = 0.5
    MLP_LR = 1e-3
    MLP_WEIGHT_DECAY = 1e-4
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 30  # High patience regime
    MLP_PATIENCE = 10

    # Ensemble
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5


# ==========================================
# Utilities
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ==========================================
# Data Processing & Caching
# ==========================================
def parse_list_col(col_data):
    # Parses string representation of list to actual list
    return col_data.apply(
        lambda x: (
            ast.literal_eval(x)
            if isinstance(x, str)
            else (x if isinstance(x, list) else [])
        )
    )


def load_and_preprocess_data(load_cached_data=True):
    ensure_dir(Config.CACHE_DIR)
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.pt")

    if load_cached_data and os.path.exists(cache_file):
        print("Loading cached data...")
        return torch.load(cache_file)

    print("Processing data from scratch...")

    # Load Metadata CSVs
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Parse Subreddit Lists
    for df in [df_train, df_val, df_test]:
        df["requester_subreddits_at_request"] = parse_list_col(
            df["requester_subreddits_at_request"]
        )
        # Fill text NaNs
        df["request_text_edit_aware"] = df["request_text_edit_aware"].fillna("")
        df["request_title"] = df["request_title"].fillna("")

    # --------------------------------------
    # 1. SBERT Embeddings (Title, Body, History)
    # --------------------------------------
    print("Generating SBERT embeddings...")
    model = SentenceTransformer(Config.SBERT_MODEL)

    def get_embeddings(text_list):
        return model.encode(
            text_list, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

    # Title & Body
    train_title_emb = get_embeddings(df_train["request_title"].tolist())
    val_title_emb = get_embeddings(df_val["request_title"].tolist())
    test_title_emb = get_embeddings(df_test["request_title"].tolist())

    train_body_emb = get_embeddings(df_train["request_text_edit_aware"].tolist())
    val_body_emb = get_embeddings(df_val["request_text_edit_aware"].tolist())
    test_body_emb = get_embeddings(df_test["request_text_edit_aware"].tolist())

    # History Subreddits
    # We need a vocabulary of subreddits to embed them efficiently
    all_subreddits = set()
    for sub_list in df_train["requester_subreddits_at_request"]:
        all_subreddits.update(sub_list)
    for sub_list in df_val["requester_subreddits_at_request"]:
        all_subreddits.update(sub_list)
    for sub_list in df_test["requester_subreddits_at_request"]:
        all_subreddits.update(sub_list)

    subreddit_list = list(all_subreddits)
    subreddit_to_idx = {sub: i for i, sub in enumerate(subreddit_list)}
    print(f"Unique subreddits: {len(subreddit_list)}")

    # Embed all unique subreddits
    subreddit_embeddings = get_embeddings(subreddit_list)  # (N_subs, D)

    # --------------------------------------
    # 2. Global Alignment Scalars & History Features
    # --------------------------------------
    def compute_history_features(df, title_embs, body_embs):
        global_sim_title = []
        global_sim_body = []
        history_embs_padded = []  # For MLP

        for idx, row in df.iterrows():
            subs = row["requester_subreddits_at_request"]
            if not subs:
                # No history
                centroid = np.zeros(subreddit_embeddings.shape[1])
                seq = np.zeros((Config.MAX_HISTORY_LEN, subreddit_embeddings.shape[1]))
            else:
                # Get indices
                sub_indices = [
                    subreddit_to_idx[s] for s in subs if s in subreddit_to_idx
                ]
                if not sub_indices:
                    centroid = np.zeros(subreddit_embeddings.shape[1])
                    seq = np.zeros(
                        (Config.MAX_HISTORY_LEN, subreddit_embeddings.shape[1])
                    )
                else:
                    # Get embeddings
                    cur_sub_embs = subreddit_embeddings[sub_indices]
                    centroid = np.mean(cur_sub_embs, axis=0)

                    # Pad/Truncate for MLP
                    seq = np.zeros(
                        (Config.MAX_HISTORY_LEN, subreddit_embeddings.shape[1])
                    )
                    slen = min(len(cur_sub_embs), Config.MAX_HISTORY_LEN)
                    seq[:slen] = cur_sub_embs[:slen]

            # Cosine Sim
            if np.linalg.norm(centroid) > 0:
                sim_t = cosine_similarity([title_embs[idx]], [centroid])[0][0]
                sim_b = cosine_similarity([body_embs[idx]], [centroid])[0][0]
            else:
                sim_t = 0.0
                sim_b = 0.0

            global_sim_title.append(sim_t)
            global_sim_body.append(sim_b)
            history_embs_padded.append(seq)

        return (
            np.array(global_sim_title),
            np.array(global_sim_body),
            np.array(history_embs_padded),
        )

    print("Computing history features...")
    train_sim_t, train_sim_b, train_hist = compute_history_features(
        df_train, train_title_emb, train_body_emb
    )
    val_sim_t, val_sim_b, val_hist = compute_history_features(
        df_val, val_title_emb, val_body_emb
    )
    test_sim_t, test_sim_b, test_hist = compute_history_features(
        df_test, test_title_emb, test_body_emb
    )

    # --------------------------------------
    # 3. Metadata & Top-K (RF Features)
    # --------------------------------------
    # Define metadata columns
    meta_cols = [
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

    # Top-K Subreddits
    all_train_subs = [
        s for sub_list in df_train["requester_subreddits_at_request"] for s in sub_list
    ]
    top_k = (
        pd.Series(all_train_subs)
        .value_counts()
        .head(Config.TOP_K_SUBREDDITS)
        .index.tolist()
    )

    def get_rf_features(df, sim_t, sim_b):
        # Metadata
        X_meta = df[meta_cols].fillna(0).values

        # Top-K
        X_topk = np.zeros((len(df), len(top_k)))
        for i, row in df.iterrows():
            subs = set(row["requester_subreddits_at_request"])
            for j, sub in enumerate(top_k):
                if sub in subs:
                    X_topk[i, j] = 1

        # Global Alignment
        X_sim = np.stack([sim_t, sim_b], axis=1)

        return np.hstack([X_meta, X_topk, X_sim])

    X_rf_train_base = get_rf_features(df_train, train_sim_t, train_sim_b)
    X_rf_val_base = get_rf_features(df_val, val_sim_t, val_sim_b)
    X_rf_test_base = get_rf_features(df_test, test_sim_t, test_sim_b)

    # TF-IDF for RF
    print("Computing TF-IDF...")
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
    train_text = df_train["request_title"] + " " + df_train["request_text_edit_aware"]
    val_text = df_val["request_title"] + " " + df_val["request_text_edit_aware"]
    test_text = df_test["request_title"] + " " + df_test["request_text_edit_aware"]

    X_tfidf_train = tfidf.fit_transform(train_text).toarray()
    X_tfidf_val = tfidf.transform(val_text).toarray()
    X_tfidf_test = tfidf.transform(test_text).toarray()

    # Final RF Data
    X_rf_train = np.hstack([X_rf_train_base, X_tfidf_train])
    X_rf_val = np.hstack([X_rf_val_base, X_tfidf_val])
    X_rf_test = np.hstack([X_rf_test_base, X_tfidf_test])

    # --------------------------------------
    # 4. MLP Data (Arcsinh Metadata)
    # --------------------------------------
    def get_mlp_meta(df, sim_t, sim_b):
        # Arcsinh transform numericals
        X_meta = df[meta_cols].fillna(0).values
        X_meta = np.arcsinh(X_meta)
        X_sim = np.stack([sim_t, sim_b], axis=1)
        return np.hstack([X_meta, X_sim])

    X_mlp_meta_train = get_mlp_meta(df_train, train_sim_t, train_sim_b)
    X_mlp_meta_val = get_mlp_meta(df_val, val_sim_t, val_sim_b)
    X_mlp_meta_test = get_mlp_meta(df_test, test_sim_t, test_sim_b)

    # Scale MLP Metadata
    scaler = StandardScaler()
    X_mlp_meta_train = scaler.fit_transform(X_mlp_meta_train)
    X_mlp_meta_val = scaler.transform(X_mlp_meta_val)
    X_mlp_meta_test = scaler.transform(X_mlp_meta_test)

    # Targets
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values

    # Test IDs for submission
    test_ids = df_test["request_id"].values

    data = {
        "rf": (X_rf_train, y_train, X_rf_val, y_val, X_rf_test, test_ids),
        "mlp": {
            "train": (
                train_title_emb,
                train_body_emb,
                train_hist,
                X_mlp_meta_train,
                y_train,
            ),
            "val": (val_title_emb, val_body_emb, val_hist, X_mlp_meta_val, y_val),
            "test": (
                test_title_emb,
                test_body_emb,
                test_hist,
                X_mlp_meta_test,
                test_ids,
            ),
        },
    }

    torch.save(data, cache_file)
    return data


# ==========================================
# Models
# ==========================================


# --- Random Forest ---
def train_rf(X_train, y_train, X_val, y_val, X_test):
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=Config.RF_ESTIMATORS,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        random_state=Config.SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    val_preds = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"RF Validation AUC: {auc:.6f}")

    test_preds = rf.predict_proba(X_test)[:, 1]
    return test_preds


# --- MLP ---
class PizzaDataset(Dataset):
    def __init__(self, title, body, history, meta, y=None):
        self.title = torch.FloatTensor(title)
        self.body = torch.FloatTensor(body)
        self.history = torch.FloatTensor(history)
        self.meta = torch.FloatTensor(meta)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        sample = {
            "title": self.title[idx],
            "body": self.body[idx],
            "history": self.history[idx],
            "meta": self.meta[idx],
        }
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample


class DualQueryMLP(nn.Module):
    def __init__(self, emb_dim, meta_dim, hidden_dim, dropout):
        super().__init__()

        # Dual Query Attention
        # Query: Title/Body (1, D), Key/Value: History (L, D)
        self.scale = np.sqrt(emb_dim)

        # Fusion
        # Input: Title(D) + Body(D) + Ctx_Title(D) + Ctx_Body(D) = 4*D
        self.fusion_dim = 4 * emb_dim

        # Gating
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.fusion_dim),
            nn.Sigmoid(),
        )

        # Main Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, history):
        # query: (B, D) -> (B, 1, D)
        # history: (B, L, D)
        Q = query.unsqueeze(1)
        K = history
        V = history

        # Scores: (B, 1, L)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Masking: Assume 0 vectors in history are padding
        # Simple check: norm of embedding
        mask = (history.abs().sum(dim=2) == 0).unsqueeze(1)  # (B, 1, L)
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V).squeeze(1)  # (B, D)
        return context

    def forward(self, title, body, history, meta):
        # 1. Dual Query Attention
        ctx_title = self.attention(title, history)
        ctx_body = self.attention(body, history)

        # 2. Concatenate Semantics
        # (B, 4*D)
        semantic_vec = torch.cat([title, body, ctx_title, ctx_body], dim=1)
        semantic_vec = self.dropout(semantic_vec)

        # 3. Gated Fusion
        gate = self.meta_gate(meta)
        fused = semantic_vec * gate

        # 4. Classification
        logits = self.classifier(fused)
        return logits.squeeze(1)


def train_mlp(data_dict):
    print("Training MLP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack
    train_t, train_b, train_h, train_m, train_y = data_dict["train"]
    val_t, val_b, val_h, val_m, val_y = data_dict["val"]
    test_t, test_b, test_h, test_m, test_ids = data_dict["test"]

    train_ds = PizzaDataset(train_t, train_b, train_h, train_m, train_y)
    val_ds = PizzaDataset(val_t, val_b, val_h, val_m, val_y)
    test_ds = PizzaDataset(test_t, test_b, test_h, test_m)

    train_loader = DataLoader(train_ds, batch_size=Config.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.MLP_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=Config.MLP_BATCH_SIZE)

    # Init Model
    emb_dim = train_t.shape[1]
    meta_dim = train_m.shape[1]

    model = DualQueryMLP(
        emb_dim, meta_dim, Config.MLP_HIDDEN_DIM, Config.MLP_DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.MLP_LR, weight_decay=Config.MLP_WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(Config.MLP_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["title"].to(device),
                batch["body"].to(device),
                batch["history"].to(device),
                batch["meta"].to(device),
            )
            loss = criterion(logits, batch["y"].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["title"].to(device),
                    batch["body"].to(device),
                    batch["history"].to(device),
                    batch["meta"].to(device),
                )
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(batch["y"].numpy())

        auc = roc_auc_score(val_targets, val_preds)
        print(
            f"Epoch {epoch+1}/{Config.MLP_EPOCHS} - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {auc:.6f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.MLP_PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Predict Test
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["title"].to(device),
                batch["body"].to(device),
                batch["history"].to(device),
                batch["meta"].to(device),
            )
            probs = torch.sigmoid(logits)
            test_preds.extend(probs.cpu().numpy())

    return np.array(test_preds)


# ==========================================
# Main Pipeline
# ==========================================
def run_pipeline():
    set_seed(Config.SEED)

    # 1. Load Data
    data = load_and_preprocess_data(load_cached_data=True)

    # 2. Train RF
    X_rf_train, y_train, X_rf_val, y_val, X_rf_test, test_ids = data["rf"]
    rf_preds = train_rf(X_rf_train, y_train, X_rf_val, y_val, X_rf_test)

    # 3. Train MLP
    mlp_preds = train_mlp(data["mlp"])

    # 4. Ensemble
    final_preds = (Config.ENSEMBLE_WEIGHT_RF * rf_preds) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_preds
    )

    # 5. Submission
    ensure_dir(Config.SUBMISSION_DIR)
    submission = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": final_preds}
    )
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
