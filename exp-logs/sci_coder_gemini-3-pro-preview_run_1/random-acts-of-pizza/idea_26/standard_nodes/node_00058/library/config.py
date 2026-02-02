import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer


# ==========================================
# Configuration
# ==========================================
class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_26"
    CACHE_DIR = WORKING_DIR
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Hyperparameters
    SEED = 42
    RF_ESTIMATORS = 500
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"

    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT = 0.3
    MLP_LR = 1e-4
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15
    MLP_BATCH_SIZE = 32

    # Feature Engineering
    TOP_K_SUBREDDITS = 50
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"


# ==========================================
# Data Processing & Caching
# ==========================================
def parse_list_col(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []


def get_sbert_embeddings(texts, model_name, cache_path, load_cache=True):
    if load_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def process_data(load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Parse Subreddits
    for df in [df_train, df_val, df_test]:
        df["subreddits"] = df["requester_subreddits_at_request"].apply(parse_list_col)
        # Fill NaNs in text
        df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("")
        df[Config.TITLE_COL] = df[Config.TITLE_COL].fillna("")

    # --- Feature Engineering: Top-K Subreddits ---
    # Count subreddits in train
    all_subs = [sub for subs in df_train["subreddits"] for sub in subs]
    if all_subs:
        top_subs = (
            pd.Series(all_subs)
            .value_counts()
            .head(Config.TOP_K_SUBREDDITS)
            .index.tolist()
        )
    else:
        top_subs = []

    def add_top_k_features(df):
        for sub in top_subs:
            df[f"sub_flag_{sub}"] = df["subreddits"].apply(
                lambda x: 1 if sub in x else 0
            )
        return df

    df_train = add_top_k_features(df_train)
    df_val = add_top_k_features(df_val)
    df_test = add_top_k_features(df_test)

    top_k_cols = [f"sub_flag_{sub}" for sub in top_subs]

    # --- Feature Engineering: TF-IDF (for RF) ---
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    # Combine title + body
    train_text = df_train[Config.TITLE_COL] + " " + df_train[Config.TEXT_COL]
    val_text = df_val[Config.TITLE_COL] + " " + df_val[Config.TEXT_COL]
    test_text = df_test[Config.TITLE_COL] + " " + df_test[Config.TEXT_COL]

    X_tfidf_train = tfidf.fit_transform(train_text).toarray()
    X_tfidf_val = tfidf.transform(val_text).toarray()
    X_tfidf_test = tfidf.transform(test_text).toarray()

    # --- Feature Engineering: Metadata ---
    meta_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_upvotes_minus_downvotes_at_request",
    ]

    # Fill NaNs in metadata
    imputer_fill = 0
    for col in meta_cols:
        df_train[col] = df_train[col].fillna(imputer_fill)
        df_val[col] = df_val[col].fillna(imputer_fill)
        df_test[col] = df_test[col].fillna(imputer_fill)

    # Scaling for MLP (Arcsinh + Standard)
    scaler = StandardScaler()

    def preprocess_meta_mlp(df):
        return np.arcsinh(df[meta_cols].values)

    X_meta_mlp_train = preprocess_meta_mlp(df_train)
    X_meta_mlp_val = preprocess_meta_mlp(df_val)
    X_meta_mlp_test = preprocess_meta_mlp(df_test)

    scaler.fit(X_meta_mlp_train)
    X_meta_mlp_train = scaler.transform(X_meta_mlp_train)
    X_meta_mlp_val = scaler.transform(X_meta_mlp_val)
    X_meta_mlp_test = scaler.transform(X_meta_mlp_test)

    # --- Feature Engineering: SBERT (for MLP) ---
    # Title
    title_emb_train = get_sbert_embeddings(
        df_train[Config.TITLE_COL].tolist(),
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "title_train.npy"),
        load_cached_data,
    )
    title_emb_val = get_sbert_embeddings(
        df_val[Config.TITLE_COL].tolist(),
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "title_val.npy"),
        load_cached_data,
    )
    title_emb_test = get_sbert_embeddings(
        df_test[Config.TITLE_COL].tolist(),
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "title_test.npy"),
        load_cached_data,
    )

    # Body
    body_emb_train = get_sbert_embeddings(
        df_train[Config.TEXT_COL].tolist(),
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "body_train.npy"),
        load_cached_data,
    )
    body_emb_val = get_sbert_embeddings(
        df_val[Config.TEXT_COL].tolist(),
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "body_val.npy"),
        load_cached_data,
    )
    body_emb_test = get_sbert_embeddings(
        df_test[Config.TEXT_COL].tolist(),
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "body_test.npy"),
        load_cached_data,
    )

    # History (Complex: List of strings -> List of embeddings -> Aggregated or Sequence)
    unique_subs = (
        set(all_subs)
        | set([s for subs in df_val["subreddits"] for s in subs])
        | set([s for subs in df_test["subreddits"] for s in subs])
    )
    unique_subs = list(unique_subs)
    if not unique_subs:
        unique_subs = ["placeholder"]

    sub_to_idx = {sub: i + 1 for i, sub in enumerate(unique_subs)}  # 0 is padding

    # Embed unique subreddits
    sub_embeddings = get_sbert_embeddings(
        unique_subs,
        Config.SBERT_MODEL_NAME,
        os.path.join(Config.CACHE_DIR, "sub_embeddings.npy"),
        load_cached_data,
    )
    # Add padding embedding (zeros)
    sub_embeddings_pad = np.vstack(
        [np.zeros((1, sub_embeddings.shape[1])), sub_embeddings]
    )

    def encode_history(subs_list, max_len=20):
        indices = [sub_to_idx.get(s, 0) for s in subs_list[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices

    hist_idx_train = np.array([encode_history(s) for s in df_train["subreddits"]])
    hist_idx_val = np.array([encode_history(s) for s in df_val["subreddits"]])
    hist_idx_test = np.array([encode_history(s) for s in df_test["subreddits"]])

    # --- Prepare RF Data ---
    # Concatenate TF-IDF + Top-K + Meta (Raw)
    X_rf_train = np.hstack(
        [X_tfidf_train, df_train[top_k_cols].values, df_train[meta_cols].values]
    )
    X_rf_val = np.hstack(
        [X_tfidf_val, df_val[top_k_cols].values, df_val[meta_cols].values]
    )
    X_rf_test = np.hstack(
        [X_tfidf_test, df_test[top_k_cols].values, df_test[meta_cols].values]
    )

    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values

    # Return dictionary
    return {
        "rf": (X_rf_train, y_train, X_rf_val, y_val, X_rf_test),
        "mlp": {
            "train": (
                title_emb_train,
                body_emb_train,
                hist_idx_train,
                X_meta_mlp_train,
                y_train,
            ),
            "val": (title_emb_val, body_emb_val, hist_idx_val, X_meta_mlp_val, y_val),
            "test": (title_emb_test, body_emb_test, hist_idx_test, X_meta_mlp_test),
            "sub_emb": sub_embeddings_pad,
        },
        "ids": df_test["request_id"].values,
    }


# ==========================================
# Models
# ==========================================
class PizzaDataset(Dataset):
    def __init__(self, title_emb, body_emb, hist_idx, meta, y=None):
        self.title_emb = torch.FloatTensor(title_emb)
        self.body_emb = torch.FloatTensor(body_emb)
        self.hist_idx = torch.LongTensor(hist_idx)
        self.meta = torch.FloatTensor(meta)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        sample = {
            "title": self.title_emb[idx],
            "body": self.body_emb[idx],
            "hist": self.hist_idx[idx],
            "meta": self.meta[idx],
        }
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample


class AttentionGatedMLP(nn.Module):
    def __init__(self, sub_embeddings, meta_dim, hidden_dim, dropout):
        super().__init__()
        self.sub_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(sub_embeddings), freeze=True, padding_idx=0
        )
        input_dim = sub_embeddings.shape[1]  # SBERT dim (384)

        # Branch 3: Attention
        # Query: Title (384), Key: History (384)
        self.attention_scale = np.sqrt(input_dim)

        # Branch 4: Metadata Gate
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 3),  # Gate for Title, Body, Attended
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, title, body, hist, meta):
        # title: [B, D], body: [B, D], hist: [B, L], meta: [B, M]

        # History Embeddings
        hist_emb = self.sub_emb(hist)  # [B, L, D]

        # Attention
        # Q = title.unsqueeze(1) # [B, 1, D]
        # K = hist_emb # [B, L, D]
        # Scores = Q @ K.T
        scores = torch.bmm(title.unsqueeze(1), hist_emb.transpose(1, 2))  # [B, 1, L]
        scores = scores / self.attention_scale

        # Mask padding (index 0)
        mask = (hist == 0).unsqueeze(1)  # [B, 1, L]
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [B, 1, L]
        context = torch.bmm(attn_weights, hist_emb).squeeze(1)  # [B, D]

        # Concatenate
        combined = torch.cat([title, body, context], dim=1)  # [B, 3*D]

        # Gate
        gate = torch.sigmoid(self.meta_gate(meta))  # [B, 3*D]

        # Gated Fusion
        gated_combined = combined * gate

        logits = self.fusion(gated_combined)
        return logits.squeeze(1)


# ==========================================
# Execution Logic
# ==========================================
def train_rf(data):
    X_train, y_train, X_val, y_val, X_test = data

    clf = RandomForestClassifier(
        n_estimators=Config.RF_ESTIMATORS,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        random_state=Config.SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"RF Validation AUC: {auc}")

    test_probs = clf.predict_proba(X_test)[:, 1]
    return test_probs


def train_mlp(data):
    train_data, val_data, test_data, sub_emb = (
        data["train"],
        data["val"],
        data["test"],
        data["sub_emb"],
    )

    train_ds = PizzaDataset(*train_data)
    val_ds = PizzaDataset(*val_data)
    test_ds = PizzaDataset(*test_data)

    train_loader = DataLoader(train_ds, batch_size=Config.MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.MLP_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=Config.MLP_BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionGatedMLP(
        sub_embeddings=sub_emb,
        meta_dim=train_data[3].shape[1],
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout=Config.MLP_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.MLP_LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(Config.MLP_EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["title"].to(device),
                batch["body"].to(device),
                batch["hist"].to(device),
                batch["meta"].to(device),
            )
            loss = criterion(logits, batch["y"].to(device))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["title"].to(device),
                    batch["body"].to(device),
                    batch["hist"].to(device),
                    batch["meta"].to(device),
                )
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(batch["y"].numpy())

        auc = roc_auc_score(val_targets, val_preds)
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.MLP_PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best AUC: {best_auc}")
            break

    print(f"MLP Best Validation AUC: {best_auc}")

    # Inference
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["title"].to(device),
                batch["body"].to(device),
                batch["hist"].to(device),
                batch["meta"].to(device),
            )
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())

    return np.array(test_preds)


def run_pipeline():
    # Set seeds
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    print("Processing Data...")
    data = process_data(load_cached_data=True)

    print("Training RF...")
    rf_preds = train_rf(data["rf"])

    print("Training MLP...")
    mlp_preds = train_mlp(data["mlp"])

    # Ensemble (Simple Average)
    final_preds = 0.5 * rf_preds + 0.5 * mlp_preds

    # Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame(
        {"request_id": data["ids"], "requester_received_pizza": final_preds}
    )
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
