import os
import ast
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration
# ==========================================


class Config:
    # Paths
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"
    CACHE_DIR = "./working/idea_8/"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Random Seed
    SEED = 42

    # Model Hyperparameters
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 2

    # MLP Hyperparameters
    MLP_HIDDEN_DIM = 128
    MLP_DROPOUT = 0.3
    MLP_LEARNING_RATE = 1e-3
    MLP_WEIGHT_DECAY = 1e-4
    MLP_EPOCHS = 40
    MLP_BATCH_SIZE = 32
    MLP_PATIENCE = 10

    # Feature Engineering
    TFIDF_MAX_FEATURES = 2000
    TFIDF_NGRAM_RANGE = (1, 2)


# ==========================================
# Data Processing & Caching
# ==========================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def get_tabular_features(df, train_cols=None):
    # Select numerical columns
    # We exclude ID, target, text, and leakage columns
    exclude = [
        "request_id",
        "requester_received_pizza",
        "request_text",
        "request_title",
        "request_text_edit_aware",
        "source_file",
        "giver_username_if_known",
        "requester_subreddits_at_request",
        "requester_username",
        "requester_user_flair",
        "post_was_edited",
    ]

    numeric_df = df.select_dtypes(include=[np.number])
    cols = [c for c in numeric_df.columns if c not in exclude]

    # Feature Engineering: Ratios
    # Upvote ratio
    if (
        "requester_upvotes_plus_downvotes_at_request" in cols
        and "requester_upvotes_minus_downvotes_at_request" in cols
    ):
        total = df["requester_upvotes_plus_downvotes_at_request"].replace(0, 1)
        diff = df["requester_upvotes_minus_downvotes_at_request"]
        # up - down = diff, up + down = total => 2*up = total + diff
        upvotes = (total + diff) / 2
        df["engineered_upvote_ratio"] = upvotes / total
        cols.append("engineered_upvote_ratio")

    # Text meta features
    if "request_text_edit_aware" in df.columns:
        texts = df["request_text_edit_aware"].fillna("").astype(str)
        df["meta_text_len"] = texts.apply(len)
        df["meta_word_count"] = texts.apply(lambda x: len(x.split()))
        df["meta_caps_ratio"] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x))
        )
        cols.extend(["meta_text_len", "meta_word_count", "meta_caps_ratio"])

    # If train_cols provided, align
    if train_cols is not None:
        # Add missing with 0
        for c in train_cols:
            if c not in df.columns:
                df[c] = 0
        return df[train_cols].values, train_cols

    return df[cols].values, cols


def process_data(load_cached_data=True):
    set_seed(Config.SEED)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_files = {
        "rf_data": os.path.join(Config.CACHE_DIR, "rf_data.npz"),
        "mlp_data": os.path.join(Config.CACHE_DIR, "mlp_data.npz"),
        "meta": os.path.join(Config.CACHE_DIR, "meta.json"),
    }

    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached data...")
        rf_data = np.load(cache_files["rf_data"], allow_pickle=True)
        mlp_data = np.load(cache_files["mlp_data"], allow_pickle=True)
        with open(cache_files["meta"], "r") as f:
            meta = json.load(f)

        data = {
            "rf": {k: rf_data[k] for k in rf_data},
            "mlp": {k: mlp_data[k] for k in mlp_data},
            "y_train": rf_data["y_train"],
            "y_val": rf_data["y_val"],
            "test_ids": rf_data["test_ids"],
        }
        return data

    print("Processing data from scratch...")

    # Load raw data
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Target
    y_train = df_train["requester_received_pizza"].astype(int).values
    y_val = df_val["requester_received_pizza"].astype(int).values
    test_ids = df_test["request_id"].values

    # Text Data
    train_text = df_train["request_text_edit_aware"].fillna("").astype(str).tolist()
    val_text = df_val["request_text_edit_aware"].fillna("").astype(str).tolist()
    test_text = df_test["request_text_edit_aware"].fillna("").astype(str).tolist()

    # ==========================
    # Stream A: Random Forest
    # ==========================
    # 1. TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=Config.TFIDF_NGRAM_RANGE,
        max_features=Config.TFIDF_MAX_FEATURES,
        binary=True,
        stop_words="english",
    )
    X_train_tfidf = tfidf.fit_transform(train_text).toarray()
    X_val_tfidf = tfidf.transform(val_text).toarray()
    X_test_tfidf = tfidf.transform(test_text).toarray()

    # 2. Tabular (Imputed)
    X_train_tab_raw, tab_cols = get_tabular_features(df_train)
    X_val_tab_raw, _ = get_tabular_features(df_val, tab_cols)
    X_test_tab_raw, _ = get_tabular_features(df_test, tab_cols)

    imputer = SimpleImputer(strategy="median")
    X_train_tab_imp = imputer.fit_transform(X_train_tab_raw)
    X_val_tab_imp = imputer.transform(X_val_tab_raw)
    X_test_tab_imp = imputer.transform(X_test_tab_raw)

    # Concat for RF
    X_train_rf = np.hstack([X_train_tfidf, X_train_tab_imp])
    X_val_rf = np.hstack([X_val_tfidf, X_val_tab_imp])
    X_test_rf = np.hstack([X_test_tfidf, X_test_tab_imp])

    # ==========================
    # Stream B: Gated MLP
    # ==========================
    # 1. SBERT (Request Text)
    sbert = SentenceTransformer(Config.SBERT_MODEL_NAME)
    X_train_sbert = sbert.encode(train_text, show_progress_bar=False)
    X_val_sbert = sbert.encode(val_text, show_progress_bar=False)
    X_test_sbert = sbert.encode(test_text, show_progress_bar=False)

    # 2. Community SBERT (History)
    def get_community_embeddings(df_in):
        embeddings = []
        for subs_str in df_in["requester_subreddits_at_request"]:
            subs = safe_literal_eval(subs_str)
            if not subs:
                embeddings.append(np.zeros(384))  # 384 is dim of miniLM
            else:
                # Embed all subreddit names and take mean
                sub_embs = sbert.encode(subs, show_progress_bar=False)
                embeddings.append(np.mean(sub_embs, axis=0))
        return np.array(embeddings)

    X_train_comm = get_community_embeddings(df_train)
    X_val_comm = get_community_embeddings(df_val)
    X_test_comm = get_community_embeddings(df_test)

    # 3. Tabular (Arcsinh + Scaler)
    # Apply arcsinh to handle heavy tails
    X_train_tab_arc = np.arcsinh(np.nan_to_num(X_train_tab_raw))
    X_val_tab_arc = np.arcsinh(np.nan_to_num(X_val_tab_raw))
    X_test_tab_arc = np.arcsinh(np.nan_to_num(X_test_tab_raw))

    scaler = StandardScaler()
    X_train_tab_scaled = scaler.fit_transform(X_train_tab_arc)
    X_val_tab_scaled = scaler.transform(X_val_tab_arc)
    X_test_tab_scaled = scaler.transform(X_test_tab_arc)

    # Save to cache
    np.savez(
        cache_files["rf_data"],
        X_train=X_train_rf,
        X_val=X_val_rf,
        X_test=X_test_rf,
        y_train=y_train,
        y_val=y_val,
        test_ids=test_ids,
    )

    np.savez(
        cache_files["mlp_data"],
        X_train_text=X_train_sbert,
        X_train_comm=X_train_comm,
        X_train_tab=X_train_tab_scaled,
        X_val_text=X_val_sbert,
        X_val_comm=X_val_comm,
        X_val_tab=X_val_tab_scaled,
        X_test_text=X_test_sbert,
        X_test_comm=X_test_comm,
        X_test_tab=X_test_tab_scaled,
    )

    with open(cache_files["meta"], "w") as f:
        json.dump({"tab_cols": tab_cols}, f)

    data = {
        "rf": {"X_train": X_train_rf, "X_val": X_val_rf, "X_test": X_test_rf},
        "mlp": {
            "X_train_text": X_train_sbert,
            "X_train_comm": X_train_comm,
            "X_train_tab": X_train_tab_scaled,
            "X_val_text": X_val_sbert,
            "X_val_comm": X_val_comm,
            "X_val_tab": X_val_tab_scaled,
            "X_test_text": X_test_sbert,
            "X_test_comm": X_test_comm,
            "X_test_tab": X_test_tab_scaled,
        },
        "y_train": y_train,
        "y_val": y_val,
        "test_ids": test_ids,
    }
    return data


# ==========================================
# Models
# ==========================================


class GatedMLP(nn.Module):
    def __init__(self, text_dim, comm_dim, tab_dim, hidden_dim, dropout):
        super(GatedMLP, self).__init__()

        # Dimensions
        self.sem_dim = text_dim + comm_dim

        # Tabular encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gate Generator: Takes encoded tabular info, outputs gate for semantic vector
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_dim, self.sem_dim), nn.Sigmoid()
        )

        # Fusion Layer
        # Input: Gated Semantic (sem_dim) + Encoded Tabular (hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.sem_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_emb, comm_emb, tab_features):
        # 1. Semantic Vector
        sem_vector = torch.cat([text_emb, comm_emb], dim=1)

        # 2. Tabular Encoding
        tab_encoded = self.tab_encoder(tab_features)

        # 3. Gating
        gate = self.gate_generator(tab_encoded)
        gated_sem = sem_vector * gate

        # 4. Fusion
        combined = torch.cat([gated_sem, tab_encoded], dim=1)
        output = self.classifier(combined)
        return output


# ==========================================
# Training & Inference
# ==========================================


def train_rf_stream(X_train, y_train, X_val, y_val, X_test):
    print("Training Random Forest Stream...")
    rf = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
        class_weight="balanced",
        random_state=Config.SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    val_preds = rf.predict_proba(X_val)[:, 1]
    test_preds = rf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_val, val_preds)
    print(f"RF Validation AUC: {auc}")
    return val_preds, test_preds


def train_mlp_stream(data, y_train, y_val):
    print("Training Gated MLP Stream...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Tensors
    X_train_text = torch.FloatTensor(data["X_train_text"]).to(device)
    X_train_comm = torch.FloatTensor(data["X_train_comm"]).to(device)
    X_train_tab = torch.FloatTensor(data["X_train_tab"]).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    X_val_text = torch.FloatTensor(data["X_val_text"]).to(device)
    X_val_comm = torch.FloatTensor(data["X_val_comm"]).to(device)
    X_val_tab = torch.FloatTensor(data["X_val_tab"]).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    X_test_text = torch.FloatTensor(data["X_test_text"]).to(device)
    X_test_comm = torch.FloatTensor(data["X_test_comm"]).to(device)
    X_test_tab = torch.FloatTensor(data["X_test_tab"]).to(device)

    # Dataset
    train_ds = TensorDataset(X_train_text, X_train_comm, X_train_tab, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=Config.MLP_BATCH_SIZE, shuffle=True)

    # Model
    model = GatedMLP(
        text_dim=X_train_text.shape[1],
        comm_dim=X_train_comm.shape[1],
        tab_dim=X_train_tab.shape[1],
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout=Config.MLP_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MLP_LEARNING_RATE,
        weight_decay=Config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCELoss()

    # Training Loop
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(Config.MLP_EPOCHS):
        model.train()
        train_loss = 0
        for b_text, b_comm, b_tab, b_y in train_loader:
            optimizer.zero_grad()
            preds = model(b_text, b_comm, b_tab)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_text, X_val_comm, X_val_tab).cpu().numpy().flatten()
            val_loss = criterion(
                torch.FloatTensor(val_preds).unsqueeze(1).to(device), y_val_t
            ).item()

        val_auc = roc_auc_score(y_val, val_preds)
        print(
            f"Epoch {epoch+1}/{Config.MLP_EPOCHS} - Loss: {train_loss/len(train_loader)} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.MLP_PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Inference
    model.eval()
    with torch.no_grad():
        final_val_preds = (
            model(X_val_text, X_val_comm, X_val_tab).cpu().numpy().flatten()
        )
        final_test_preds = (
            model(X_test_text, X_test_comm, X_test_tab).cpu().numpy().flatten()
        )

    return final_val_preds, final_test_preds


def run_pipeline():
    set_seed(Config.SEED)

    # 1. Process Data
    data = process_data(load_cached_data=True)

    # 2. Train RF
    rf_val, rf_test = train_rf_stream(
        data["rf"]["X_train"],
        data["y_train"],
        data["rf"]["X_val"],
        data["y_val"],
        data["rf"]["X_test"],
    )

    # 3. Train MLP
    mlp_val, mlp_test = train_mlp_stream(data["mlp"], data["y_train"], data["y_val"])

    # 4. Ensemble (Average)
    final_val_preds = 0.5 * rf_val + 0.5 * mlp_val
    final_test_preds = 0.5 * rf_test + 0.5 * mlp_test

    val_auc = roc_auc_score(data["y_val"], final_val_preds)
    print(f"Ensemble Validation AUC: {val_auc}")

    # 5. Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame(
        {"request_id": data["test_ids"], "requester_received_pizza": final_test_preds}
    )
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
