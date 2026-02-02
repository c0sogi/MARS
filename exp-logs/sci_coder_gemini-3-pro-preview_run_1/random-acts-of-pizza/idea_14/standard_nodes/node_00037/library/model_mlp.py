import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

from library import config
from library import data_loader
from library.feature_engineering import FeatureEngineer

# -----------------------------------------------------------------------------
# 1. Dataset Class
# -----------------------------------------------------------------------------


class PizzaDataset(Dataset):
    def __init__(
        self,
        request_emb,
        history_emb,
        history_mask,
        vol_meta,
        rep_meta,
        all_meta,
        labels=None,
    ):
        """
        Args:
            request_emb (Tensor): (N, D) SBERT embeddings of request text.
            history_emb (Tensor): (N, L, D) SBERT embeddings of user subreddits.
            history_mask (Tensor): (N, L) Boolean mask (True = padding).
            vol_meta (Tensor): (N, D_vol) Volume-related metadata.
            rep_meta (Tensor): (N, D_rep) Reputation-related metadata.
            all_meta (Tensor): (N, D_meta) All numerical metadata.
            labels (Tensor, optional): (N,) Binary targets.
        """
        self.request_emb = request_emb
        self.history_emb = history_emb
        self.history_mask = history_mask
        self.vol_meta = vol_meta
        self.rep_meta = rep_meta
        self.all_meta = all_meta
        self.labels = labels

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        item = {
            "request_emb": self.request_emb[idx],
            "history_emb": self.history_emb[idx],
            "history_mask": self.history_mask[idx],
            "vol_meta": self.vol_meta[idx],
            "rep_meta": self.rep_meta[idx],
            "all_meta": self.all_meta[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# -----------------------------------------------------------------------------
# 2. Data Processing & Caching
# -----------------------------------------------------------------------------


def parse_subreddits(x):
    """Safely parse the string representation of list of subreddits."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(x, list):
        return x
    return []


def prepare_mlp_data(load_cached_data=True, debug=False):
    """
    Generates features for the MLP model.
    Handles SBERT embedding, history padding, and metadata splitting.
    """
    cache_path = os.path.join(config.WORKING_DIR, "mlp_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print("Loading MLP data from cache...")
        data = np.load(cache_path, allow_pickle=True)

        # Validate cache against configuration (Cite debug_lesson_1)
        # If not in debug mode, but cache size is small, invalidate.
        if not debug and data["y_train"].shape[0] <= config.DEBUG_SAMPLE_SIZE:
            print(
                f"Detected stale debug MLP cache (Size: {data['y_train'].shape[0]}). Reprocessing..."
            )
        else:
            return {k: data[k] for k in data.files}

    print("Processing MLP data from scratch...")

    # Load raw data
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=load_cached_data, debug=debug
    )

    # Initialize Feature Engineer
    fe = FeatureEngineer()
    fe._load_sbert()  # Ensure model is loaded

    # --- A. Request Embeddings (Title + Body) ---
    print("  Encoding Requests...")

    def encode_requests(df):
        texts = (
            df[config.TEXT_COL_TITLE].fillna("")
            + " "
            + df[config.TEXT_COL_BODY].fillna("")
        ).tolist()
        return fe.sbert_model.encode(texts, batch_size=64, show_progress_bar=False)

    req_train = encode_requests(train_df)
    req_val = encode_requests(val_df)
    req_test = encode_requests(test_df)

    # --- B. History Embeddings (Subreddits) ---
    print("  Encoding Histories...")
    # 1. Collect all unique subreddits to encode once
    all_subs = set()
    for df in [train_df, val_df, test_df]:
        for sub_list in df["requester_subreddits_at_request"].apply(parse_subreddits):
            all_subs.update(sub_list)

    unique_subs = sorted(list(all_subs))
    sub_to_idx = {sub: i for i, sub in enumerate(unique_subs)}

    # Encode unique subreddits
    if unique_subs:
        sub_embeddings = fe.sbert_model.encode(
            unique_subs, batch_size=64, show_progress_bar=False
        )
    else:
        sub_embeddings = np.zeros((0, 384))  # Fallback

    # 2. Map users to embeddings with padding
    MAX_HIST_LEN = 50  # Truncate long histories to save memory/compute
    EMBED_DIM = sub_embeddings.shape[1] if len(unique_subs) > 0 else 384

    def process_history(df):
        N = len(df)
        hist_emb = np.zeros((N, MAX_HIST_LEN, EMBED_DIM), dtype=np.float32)
        hist_mask = np.ones((N, MAX_HIST_LEN), dtype=bool)  # True = Padding

        for i, row in enumerate(
            df["requester_subreddits_at_request"].apply(parse_subreddits)
        ):
            valid_subs = [s for s in row if s in sub_to_idx][:MAX_HIST_LEN]
            if not valid_subs:
                hist_mask[i, 0] = False
                continue

            indices = [sub_to_idx[s] for s in valid_subs]
            count = len(indices)

            hist_emb[i, :count, :] = sub_embeddings[indices]
            hist_mask[i, :count] = False  # False = Real Data

        return hist_emb, hist_mask

    hist_train, mask_train = process_history(train_df)
    hist_val, mask_val = process_history(val_df)
    hist_test, mask_test = process_history(test_df)

    # --- C. Metadata Processing ---
    print("  Processing Metadata...")
    # Generate full metadata
    meta_train_df = fe.generate_metadata_features(train_df)
    meta_val_df = fe.generate_metadata_features(val_df)
    meta_test_df = fe.generate_metadata_features(test_df)

    # Define Gate Feature Groups
    # Volume: Account age, number of posts/comments/subs
    vol_cols = [
        c
        for c in meta_train_df.columns
        if any(x in c for x in ["age", "number_of", "days_since", "len"])
    ]
    # Reputation: Upvotes, downvotes, ratios
    rep_cols = [
        c
        for c in meta_train_df.columns
        if any(x in c for x in ["upvotes", "ratio", "flair"])
    ]

    # If overlap or missing, ensure robust selection
    if not vol_cols:
        vol_cols = meta_train_df.columns.tolist()
    if not rep_cols:
        rep_cols = meta_train_df.columns.tolist()

    # Impute and Scale
    # We use 0 imputation for Arcsinh features (log(0+1)=0) and median for others if needed
    # Since generate_metadata_features handles most NaNs, we just do a final fillna(0)

    scaler_all = StandardScaler()
    all_train = scaler_all.fit_transform(meta_train_df.fillna(0))
    all_val = scaler_all.transform(meta_val_df.fillna(0))
    all_test = scaler_all.transform(meta_test_df.fillna(0))

    # Create subsets for gates (using indices from columns)
    vol_indices = [meta_train_df.columns.get_loc(c) for c in vol_cols]
    rep_indices = [meta_train_df.columns.get_loc(c) for c in rep_cols]

    vol_train, rep_train = all_train[:, vol_indices], all_train[:, rep_indices]
    vol_val, rep_val = all_val[:, vol_indices], all_val[:, rep_indices]
    vol_test, rep_test = all_test[:, vol_indices], all_test[:, rep_indices]

    # --- D. Targets ---
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    # --- Save to Cache ---
    data_dict = {
        "req_train": req_train,
        "req_val": req_val,
        "req_test": req_test,
        "hist_train": hist_train,
        "hist_val": hist_val,
        "hist_test": hist_test,
        "mask_train": mask_train,
        "mask_val": mask_val,
        "mask_test": mask_test,
        "vol_train": vol_train,
        "vol_val": vol_val,
        "vol_test": vol_test,
        "rep_train": rep_train,
        "rep_val": rep_val,
        "rep_test": rep_test,
        "all_train": all_train,
        "all_val": all_val,
        "all_test": all_test,
        "y_train": y_train,
        "y_val": y_val,
    }

    np.savez_compressed(cache_path, **data_dict)
    return data_dict


# -----------------------------------------------------------------------------
# 3. Model Architecture
# -----------------------------------------------------------------------------


class HierarchicalGatedNet(nn.Module):
    def __init__(
        self, embed_dim, vol_dim, rep_dim, all_meta_dim, hidden_dim, dropout=0.3
    ):
        super().__init__()

        # 1. Semantic Branch (Request)
        self.request_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. History Branch (Attention)
        # Query = Request (hidden_dim), Key/Value = History (embed_dim)
        # We project History to hidden_dim first
        self.history_proj = nn.Linear(embed_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout
        )

        # 3. Metadata Branch
        self.meta_encoder = nn.Sequential(
            nn.Linear(all_meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Gates
        # Validity Gate: Volume -> Scalar (0-1)
        self.validity_gate = nn.Sequential(
            nn.Linear(vol_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Credibility Gate: Reputation -> Scalar (0-1)
        self.credibility_gate = nn.Sequential(
            nn.Linear(rep_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # 5. Classifier
        # Input: Request + Gated History + Meta
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, req_emb, hist_emb, hist_mask, vol_meta, rep_meta, all_meta):
        # A. Encode Request
        # req_emb: (B, Embed_Dim) -> (B, Hidden)
        req_feat = self.request_encoder(req_emb)

        # B. Encode History via Attention
        # hist_emb: (B, L, Embed_Dim) -> (B, L, Hidden)
        hist_feat_proj = self.history_proj(hist_emb)

        # Attention: Query=Request, Key=History, Value=History
        # Query needs sequence dim: (B, 1, Hidden)
        query = req_feat.unsqueeze(1)

        # attn_output: (B, 1, Hidden)
        # key_padding_mask: (B, L) - True indicates padding
        attn_output, _ = self.attention(
            query, hist_feat_proj, hist_feat_proj, key_padding_mask=hist_mask
        )
        hist_context = attn_output.squeeze(1)  # (B, Hidden)

        # C. Validity Gating
        # "Is the history valid/rich enough?"
        val_score = self.validity_gate(vol_meta)  # (B, 1)
        gated_history = hist_context * val_score

        # D. Encode Metadata
        meta_feat = self.meta_encoder(all_meta)  # (B, Hidden)

        # E. Combine
        combined = torch.cat(
            [req_feat, gated_history, meta_feat], dim=1
        )  # (B, Hidden*3)

        # F. Credibility Gating
        # "Is the user trustworthy?"
        cred_score = self.credibility_gate(rep_meta)  # (B, 1)
        gated_combined = combined * cred_score

        # G. Classify
        logits = self.classifier(gated_combined)
        return logits


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------


def train_mlp_model(load_cached_data=True, debug=False):
    """
    Trains the Hierarchical Reliability-Gated MLP.
    Returns model, validation probabilities, test probabilities, and metrics.
    """
    print("=" * 40)
    print("Stream B: Hierarchical Reliability-Gated MLP")
    print("=" * 40)

    # Reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    data = prepare_mlp_data(load_cached_data=load_cached_data, debug=debug)

    # Create Datasets
    train_ds = PizzaDataset(
        data["req_train"],
        data["hist_train"],
        data["mask_train"],
        data["vol_train"],
        data["rep_train"],
        data["all_train"],
        data["y_train"],
    )
    val_ds = PizzaDataset(
        data["req_val"],
        data["hist_val"],
        data["mask_val"],
        data["vol_val"],
        data["rep_val"],
        data["all_val"],
        data["y_val"],
    )
    test_ds = PizzaDataset(
        data["req_test"],
        data["hist_test"],
        data["mask_test"],
        data["vol_test"],
        data["rep_test"],
        data["all_test"],
        None,
    )

    # Create DataLoaders
    batch_size = config.MLP_PARAMS["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model
    embed_dim = data["req_train"].shape[1]
    vol_dim = data["vol_train"].shape[1]
    rep_dim = data["rep_train"].shape[1]
    all_meta_dim = data["all_train"].shape[1]

    model = HierarchicalGatedNet(
        embed_dim=embed_dim,
        vol_dim=vol_dim,
        rep_dim=rep_dim,
        all_meta_dim=all_meta_dim,
        hidden_dim=config.MLP_PARAMS["hidden_dim"],
        dropout=config.MLP_PARAMS["dropout_rate"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.MLP_PARAMS["learning_rate"],
        weight_decay=config.MLP_PARAMS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    epochs = config.MLP_PARAMS["epochs"]
    patience = config.MLP_PARAMS["patience"]
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # Move to device
            req = batch["request_emb"].to(device)
            hist = batch["history_emb"].to(device)
            mask = batch["history_mask"].to(device)
            vol = batch["vol_meta"].to(device).float()
            rep = batch["rep_meta"].to(device).float()
            meta = batch["all_meta"].to(device).float()
            labels = batch["label"].to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(req, hist, mask, vol, rep, meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                req = batch["request_emb"].to(device)
                hist = batch["history_emb"].to(device)
                mask = batch["history_mask"].to(device)
                vol = batch["vol_meta"].to(device).float()
                rep = batch["rep_meta"].to(device).float()
                meta = batch["all_meta"].to(device).float()
                labels = batch["label"].to(device).float()

                logits = model(req, hist, mask, vol, rep, meta)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val AUC: {val_auc}"
        )

        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 4. Final Inference
    print("Restoring best model and generating predictions...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    # Recalculate Val Probs with best model
    final_val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            req = batch["request_emb"].to(device)
            hist = batch["history_emb"].to(device)
            mask = batch["history_mask"].to(device)
            vol = batch["vol_meta"].to(device).float()
            rep = batch["rep_meta"].to(device).float()
            meta = batch["all_meta"].to(device).float()

            logits = model(req, hist, mask, vol, rep, meta)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            final_val_probs.extend(probs)

    # Test Probs
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            req = batch["request_emb"].to(device)
            hist = batch["history_emb"].to(device)
            mask = batch["history_mask"].to(device)
            vol = batch["vol_meta"].to(device).float()
            rep = batch["rep_meta"].to(device).float()
            meta = batch["all_meta"].to(device).float()

            logits = model(req, hist, mask, vol, rep, meta)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            test_probs.extend(probs)

    return {
        "model": model,
        "val_probs": np.array(final_val_probs),
        "test_probs": np.array(test_probs),
        "auc": best_val_auc,
        "y_val": data["y_val"],
    }
