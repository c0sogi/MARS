import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from library import config, data_loader, features

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class PizzaDataset(Dataset):
    def __init__(self, request_embs, history_seqs, metadata, targets=None, ids=None):
        """
        Args:
            request_embs (np.array): (N, 384) SBERT embeddings of request text.
            history_seqs (np.array): (N, Max_Len, 384) SBERT embeddings of subreddit history.
            metadata (np.array): (N, Meta_Dim) Scaled metadata features.
            targets (np.array, optional): (N,) Binary targets.
            ids (np.array, optional): (N,) Request IDs.
        """
        self.request_embs = torch.FloatTensor(request_embs)
        self.history_seqs = torch.FloatTensor(history_seqs)
        self.metadata = torch.FloatTensor(metadata)

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

        self.ids = ids

    def __len__(self):
        return len(self.request_embs)

    def __getitem__(self, idx):
        sample = {
            "request_emb": self.request_embs[idx],
            "history_seq": self.history_seqs[idx],
            "metadata": self.metadata[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class ResidualAttentionNet(nn.Module):
    def __init__(self, meta_dim, hidden_dim=256, embedding_dim=384, dropout_rate=0.3):
        super(ResidualAttentionNet, self).__init__()

        # Branch 1: Request Semantics
        self.req_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Branch 2: Attention Mechanism (Query=Request, Key=History)
        self.attn_query = nn.Linear(embedding_dim, hidden_dim)
        self.attn_key = nn.Linear(embedding_dim, hidden_dim)
        self.attn_value = nn.Linear(embedding_dim, hidden_dim)
        self.attn_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Branch 3: Metadata
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Fusion & Residual Interaction
        # Input to fusion is Concat(Req, Hist, Meta) -> 3 * hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU()
        )

        # Residual Block: Dense -> ReLU -> Dropout -> Dense
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output
        self.classifier = nn.Linear(hidden_dim, 1)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, request_emb, history_seq, metadata):
        # Move scale to device
        self.scale = self.scale.to(request_emb.device)

        # 1. Process Request
        h_req = self.req_proj(request_emb)  # (B, H)

        # 2. Process History (Attention)
        # Query from Request
        Q = self.attn_query(request_emb).unsqueeze(1)  # (B, 1, H)

        # Key and Value from History Sequence
        K = self.attn_key(history_seq)  # (B, L, H)
        V = self.attn_value(history_seq)  # (B, L, H)

        # Compute Scores: (B, 1, H) @ (B, H, L) -> (B, 1, L)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Mask Padding
        # history_seq is (B, L, E). If slice (B, i, :) is all zeros, it's padding.
        # We can detect padding by checking norm of embedding
        is_padding = (history_seq.abs().sum(dim=2) == 0).unsqueeze(1)  # (B, 1, L)
        scores = scores.masked_fill(is_padding, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, L)

        # Context Vector: (B, 1, L) @ (B, L, H) -> (B, 1, H)
        context = torch.bmm(attn_weights, V).squeeze(1)  # (B, H)

        h_hist = self.attn_out(context)

        # 3. Process Metadata
        h_meta = self.meta_proj(metadata)  # (B, H)

        # 4. Fusion
        combined = torch.cat([h_req, h_hist, h_meta], dim=1)  # (B, 3H)
        x = self.fusion_proj(combined)  # (B, H)

        # Residual Interaction
        # out = x + Block(x)
        res = self.residual_block(x)
        out = x + res

        # 5. Output
        logits = self.classifier(out)
        return logits


# -----------------------------------------------------------------------------
# Data Preparation Helper
# -----------------------------------------------------------------------------


def _generate_history_sequences(df, split_name, load_cached_data=True, max_len=50):
    """
    Generates padded sequences of subreddit embeddings for user history.
    """
    cache_file = os.path.join(config.WORKING_DIR, f"history_seq_{split_name}.npy")

    if load_cached_data and os.path.exists(cache_file):
        return np.load(cache_file)

    print(f"Generating history sequences for {split_name}...")

    # Initialize FeatureEngineer to access SBERT model
    fe = features.FeatureEngineer()
    model = fe._get_sbert_model()

    # 1. Identify all unique subreddits in this split
    all_subs = set()
    for sub_list in df["requester_subreddits_at_request"]:
        if isinstance(sub_list, list):
            all_subs.update(sub_list)

    unique_subs = sorted(list(all_subs))

    # 2. Encode all subreddits
    if unique_subs:
        sub_embeddings = model.encode(
            unique_subs, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}
        emb_dim = sub_embeddings.shape[1]
    else:
        sub_map = {}
        emb_dim = 384  # Default

    # 3. Build Sequences
    num_samples = len(df)
    sequences = np.zeros((num_samples, max_len, emb_dim), dtype=np.float32)

    for i, sub_list in enumerate(df["requester_subreddits_at_request"]):
        if isinstance(sub_list, list) and len(sub_list) > 0:
            # Take last 'max_len' subreddits (most recent usually at end or just truncate)
            # List order in JSON is usually preserved. Assuming input list is chronological or relevant.
            # We truncate to fit max_len.
            current_subs = sub_list[:max_len]

            for j, sub in enumerate(current_subs):
                if sub in sub_map:
                    sequences[i, j, :] = sub_map[sub]

    # Save to cache
    np.save(cache_file, sequences)
    return sequences


def get_mlp_data(load_cached_data=True):
    """
    Loads data, generates features, scales metadata, and returns PyTorch Datasets.
    """
    # 1. Load Dataframes
    train_df, val_df, test_df = data_loader.load_datasets(
        load_cached_data=load_cached_data
    )

    # 2. Feature Engineer
    fe = features.FeatureEngineer()

    # A. Request Embeddings
    print("Loading Request Embeddings...")
    req_train, _ = fe.compute_sbert_embeddings(train_df, "train", load_cached_data)
    req_val, _ = fe.compute_sbert_embeddings(val_df, "val", load_cached_data)
    req_test, _ = fe.compute_sbert_embeddings(test_df, "test", load_cached_data)

    # B. History Sequences
    print("Loading History Sequences...")
    hist_train = _generate_history_sequences(train_df, "train", load_cached_data)
    hist_val = _generate_history_sequences(val_df, "val", load_cached_data)
    hist_test = _generate_history_sequences(test_df, "test", load_cached_data)

    # C. Metadata (Arcsinh)
    print("Loading Metadata...")
    meta_train_df = fe.generate_metadata_features(train_df, "train", load_cached_data)
    meta_val_df = fe.generate_metadata_features(val_df, "val", load_cached_data)
    meta_test_df = fe.generate_metadata_features(test_df, "test", load_cached_data)

    # Select numeric columns (including arcsinh ones)
    numeric_cols = meta_train_df.select_dtypes(include=[np.number]).columns.tolist()

    # Impute (Median)
    imputer = SimpleImputer(strategy="median")
    meta_train_raw = imputer.fit_transform(meta_train_df[numeric_cols])
    meta_val_raw = imputer.transform(meta_val_df[numeric_cols])
    meta_test_raw = imputer.transform(meta_test_df[numeric_cols])

    # Scale (StandardScaler)
    scaler = StandardScaler()
    meta_train = scaler.fit_transform(meta_train_raw)
    meta_val = scaler.transform(meta_val_raw)
    meta_test = scaler.transform(meta_test_raw)

    # 3. Targets and IDs
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values
    test_ids = test_df["request_id"].values

    # 4. Create Datasets
    train_dataset = PizzaDataset(req_train, hist_train, meta_train, y_train)
    val_dataset = PizzaDataset(req_val, hist_val, meta_val, y_val)
    test_dataset = PizzaDataset(req_test, hist_test, meta_test, ids=test_ids)

    return train_dataset, val_dataset, test_dataset, meta_train.shape[1]


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------


def train_mlp_stream(load_cached_data=True):
    """
    Trains the Residual Attention MLP.
    Returns: (model, val_probs, test_probs)
    """
    # Set seeds
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_STATE)

    # 1. Get Data
    train_dataset, val_dataset, test_dataset, meta_dim = get_mlp_data(load_cached_data)

    train_loader = DataLoader(
        train_dataset, batch_size=config.MLP_PARAMS["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.MLP_PARAMS["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.MLP_PARAMS["batch_size"], shuffle=False
    )

    # 2. Initialize Model
    model = ResidualAttentionNet(
        meta_dim=meta_dim,
        hidden_dim=config.MLP_PARAMS["hidden_dim"],
        embedding_dim=config.MLP_PARAMS["embedding_dim"],
        dropout_rate=config.MLP_PARAMS["dropout_rate"],
    ).to(config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.MLP_PARAMS["learning_rate"],
        weight_decay=config.MLP_PARAMS["weight_decay"],
    )

    # 3. Training Loop
    print("Starting MLP Training...")
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.MLP_PARAMS["epochs"]):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            req = batch["request_emb"].to(config.DEVICE)
            hist = batch["history_seq"].to(config.DEVICE)
            meta = batch["metadata"].to(config.DEVICE)
            target = batch["target"].to(config.DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(req, hist, meta)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * req.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                req = batch["request_emb"].to(config.DEVICE)
                hist = batch["history_seq"].to(config.DEVICE)
                meta = batch["metadata"].to(config.DEVICE)
                target = batch["target"]

                logits = model(req, hist, meta)
                probs = torch.sigmoid(logits).cpu().numpy()

                val_preds.extend(probs)
                val_targets.extend(target.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{config.MLP_PARAMS['epochs']} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc}"
        )

        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.MLP_PARAMS["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 4. Final Predictions
    model.eval()

    # Validation Probs
    final_val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            req = batch["request_emb"].to(config.DEVICE)
            hist = batch["history_seq"].to(config.DEVICE)
            meta = batch["metadata"].to(config.DEVICE)
            logits = model(req, hist, meta)
            final_val_probs.extend(torch.sigmoid(logits).cpu().numpy())

    # Test Probs
    final_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            req = batch["request_emb"].to(config.DEVICE)
            hist = batch["history_seq"].to(config.DEVICE)
            meta = batch["metadata"].to(config.DEVICE)
            logits = model(req, hist, meta)
            final_test_probs.extend(torch.sigmoid(logits).cpu().numpy())

    return (
        model,
        np.array(final_val_probs).flatten(),
        np.array(final_test_probs).flatten(),
    )
