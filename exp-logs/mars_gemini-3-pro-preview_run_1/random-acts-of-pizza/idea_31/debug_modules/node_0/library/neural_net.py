import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from library import config, data_loader


# Set fixed seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(config.RANDOM_STATE)

# =============================================================================
# DATASET CLASS
# =============================================================================


class PizzaDataset(Dataset):
    def __init__(
        self,
        title_emb,
        body_emb,
        history_emb,
        history_mask,
        meta,
        alignment,
        labels=None,
    ):
        self.title_emb = torch.FloatTensor(title_emb)
        self.body_emb = torch.FloatTensor(body_emb)
        self.history_emb = torch.FloatTensor(history_emb)
        self.history_mask = torch.BoolTensor(history_mask)
        self.meta = torch.FloatTensor(meta)
        self.alignment = torch.FloatTensor(alignment)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        item = {
            "title": self.title_emb[idx],
            "body": self.body_emb[idx],
            "history": self.history_emb[idx],
            "mask": self.history_mask[idx],
            "meta": self.meta[idx],
            "alignment": self.alignment[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class DualQueryMLP(nn.Module):
    def __init__(self, embedding_dim, meta_dim, hidden_dim, dropout_prob):
        super(DualQueryMLP, self).__init__()

        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout_prob)

        # 1. Attention Mechanisms
        # We use dot-product attention, so no learned weights here,
        # but we might project queries/keys if needed. Keeping it simple (raw SBERT) as per design.

        # 2. Metadata Gate
        # Projects metadata to the size of the semantic vector (Title + Body + CtxA + CtxB)
        self.semantic_dim = embedding_dim * 4
        self.meta_gate_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, self.semantic_dim),
            nn.Sigmoid(),
        )

        # 3. Final Classifier
        # Input: Gated Semantic (4*Emb) + Alignment (2) + Metadata (Meta_Dim)
        self.classifier_input_dim = self.semantic_dim + 2 + meta_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1),
        )

    def attention(self, query, key, value, mask):
        """
        query: (B, Emb)
        key: (B, Seq, Emb)
        value: (B, Seq, Emb)
        mask: (B, Seq) - True for valid tokens, False for padding
        """
        # (B, 1, Emb) @ (B, Emb, Seq) -> (B, 1, Seq)
        scores = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))
        scores = scores.squeeze(1) / np.sqrt(self.embedding_dim)

        # Apply mask
        scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=1)  # (B, Seq)

        # (B, 1, Seq) @ (B, Seq, Emb) -> (B, 1, Emb)
        context = torch.bmm(weights.unsqueeze(1), value).squeeze(1)
        return context

    def forward(self, title, body, history, mask, meta, alignment):
        # Branch 3: Dual-Query History Attention
        # Topic Context (Query=Title)
        ctx_topic = self.attention(title, history, history, mask)

        # Narrative Context (Query=Body)
        ctx_narrative = self.attention(body, history, history, mask)

        # Construct Semantic Vector
        semantic_raw = torch.cat([title, body, ctx_topic, ctx_narrative], dim=1)

        # Branch 4: Gated Fusion
        gate = self.meta_gate_mlp(meta)
        semantic_gated = semantic_raw * gate

        # Alignment Injection & Final Concatenation
        # Combine: Gated Semantics + Alignment Scalars + Raw Metadata
        combined = torch.cat([semantic_gated, alignment, meta], dim=1)

        # Classifier
        logits = self.classifier(combined)
        return logits


# =============================================================================
# DATA PROCESSING
# =============================================================================


def get_sbert_embeddings(texts, model, batch_size=64):
    """Helper to get embeddings in batches."""
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    return embeddings


def process_history(subreddits_list, model, max_len=50):
    """
    Encodes list of subreddits into (N, Max_Len, Emb_Dim) and generates masks.
    Handles empty histories by adding a placeholder.
    """
    # Flatten unique subreddits to encode efficiently
    unique_subs = set()
    for subs in subreddits_list:
        if not subs:
            unique_subs.add("[EMPTY]")
        else:
            unique_subs.update(subs[:max_len])

    sorted_subs = sorted(list(unique_subs))
    sub_to_idx = {sub: i for i, sub in enumerate(sorted_subs)}

    # Encode all unique subreddits
    sub_embeddings = model.encode(
        sorted_subs, batch_size=128, show_progress_bar=False, convert_to_numpy=True
    )
    emb_dim = sub_embeddings.shape[1]

    # Construct tensors
    n_samples = len(subreddits_list)
    history_emb = np.zeros((n_samples, max_len, emb_dim), dtype=np.float32)
    history_mask = np.zeros((n_samples, max_len), dtype=bool)

    for i, subs in enumerate(subreddits_list):
        if not subs:
            idx = sub_to_idx["[EMPTY]"]
            history_emb[i, 0, :] = sub_embeddings[idx]
            history_mask[i, 0] = True  # Valid token
        else:
            current_subs = subs[:max_len]
            for j, sub in enumerate(current_subs):
                idx = sub_to_idx[sub]
                history_emb[i, j, :] = sub_embeddings[idx]
                history_mask[i, j] = True

    return history_emb, history_mask


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches data for the Neural Network.
    Returns dictionaries containing numpy arrays for train, val, and test.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        "train": os.path.join(cache_dir, "nn_data_train.npz"),
        "val": os.path.join(cache_dir, "nn_data_val.npz"),
        "test": os.path.join(cache_dir, "nn_data_test.npz"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print("Loading neural network data from cache...")
        data = {}
        for split, path in cache_files.items():
            loaded = np.load(path)
            data[split] = {k: loaded[k] for k in loaded.files}
        return data["train"], data["val"], data["test"]

    print("Processing neural network data from scratch...")

    # Load Raw Data
    df_train, df_val = data_loader.get_stratified_split(load_cached_data=True)
    df_test = data_loader.load_dataset("test", load_cached_data=True)

    # Initialize SBERT
    sbert = SentenceTransformer(config.SBERT_MODEL)

    # Helper for processing a single dataframe
    def process_split(df, is_train=False, scaler=None):
        # 1. Text Embeddings
        print(f"Embedding titles/bodies for {len(df)} samples...")
        titles = df["request_title"].fillna("").astype(str).tolist()
        bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()

        title_emb = get_sbert_embeddings(titles, sbert)
        body_emb = get_sbert_embeddings(bodies, sbert)

        # 2. History Embeddings
        print(f"Embedding histories for {len(df)} samples...")
        histories = df["requester_subreddits_at_request"].tolist()
        history_emb, history_mask = process_history(histories, sbert)

        # 3. Alignment Scalars
        # Compute mean history vector for each user
        # Sum valid embeddings and divide by count
        # (N, L, D) * (N, L, 1) -> sum -> (N, D)
        mask_expanded = history_mask[:, :, np.newaxis]
        valid_sum = np.sum(history_emb * mask_expanded, axis=1)
        valid_count = np.sum(mask_expanded, axis=1)
        valid_count = np.maximum(valid_count, 1)  # Avoid div by zero
        mean_history = valid_sum / valid_count

        # Cosine Similarity
        # (N, D) . (N, D) / (norm * norm)
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a, axis=1)
            norm_b = np.linalg.norm(b, axis=1)
            dot = np.sum(a * b, axis=1)
            sim = dot / (np.maximum(norm_a * norm_b, 1e-9))
            return sim[:, np.newaxis]  # (N, 1)

        topic_sim = cosine_sim(title_emb, mean_history)
        narrative_sim = cosine_sim(body_emb, mean_history)
        alignment = np.hstack([topic_sim, narrative_sim])

        # 4. Metadata
        # Select numeric columns
        exclude = ["requester_received_pizza", "request_id", "source_file"]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in exclude]

        meta = df[num_cols].fillna(0).values

        # Arcsinh Transform
        if config.USE_ARCSINH_TRANSFORM:
            meta = np.arcsinh(meta)

        # Scaling
        if is_train:
            scaler = StandardScaler()
            meta = scaler.fit_transform(meta)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test sets")
            meta = scaler.transform(meta)

        # Labels
        labels = None
        if "requester_received_pizza" in df.columns:
            labels = df["requester_received_pizza"].astype(int).values

        return {
            "title_emb": title_emb,
            "body_emb": body_emb,
            "history_emb": history_emb,
            "history_mask": history_mask,
            "meta": meta,
            "alignment": alignment,
            "labels": labels,
        }, scaler

    # Process Train
    train_data, scaler = process_split(df_train, is_train=True)

    # Process Val
    val_data, _ = process_split(df_val, is_train=False, scaler=scaler)

    # Process Test
    test_data, _ = process_split(df_test, is_train=False, scaler=scaler)

    # Cache Data
    print("Caching processed data...")
    np.savez(cache_files["train"], **train_data)
    np.savez(cache_files["val"], **val_data)
    np.savez(cache_files["test"], **test_data)

    return train_data, val_data, test_data


# =============================================================================
# TRAINING & INFERENCE
# =============================================================================


def train_model(train_data, val_data, device="cpu"):
    """
    Trains the DualQueryMLP model with early stopping.
    """
    # Hyperparameters
    BATCH_SIZE = config.MLP_BATCH_SIZE
    EPOCHS = config.MLP_EPOCHS
    LR = config.MLP_LR
    WD = config.MLP_WEIGHT_DECAY
    PATIENCE = config.MLP_PATIENCE

    # Datasets & Loaders
    train_dataset = PizzaDataset(
        train_data["title_emb"],
        train_data["body_emb"],
        train_data["history_emb"],
        train_data["history_mask"],
        train_data["meta"],
        train_data["alignment"],
        train_data["labels"],
    )
    val_dataset = PizzaDataset(
        val_data["title_emb"],
        val_data["body_emb"],
        val_data["history_emb"],
        val_data["history_mask"],
        val_data["meta"],
        val_data["alignment"],
        val_data["labels"],
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    emb_dim = train_data["title_emb"].shape[1]
    meta_dim = train_data["meta"].shape[1]

    model = DualQueryMLP(
        embedding_dim=emb_dim,
        meta_dim=meta_dim,
        hidden_dim=config.MLP_HIDDEN_DIM,
        dropout_prob=config.MLP_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            # Move to device
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            alignment = batch["alignment"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(title, body, history, mask, meta, alignment)
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
                title = batch["title"].to(device)
                body = batch["body"].to(device)
                history = batch["history"].to(device)
                mask = batch["mask"].to(device)
                meta = batch["meta"].to(device)
                alignment = batch["alignment"].to(device)
                labels = batch["label"].to(device)

                logits = model(title, body, history, mask, meta, alignment)
                probs = torch.sigmoid(logits).squeeze(1)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_auc


def predict(model, test_data, device="cpu"):
    """
    Generates predictions for the test set.
    """
    BATCH_SIZE = config.MLP_BATCH_SIZE

    test_dataset = PizzaDataset(
        test_data["title_emb"],
        test_data["body_emb"],
        test_data["history_emb"],
        test_data["history_mask"],
        test_data["meta"],
        test_data["alignment"],
        labels=None,
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            title = batch["title"].to(device)
            body = batch["body"].to(device)
            history = batch["history"].to(device)
            mask = batch["mask"].to(device)
            meta = batch["meta"].to(device)
            alignment = batch["alignment"].to(device)

            logits = model(title, body, history, mask, meta, alignment)
            probs = torch.sigmoid(logits).squeeze(1)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)
