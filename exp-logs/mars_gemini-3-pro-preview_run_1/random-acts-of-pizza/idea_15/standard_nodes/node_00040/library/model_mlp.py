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
    def __init__(
        self, request_embs, history_centroids, metadata, targets=None, ids=None
    ):
        """
        Args:
            request_embs (np.array): (N, 384) SBERT embeddings of request text.
            history_centroids (np.array): (N, 384) SBERT embeddings of history centroid.
            metadata (np.array): (N, Meta_Dim) Scaled metadata features.
            targets (np.array, optional): (N,) Binary targets.
            ids (np.array, optional): (N,) Request IDs.
        """
        self.request_embs = torch.FloatTensor(request_embs)
        self.history_centroids = torch.FloatTensor(history_centroids)
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
            "history_centroid": self.history_centroids[idx],
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


class GatedFusionNet(nn.Module):
    """
    Implements Context-Gated Semantic Fusion (Cite Lesson 28).
    Metadata generates a sigmoid gate to modulate semantic embeddings.
    """

    def __init__(self, meta_dim, hidden_dim=256, embedding_dim=384, dropout_rate=0.3):
        super(GatedFusionNet, self).__init__()

        # Branch 1: Semantic Content (Request + History)
        # We concatenate Request and History Centroid -> 2 * 384
        self.sem_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Branch 2: Metadata Context Gate
        # Projects metadata to same dimension as semantic features
        self.meta_gate_proj = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),  # Gate activation
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, request_emb, history_centroid, metadata):
        # 1. Semantic Branch
        sem_input = torch.cat([request_emb, history_centroid], dim=1)
        h_sem = self.sem_proj(sem_input)

        # 2. Metadata Gate
        gate = self.meta_gate_proj(metadata)

        # 3. Gated Fusion (Element-wise multiplication)
        # The gate amplifies/suppresses semantic features based on metadata context
        h_fused = h_sem * gate

        # 4. Output
        logits = self.classifier(h_fused)
        return logits


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

    # A. Request & History Embeddings (Centroids)
    # Cite Lesson 28: Averaged dense embeddings for history
    print("Loading Request & History Embeddings...")
    req_train, hist_train = fe.compute_sbert_embeddings(
        train_df, "train", load_cached_data
    )
    req_val, hist_val = fe.compute_sbert_embeddings(val_df, "val", load_cached_data)
    req_test, hist_test = fe.compute_sbert_embeddings(test_df, "test", load_cached_data)

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
    model = GatedFusionNet(
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
            hist = batch["history_centroid"].to(config.DEVICE)
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
                hist = batch["history_centroid"].to(config.DEVICE)
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
            hist = batch["history_centroid"].to(config.DEVICE)
            meta = batch["metadata"].to(config.DEVICE)
            logits = model(req, hist, meta)
            final_val_probs.extend(torch.sigmoid(logits).cpu().numpy())

    # Test Probs
    final_test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            req = batch["request_emb"].to(config.DEVICE)
            hist = batch["history_centroid"].to(config.DEVICE)
            meta = batch["metadata"].to(config.DEVICE)
            logits = model(req, hist, meta)
            final_test_probs.extend(torch.sigmoid(logits).cpu().numpy())

    return (
        model,
        np.array(final_val_probs).flatten(),
        np.array(final_test_probs).flatten(),
    )
