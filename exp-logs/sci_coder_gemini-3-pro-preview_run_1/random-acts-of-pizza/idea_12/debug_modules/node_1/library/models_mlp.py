import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import copy
import random
import os

from library.config import Config


# Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.RANDOM_SEED)


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Hierarchical Multi-View Attention model.
    """

    def __init__(self, data_dict, is_train=True):
        self.mlp_meta = torch.FloatTensor(data_dict["mlp_meta"])
        self.mlp_title_emb = torch.FloatTensor(data_dict["mlp_title_emb"])
        self.mlp_body_emb = torch.FloatTensor(data_dict["mlp_body_emb"])
        self.mlp_hist_emb = torch.FloatTensor(data_dict["mlp_hist_emb"])

        if is_train and "y" in data_dict and len(data_dict["y"]) > 0:
            self.y = torch.FloatTensor(data_dict["y"])
        else:
            self.y = None

    def __len__(self):
        return len(self.mlp_meta)

    def __getitem__(self, idx):
        sample = {
            "meta": self.mlp_meta[idx],
            "title": self.mlp_title_emb[idx],
            "body": self.mlp_body_emb[idx],
            "hist": self.mlp_hist_emb[idx],
        }
        if self.y is not None:
            return sample, self.y[idx]
        return sample


class HistoryAttention(nn.Module):
    """
    Dot-Product Attention Module.
    Query: Request Title Embedding (Batch, EmbDim)
    Keys/Values: User History Embeddings (Batch, SeqLen, EmbDim)
    """

    def __init__(self, embedding_dim):
        super(HistoryAttention, self).__init__()
        self.scale = 1.0 / (embedding_dim**0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys):
        # query: (B, D) -> (B, 1, D)
        query = query.unsqueeze(1)

        # keys: (B, L, D)
        # keys_t: (B, D, L)
        keys_t = keys.transpose(1, 2)

        # scores: (B, 1, D) @ (B, D, L) -> (B, 1, L)
        scores = torch.bmm(query, keys_t) * self.scale

        # Masking zero-padded entries could be done here if lengths were variable,
        # but SBERT embeddings are non-zero and fixed length simplifies this for now.
        # We rely on the model learning to ignore padding if it's consistently zero.

        weights = self.softmax(scores)

        # context: (B, 1, L) @ (B, L, D) -> (B, 1, D)
        context = torch.bmm(weights, keys)

        # Remove singleton dim -> (B, D)
        return context.squeeze(1)


class HierarchicalAttentionNetwork(nn.Module):
    """
    Neural Network with:
    1. Title Branch
    2. Body Branch
    3. History Attention Branch (Query=Title)
    4. Metadata Branch
    5. Gated Fusion
    """

    def __init__(
        self, meta_dim, embedding_dim=384, hidden_dims=[256, 128], dropout=0.3
    ):
        super(HierarchicalAttentionNetwork, self).__init__()

        # --- Semantic Branches ---
        # We assume SBERT embeddings are already high quality, so we just apply dropout
        # before fusion or light projection if needed. Here we use them directly.
        self.embedding_dropout = nn.Dropout(dropout)

        # Attention Mechanism for History
        self.history_attn = HistoryAttention(embedding_dim)

        # --- Metadata Branch ---
        # Projects metadata to generate the Gate
        # The Gate must match the dimension of concatenated semantic features
        # Semantic dim = Title(D) + Body(D) + HistContext(D) = 3 * D
        self.semantic_concat_dim = embedding_dim * 3

        self.meta_gate_net = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.semantic_concat_dim),
            nn.Sigmoid(),  # Output is a gate between 0 and 1
        )

        # --- Classification Head ---
        input_dim = self.semantic_concat_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, 1))  # Binary classification
        self.classifier = nn.Sequential(*layers)

    def forward(self, meta, title, body, hist):
        # 1. Semantic Features
        title = self.embedding_dropout(title)
        body = self.embedding_dropout(body)

        # Calculate Context from History using Title as Query
        hist_context = self.history_attn(title, hist)
        hist_context = self.embedding_dropout(hist_context)

        # Concatenate Semantic Views
        # (B, 3 * D)
        h_sem = torch.cat([title, body, hist_context], dim=1)

        # 2. Metadata Gating
        # (B, 3 * D)
        gate = self.meta_gate_net(meta)

        # 3. Gated Fusion
        # Element-wise multiplication
        h_fused = h_sem * gate

        # 4. Classification
        logits = self.classifier(h_fused)
        return logits


class StreamBMLP:
    """
    Stream B: Hierarchical Decoupled-Attention MLP Wrapper.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = None
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, train_data, val_data=None):
        """
        Trains the MLP model.
        """
        # Prepare Datasets
        train_dataset = PizzaDataset(train_data, is_train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.MLP_BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing overhead/issues in this env
        )

        val_loader = None
        if val_data:
            val_dataset = PizzaDataset(val_data, is_train=True)
            val_loader = DataLoader(
                val_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
            )

        # Initialize Model
        # Determine metadata dimension from data
        meta_dim = train_data["mlp_meta"].shape[1]

        self.model = HierarchicalAttentionNetwork(
            meta_dim=meta_dim,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dims=Config.MLP_HIDDEN_DIMS,
            dropout=Config.MLP_DROPOUT,
        ).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )

        # Training Loop with Early Stopping
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None

        print(f"Stream B (MLP): Starting training on {self.device}...")

        for epoch in range(Config.MLP_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                inputs, targets = batch

                # Move to device
                meta = inputs["meta"].to(self.device)
                title = inputs["title"].to(self.device)
                body = inputs["body"].to(self.device)
                hist = inputs["hist"].to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = self.model(meta, title, body, hist)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * targets.size(0)

            train_loss /= len(train_dataset)

            # Validation
            val_auc = 0.0
            if val_loader:
                val_auc = self._evaluate_loader(val_loader)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.MLP_PATIENCE:
                    print(
                        f"Stream B (MLP): Early stopping at epoch {epoch+1}. Best Val AUC: {best_val_auc}"
                    )
                    break

            # Optional: Print progress every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc}"
                )

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print("Stream B (MLP): Training complete.")

    def _evaluate_loader(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch

                meta = inputs["meta"].to(self.device)
                title = inputs["title"].to(self.device)
                body = inputs["body"].to(self.device)
                hist = inputs["hist"].to(self.device)

                outputs = self.model(meta, title, body, hist)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                all_preds.extend(probs)
                all_targets.extend(targets.numpy())

        if len(all_targets) == 0:
            return 0.0

        try:
            return roc_auc_score(all_targets, all_preds)
        except ValueError:
            return 0.0

    def predict_proba(self, data):
        """
        Generates probability predictions for the positive class.
        """
        dataset = PizzaDataset(data, is_train=False)
        loader = DataLoader(dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch  # No targets in test mode usually, but Dataset might return them if present
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]

                meta = inputs["meta"].to(self.device)
                title = inputs["title"].to(self.device)
                body = inputs["body"].to(self.device)
                hist = inputs["hist"].to(self.device)

                outputs = self.model(meta, title, body, hist)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(probs)

        return np.array(all_preds)

    def evaluate(self, val_data):
        """
        Evaluates the model on validation data using ROC AUC.
        """
        val_dataset = PizzaDataset(val_data, is_train=True)
        val_loader = DataLoader(
            val_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=False
        )

        auc = self._evaluate_loader(val_loader)
        print(f"Stream B (MLP) Validation ROC AUC: {auc}")
        return auc
