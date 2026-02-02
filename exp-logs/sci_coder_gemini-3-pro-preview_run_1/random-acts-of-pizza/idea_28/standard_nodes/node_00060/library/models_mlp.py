import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config

# Set deterministic seeds
torch.manual_seed(Config.RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.RANDOM_STATE)
np.random.seed(Config.RANDOM_STATE)


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Multi-Modal Pizza Request Data.
    Handles Title (SBERT), Body (SBERT), History (Sequence SBERT), and Metadata (Dense).
    """

    def __init__(self, title_emb, body_emb, history_emb, meta_features, labels=None):
        self.title_emb = torch.tensor(title_emb, dtype=torch.float32)
        self.body_emb = torch.tensor(body_emb, dtype=torch.float32)
        self.history_emb = torch.tensor(history_emb, dtype=torch.float32)
        self.meta_features = torch.tensor(meta_features, dtype=torch.float32)

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        item = {
            "title": self.title_emb[idx],
            "body": self.body_emb[idx],
            "history": self.history_emb[idx],
            "meta": self.meta_features[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class DualQueryAttention(nn.Module):
    """
    Computes attention between a Query (Title or Body) and Keys (User History).
    Applies additive masking to ignore padding (zero-vectors) in history.
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, history):
        """
        Args:
            query: (Batch, EmbedDim)
            history: (Batch, SeqLen, EmbedDim)
        Returns:
            context: (Batch, EmbedDim)
        """
        # Reshape Query: (B, 1, D)
        query = query.unsqueeze(1)

        # Calculate Scores: (B, 1, D) @ (B, D, L) -> (B, 1, L)
        # history.transpose(1, 2) makes it (B, D, L)
        scores = torch.bmm(query, history.transpose(1, 2)) * self.scale

        # Create Mask for Padding
        # Assume padding vectors in history are all zeros.
        # Sum absolute values across embedding dim: (B, L)
        # If sum is close to 0, it's padding.
        is_padding = history.abs().sum(dim=2) < 1e-9

        # Expand mask to match scores shape: (B, 1, L)
        mask = is_padding.unsqueeze(1)

        # Apply Mask (-inf)
        scores = scores.masked_fill(mask, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted Sum: (B, 1, L) @ (B, L, D) -> (B, 1, D)
        context = torch.bmm(attn_weights, history)

        # Squeeze back to (B, D)
        return context.squeeze(1)


class GatedFusion(nn.Module):
    """
    Fuses semantic features with a gate derived from metadata.
    """

    def __init__(self, semantic_dim, meta_dim, hidden_dim):
        super(GatedFusion, self).__init__()

        # Gate Network: Meta -> Gate
        self.gate_net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
            nn.Sigmoid(),
        )

    def forward(self, semantic_features, meta_features):
        """
        Args:
            semantic_features: Concatenated embeddings (Batch, SemanticDim)
            meta_features: Metadata (Batch, MetaDim)
        Returns:
            gated_features: (Batch, SemanticDim)
        """
        gate = self.gate_net(meta_features)
        return semantic_features * gate


class PizzaNet(nn.Module):
    """
    Main Architecture:
    1. Branch 1 & 2: Raw Title/Body Embeddings.
    2. Branch 3: Dual-Query Attention (Title->History, Body->History).
    3. Branch 4: Metadata -> Credibility Gate.
    4. Fusion: Gated modulation of concatenated text features.
    5. Head: Classification MLP.
    """

    def __init__(self, embed_dim, meta_dim, hidden_dim, dropout_prob):
        super(PizzaNet, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)

        # Attention Modules
        self.attn_topic = DualQueryAttention(embed_dim)
        self.attn_narrative = DualQueryAttention(embed_dim)

        # Semantic Dimension: Title + Body + Context_Topic + Context_Narrative
        self.semantic_dim = embed_dim * 4

        # Fusion
        self.fusion = GatedFusion(self.semantic_dim, meta_dim, hidden_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),  # Logits
        )

    def forward(self, title, body, history, meta):
        # Apply dropout to embeddings early
        title = self.dropout(title)
        body = self.dropout(body)
        history = self.dropout(history)

        # Branch 3: Attention
        context_topic = self.attn_topic(title, history)
        context_narrative = self.attn_narrative(body, history)

        # Concatenate Semantics
        semantic_vec = torch.cat([title, body, context_topic, context_narrative], dim=1)

        # Branch 4 & Fusion: Gated Modulation
        fused_vec = self.fusion(semantic_vec, meta)

        # Classification
        logits = self.classifier(fused_vec)
        return logits


class MLPModelWrapper:
    """
    Wrapper for training and inference of the Neural Network stream.
    """

    def __init__(self):
        self.params = Config.MLP_PARAMS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _get_dataloader(self, mlp_data, split, labels=None, shuffle=False):
        """Helper to create DataLoader from data dictionary."""
        prefix = split  # e.g., 'train', 'val', 'test'

        title = mlp_data[f"{prefix}_title"]
        body = mlp_data[f"{prefix}_body"]
        history = mlp_data[f"{prefix}_history"]
        meta = mlp_data[f"{prefix}_meta"]

        y = labels[f"y_{split}"] if labels and f"y_{split}" in labels else None

        dataset = PizzaDataset(title, body, history, meta, y)
        return DataLoader(
            dataset, batch_size=self.params["batch_size"], shuffle=shuffle
        )

    def train(self, mlp_data, labels):
        print(f"Initializing MLP on device: {self.device}")

        # Determine dimensions from data
        embed_dim = mlp_data["train_title"].shape[1]
        meta_dim = mlp_data["train_meta"].shape[1]

        # Initialize Model
        self.model = PizzaNet(
            embed_dim=embed_dim,
            meta_dim=meta_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout_prob=self.params["dropout"],
        ).to(self.device)

        # Optimizer & Loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.BCEWithLogitsLoss()

        # DataLoaders
        train_loader = self._get_dataloader(mlp_data, "train", labels, shuffle=True)
        val_loader = self._get_dataloader(mlp_data, "val", labels, shuffle=False)

        # Training Loop
        best_auc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting training for {self.params['epochs']} epochs...")

        for epoch in range(self.params["epochs"]):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # Move to device
                title = batch["title"].to(self.device)
                body = batch["body"].to(self.device)
                history = batch["history"].to(self.device)
                meta = batch["meta"].to(self.device)
                target = batch["label"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                outputs = self.model(title, body, history, meta)
                loss = criterion(outputs, target)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * title.size(0)

            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            val_auc, val_loss = self._evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{self.params['epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val AUC: {val_auc}"
            )  # Full precision printing

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.params["patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load best weights
        print(f"Training complete. Best Val AUC: {best_auc}")
        self.model.load_state_dict(best_model_wts)
        return best_auc

    def _evaluate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                title = batch["title"].to(self.device)
                body = batch["body"].to(self.device)
                history = batch["history"].to(self.device)
                meta = batch["meta"].to(self.device)
                target = batch["label"].to(self.device).unsqueeze(1)

                outputs = self.model(title, body, history, meta)
                loss = criterion(outputs, target)

                running_loss += loss.item() * title.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(target.cpu().numpy())

        total_loss = running_loss / len(dataloader.dataset)

        # Handle case with single class in batch (though unlikely in full val set)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5

        return auc, total_loss

    def predict(self, mlp_data):
        """
        Generates predictions for the test set.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        test_loader = self._get_dataloader(mlp_data, "test", shuffle=False)

        self.model.eval()
        all_preds = []

        print("Generating MLP Predictions...")
        with torch.no_grad():
            for batch in test_loader:
                title = batch["title"].to(self.device)
                body = batch["body"].to(self.device)
                history = batch["history"].to(self.device)
                meta = batch["meta"].to(self.device)

                outputs = self.model(title, body, history, meta)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(probs)

        return np.array(all_preds)
