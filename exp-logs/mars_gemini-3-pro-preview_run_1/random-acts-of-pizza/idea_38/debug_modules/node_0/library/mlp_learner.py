import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import copy

from library.config import Config
from library.utils import set_seed


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Query MLP.
    Handles multi-modal inputs: SBERT embeddings (Title, Body, History),
    History Masks, Metadata, and Prototype Scores.
    """

    def __init__(self, components, labels=None):
        """
        Args:
            components (dict): Dictionary containing feature arrays.
                Expected keys: 'title_emb', 'body_emb', 'hist_emb', 'hist_mask',
                               'metadata', 'prototypes'.
            labels (array-like, optional): Target labels.
        """
        self.title_emb = components["title_emb"]
        self.body_emb = components["body_emb"]
        self.hist_emb = components["hist_emb"]
        self.hist_mask = components["hist_mask"]
        self.metadata = components["metadata"]
        self.prototypes = components["prototypes"]
        self.labels = labels

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        # Convert to torch tensors
        item = {
            "title_emb": torch.tensor(self.title_emb[idx], dtype=torch.float32),
            "body_emb": torch.tensor(self.body_emb[idx], dtype=torch.float32),
            "hist_emb": torch.tensor(self.hist_emb[idx], dtype=torch.float32),
            "hist_mask": torch.tensor(
                self.hist_mask[idx], dtype=torch.bool
            ),  # Boolean mask
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float32),
            "prototypes": torch.tensor(self.prototypes[idx], dtype=torch.float32),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


class DualQueryNetwork(nn.Module):
    """
    Neural Network Architecture with Dual-Query Attention and Gated Fusion.

    Structure:
    1. Title & Body Branches: Raw SBERT embeddings.
    2. History Branch: Dual-Query Attention (Title->History, Body->History).
    3. Alignment: Concatenates scalar similarities between queries and contexts.
    4. Metadata Branch: Processes metadata + prototypes to generate a Credibility Gate.
    5. Fusion: Semantic vector modulated by Credibility Gate.
    """

    def __init__(
        self, embedding_dim, metadata_dim, hidden_dim, dropout_emb, dropout_dense
    ):
        super(DualQueryNetwork, self).__init__()

        self.dropout_emb = nn.Dropout(dropout_emb)
        self.dropout_dense = nn.Dropout(dropout_dense)

        # Metadata processing (Metadata + 4 Prototype scores)
        # Projects to the size of the concatenated semantic vector for gating
        # Semantic Vector Size:
        #   Title(D) + Body(D) + Context_Title(D) + Context_Body(D) + Align_Title(1) + Align_Body(1)
        #   = 4*D + 2
        self.semantic_dim = 4 * embedding_dim + 2

        self.metadata_gate = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim, self.semantic_dim),
            nn.Sigmoid(),  # Gate output 0-1
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_dim, 1),
        )

    def compute_attention(self, query, key, value, mask):
        """
        Computes Dot-Product Attention with explicit masking.
        Args:
            query: (B, D)
            key: (B, T, D)
            value: (B, T, D)
            mask: (B, T) Boolean mask (True where data exists)
        Returns:
            context: (B, D)
            alignment_score: (B, 1)
        """
        # Expand query to (B, 1, D)
        query_unsqueezed = query.unsqueeze(1)

        # Scores: (B, 1, D) @ (B, D, T) -> (B, 1, T) -> (B, T)
        scores = torch.bmm(query_unsqueezed, key.transpose(1, 2)).squeeze(1)

        # Apply Masking (-inf to padding)
        # Mask is True for valid tokens, False for padding.
        # We want to mask where mask is False.
        scores = scores.masked_fill(~mask, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=1)  # (B, T)

        # Context: (B, 1, T) @ (B, T, D) -> (B, 1, D) -> (B, D)
        context = torch.bmm(attn_weights.unsqueeze(1), value).squeeze(1)

        # Alignment Scalar: Dot product of Query and Context
        # (B, D) * (B, D) -> sum -> (B, 1)
        alignment = (query * context).sum(dim=1, keepdim=True)

        return context, alignment

    def forward(self, title_emb, body_emb, hist_emb, hist_mask, metadata, prototypes):
        # Apply Dropout to embeddings
        t_emb = self.dropout_emb(title_emb)
        b_emb = self.dropout_emb(body_emb)
        h_emb = self.dropout_emb(hist_emb)

        # --- Dual-Query Attention ---
        # Head A: Topic Context (Query=Title)
        ctx_title, align_title = self.compute_attention(t_emb, h_emb, h_emb, hist_mask)

        # Head B: Narrative Context (Query=Body)
        ctx_body, align_body = self.compute_attention(b_emb, h_emb, h_emb, hist_mask)

        # --- Semantic Concatenation ---
        # [Title, Body, Context_Title, Context_Body, Align_Title, Align_Body]
        semantic_vector = torch.cat(
            [t_emb, b_emb, ctx_title, ctx_body, align_title, align_body], dim=1
        )

        # --- Credibility Gate ---
        # Combine metadata and prototypes
        meta_input = torch.cat([metadata, prototypes], dim=1)
        gate = self.metadata_gate(meta_input)

        # --- Gated Fusion ---
        fused_vector = semantic_vector * gate

        # --- Classification ---
        logits = self.classifier(fused_vector)

        return logits


class MLPLearner:
    """
    Learner class for the MLP Stream.
    Handles data loading, training loop, early stopping, and inference.
    """

    def __init__(self, cache_dir=Config.WORKING_DIR):
        self.cache_dir = cache_dir
        self.params = Config.MLP_PARAMS
        self.device = torch.device(
            self.params["device"] if torch.cuda.is_available() else "cpu"
        )
        self.model = None

    def train(self, train_components, y_train, val_components, y_val):
        """
        Trains the DualQueryNetwork.
        """
        set_seed(Config.RANDOM_SEED)

        # Prepare Datasets and Loaders
        train_dataset = PizzaDataset(train_components, y_train)
        val_dataset = PizzaDataset(val_components, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        # Determine input dimensions
        embedding_dim = train_components["title_emb"].shape[1]
        # Metadata dim = metadata cols + 4 prototype scores
        metadata_dim = (
            train_components["metadata"].shape[1]
            + train_components["prototypes"].shape[1]
        )

        # Initialize Model
        self.model = DualQueryNetwork(
            embedding_dim=embedding_dim,
            metadata_dim=metadata_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout_emb=self.params["dropout_emb"],
            dropout_dense=self.params["dropout_dense"],
        ).to(self.device)

        # Optimizer and Loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping State
        best_val_auc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(self.params["epochs"]):
            # --- Training Phase ---
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                # Move to device
                t_emb = batch["title_emb"].to(self.device)
                b_emb = batch["body_emb"].to(self.device)
                h_emb = batch["hist_emb"].to(self.device)
                h_mask = batch["hist_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)
                proto = batch["prototypes"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                outputs = self.model(t_emb, b_emb, h_emb, h_mask, meta, proto)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * t_emb.size(0)

            epoch_loss = running_loss / len(train_dataset)

            # --- Validation Phase ---
            self.model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    t_emb = batch["title_emb"].to(self.device)
                    b_emb = batch["body_emb"].to(self.device)
                    h_emb = batch["hist_emb"].to(self.device)
                    h_mask = batch["hist_mask"].to(self.device)
                    meta = batch["metadata"].to(self.device)
                    proto = batch["prototypes"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = self.model(t_emb, b_emb, h_emb, h_mask, meta, proto)
                    probs = torch.sigmoid(outputs).squeeze(1)

                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_auc = roc_auc_score(val_targets, val_preds)

            # Early Stopping Check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.params["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        print(f"Best MLP Validation AUC: {best_val_auc}")

        return self.model

    def predict(self, test_components):
        """
        Generates predictions for the test set.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        test_dataset = PizzaDataset(test_components, labels=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                t_emb = batch["title_emb"].to(self.device)
                b_emb = batch["body_emb"].to(self.device)
                h_emb = batch["hist_emb"].to(self.device)
                h_mask = batch["hist_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)
                proto = batch["prototypes"].to(self.device)

                outputs = self.model(t_emb, b_emb, h_emb, h_mask, meta, proto)
                probs = torch.sigmoid(outputs).squeeze(1)

                all_preds.extend(probs.cpu().numpy())

        return np.array(all_preds)
