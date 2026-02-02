import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import random
from library import config


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ResidualAttentionNet(nn.Module):
    """
    Stream B: Residual-Attention MLP.

    Architecture:
    1. Text Branch: SBERT embeddings.
    2. History Branch: Subreddit embeddings aggregated via Dot-Product Attention
       (Query=Text, Key=History).
    3. Metadata Branch: Numerical metadata processed via MLP.
    4. Fusion: Concatenation -> Residual Block -> Classifier.
    """

    def __init__(
        self,
        vocab_size,
        sbert_dim,
        meta_dim,
        hidden_dim=256,
        dropout_prob=0.3,
        embedding_dim=64,
    ):
        super(ResidualAttentionNet, self).__init__()

        # --- Branch 1: Text ---
        # SBERT input is already dense (sbert_dim=384), we might project it or use as is.
        # We project it to match embedding_dim for attention compatibility.
        self.text_proj = nn.Linear(sbert_dim, embedding_dim)
        self.text_dropout = nn.Dropout(dropout_prob)

        # --- Branch 2: History ---
        self.history_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.history_dropout = nn.Dropout(dropout_prob)

        # --- Branch 3: Metadata ---
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, embedding_dim), nn.ReLU(), nn.Dropout(dropout_prob)
        )

        # --- Fusion & Residual Block ---
        # Concatenated dimension: Text (sbert_dim) + History Context (embedding_dim) + Meta (embedding_dim)
        # Note: We keep the original SBERT embedding in the concat for richness,
        # plus the attended history and processed meta.
        self.concat_dim = sbert_dim + embedding_dim + embedding_dim

        # Residual Block: H_final = H_joint + Dense(ReLU(Dense(H_joint)))
        # To make dimensions match for addition, the residual path must preserve shape.
        # We first project concat to hidden_dim, then apply residual on hidden_dim.
        self.pre_residual = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)
        )

        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_prob),
        )

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, text_emb, history_idx, meta_features):
        """
        Args:
            text_emb: (batch, sbert_dim)
            history_idx: (batch, seq_len)
            meta_features: (batch, meta_dim)
        """
        # 1. Text Processing
        # (batch, sbert_dim)
        t_raw = self.text_dropout(text_emb)
        # Project for attention query: (batch, embedding_dim)
        t_query = self.text_proj(t_raw)
        t_query = t_query.unsqueeze(2)  # (batch, embedding_dim, 1)

        # 2. History Attention
        # (batch, seq_len, embedding_dim)
        h_emb = self.history_embedding(history_idx)
        h_emb = self.history_dropout(h_emb)

        # Attention Scores: Query (Text) dot Key (History)
        # (batch, seq_len, embedding_dim) @ (batch, embedding_dim, 1) -> (batch, seq_len, 1)
        scores = torch.bmm(h_emb, t_query)

        # Mask padding (index 0)
        mask = (history_idx == 0).unsqueeze(2)  # (batch, seq_len, 1)
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Context Vector: Weighted Sum of History
        # (batch, embedding_dim, seq_len) @ (batch, seq_len, 1) -> (batch, embedding_dim, 1)
        context = torch.bmm(h_emb.transpose(1, 2), attn_weights).squeeze(2)

        # 3. Metadata Processing
        m_emb = self.meta_proj(meta_features)

        # 4. Fusion
        # Concatenate: Original Text + Attended History + Processed Meta
        combined = torch.cat([t_raw, context, m_emb], dim=1)

        # Project to hidden dimension
        h = self.pre_residual(combined)

        # Residual Connection
        res = self.residual_block(h)
        h_final = h + res

        # Output
        logits = self.classifier(h_final)
        return logits


class MLPTrainer:
    """
    Trainer class for the ResidualAttentionNet.
    Handles data loading, training loop, validation, early stopping, and prediction.
    """

    def __init__(self, dims, params=None):
        """
        Args:
            dims (dict): Dictionary containing dimension info (vocab_size, sbert_dim, etc.)
            params (dict): Hyperparameters. Defaults to config.MLP_PARAMS.
        """
        self.dims = dims
        self.params = params if params is not None else config.MLP_PARAMS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(config.WORKING_DIR, "mlp_model.pth")

        set_seed(config.RANDOM_STATE)

        self.model = ResidualAttentionNet(
            vocab_size=dims["vocab_size"],
            sbert_dim=dims["sbert_dim"],
            meta_dim=dims["meta_dim"],
            hidden_dim=self.params["hidden_dim"],
            dropout_prob=self.params["dropout_prob"],
        ).to(self.device)

        print(f"Initialized ResidualAttentionNet on {self.device}")

    def _prepare_loader(self, data_dict, targets=None, shuffle=False):
        """Creates a DataLoader from the data dictionary."""
        text = torch.tensor(data_dict["text"], dtype=torch.float32)
        history = torch.tensor(data_dict["history"], dtype=torch.long)
        meta = torch.tensor(data_dict["meta"], dtype=torch.float32)

        if targets is not None:
            y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(text, history, meta, y)
        else:
            dataset = TensorDataset(text, history, meta)

        return DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def train(self, train_data, train_targets, val_data, val_targets):
        """
        Executes the training loop with Early Stopping.
        """
        train_loader = self._prepare_loader(train_data, train_targets, shuffle=True)
        val_loader = self._prepare_loader(val_data, val_targets, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        criterion = nn.BCEWithLogitsLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            verbose=False,
        )

        best_val_auc = 0.0
        patience_counter = 0

        print(f"Starting training for {self.params['epochs']} epochs...")

        for epoch in range(self.params["epochs"]):
            # --- Training ---
            self.model.train()
            train_loss_sum = 0

            for batch in train_loader:
                text_b, hist_b, meta_b, y_b = [t.to(self.device) for t in batch]

                optimizer.zero_grad()
                logits = self.model(text_b, hist_b, meta_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * text_b.size(0)

            avg_train_loss = train_loss_sum / len(train_loader.dataset)

            # --- Validation ---
            self.model.eval()
            val_preds = []
            val_true = []
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    text_b, hist_b, meta_b, y_b = [t.to(self.device) for t in batch]
                    logits = self.model(text_b, hist_b, meta_b)
                    loss = criterion(logits, y_b)

                    val_loss_sum += loss.item() * text_b.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_true.extend(y_b.cpu().numpy())

            val_auc = roc_auc_score(val_true, val_preds)
            scheduler.step(val_auc)

            # --- Logging & Early Stopping ---
            # Using full precision for logging as requested
            print(
                f"Epoch {epoch+1}/{self.params['epochs']} - Loss: {avg_train_loss:.6f} - Val AUC: {val_auc}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1
                if patience_counter >= self.params["patience"]:
                    print("Early stopping triggered.")
                    break

        print(f"Best MLP Validation AUC: {best_val_auc}")

    def predict(self, test_data):
        """
        Generates predictions for the test set.
        Loads the best saved model first.
        """
        self.load()
        self.model.eval()

        test_loader = self._prepare_loader(test_data, shuffle=False)
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                text_b, hist_b, meta_b = [t.to(self.device) for t in batch]
                logits = self.model(text_b, hist_b, meta_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs)

        # Flatten list of arrays
        return np.concatenate(predictions).flatten()

    def save(self):
        """Saves the model state dict."""
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        """Loads the model state dict."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")
