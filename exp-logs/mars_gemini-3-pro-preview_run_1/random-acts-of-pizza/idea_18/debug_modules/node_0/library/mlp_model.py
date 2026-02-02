import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request data.
    """

    def __init__(self, features_dict, y=None):
        self.request_emb = torch.FloatTensor(features_dict["request_emb"])
        self.history_seq = torch.FloatTensor(features_dict["history_seq"])
        self.metadata = torch.FloatTensor(features_dict["metadata"])

        if y is not None:
            self.y = torch.FloatTensor(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        sample = {
            "request_emb": self.request_emb[idx],
            "history_seq": self.history_seq[idx],
            "metadata": self.metadata[idx],
        }
        if self.y is not None:
            return sample, self.y[idx]
        return sample


class GatedAttentionNet(nn.Module):
    """
    Stream B: Credibility-Gated Attention MLP.

    Branches:
    1. Semantic: Processes Request SBERT embedding.
    2. History: Attention mechanism (Query=Request, Key=History) to aggregate user history.
    3. Metadata: Generates a scalar 'Credibility Gate'.

    Fusion:
    The semantic and history features are concatenated, then modulated (multiplied)
    by the credibility gate.
    """

    def __init__(self, metadata_dim):
        super(GatedAttentionNet, self).__init__()

        # Dimensions
        self.sbert_dim = Config.SBERT_EMBEDDING_DIM
        self.hidden_dim = Config.MLP_HIDDEN_DIM
        self.dropout_prob = Config.MLP_DROPOUT

        # 1. Request Branch
        self.req_fc = nn.Linear(self.sbert_dim, self.hidden_dim)
        self.req_dropout = nn.Dropout(self.dropout_prob)

        # 2. History Attention Branch
        # Query projection (from Request)
        self.attn_query = nn.Linear(self.sbert_dim, self.hidden_dim)
        # Key and Value projection (from History)
        self.attn_key = nn.Linear(self.sbert_dim, self.hidden_dim)
        self.attn_value = nn.Linear(self.sbert_dim, self.hidden_dim)

        # 3. Metadata / Credibility Gate Branch
        # Produces a scalar gate [0, 1]
        self.meta_fc1 = nn.Linear(metadata_dim, self.hidden_dim // 2)
        self.meta_fc2 = nn.Linear(self.hidden_dim // 2, 1)

        # 4. Final Classifier
        # Input is concatenated (Request_Hidden + History_Context) = 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, request_emb, history_seq, metadata):
        # --- 1. Request Feature ---
        # (Batch, SBERT_Dim) -> (Batch, Hidden)
        req_feat = F.relu(self.req_fc(request_emb))
        req_feat = self.req_dropout(req_feat)

        # --- 2. History Attention ---
        # history_seq: (Batch, Seq_Len, SBERT_Dim)

        # Query: (Batch, 1, Hidden) derived from original request embedding
        query = self.attn_query(request_emb).unsqueeze(1)

        # Key, Value: (Batch, Seq_Len, Hidden)
        key = self.attn_key(history_seq)
        value = self.attn_value(history_seq)

        # Scores: (Batch, 1, Seq_Len)
        # Q * K^T
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = scores / (self.hidden_dim**0.5)  # Scale

        # Masking padding
        # history_seq was padded with zeros. If a vector is all zeros, it's padding.
        # (Batch, Seq_Len)
        mask = (history_seq.abs().sum(dim=2) > 0).unsqueeze(1)  # (Batch, 1, Seq_Len)

        # Apply mask: set scores to -inf where mask is False (padding)
        # We need to handle the case where a user has NO history (all masked).
        # In that case, softmax would be over all -inf.
        # We fill masked positions with a very large negative number.
        scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        # Context: (Batch, 1, Hidden) -> (Batch, Hidden)
        context = torch.bmm(attn_weights, value).squeeze(1)

        # Handle case where all history was padding (context might be NaN if softmax exploded or zero)
        # If mask sum is 0 for a sample, context should be 0.
        # The masked_fill -1e9 usually handles softmax safely (results in 0 weight),
        # but if all are -1e9, softmax is uniform.
        # Ideally, if no history, context is 0.
        # We can multiply context by a "has_history" flag.
        has_history = (mask.sum(dim=-1) > 0).float().view(-1, 1)
        context = context * has_history

        # --- 3. Credibility Gate ---
        # (Batch, Meta_Dim) -> (Batch, 1)
        gate_h = F.relu(self.meta_fc1(metadata))
        gate = torch.sigmoid(self.meta_fc2(gate_h))

        # --- 4. Fusion ---
        # Concatenate Semantic (Request) and Context (History)
        fused = torch.cat([req_feat, context], dim=1)

        # Modulate by Gate (Broadcasting scalar gate across vector)
        gated_fused = fused * gate

        # --- 5. Classification ---
        logits = self.classifier(gated_fused)

        return logits


class MLPModel:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = None
        self.model_path = os.path.join(
            Config.WORKING_DIR, Config.CACHE_FILES["mlp_model"]
        )

    def _init_model(self, metadata_dim):
        set_seed(Config.RANDOM_STATE)
        self.model = GatedAttentionNet(metadata_dim=metadata_dim)
        self.model.to(self.device)

    def train(self, data_train, data_val=None):
        """
        Args:
            data_train (dict): {'mlp': {...}, 'y': ...}
            data_val (dict): {'mlp': {...}, 'y': ...}
        """
        # Extract metadata dimension from training data
        meta_dim = data_train["mlp"]["metadata"].shape[1]
        self._init_model(meta_dim)

        # Prepare Datasets and Loaders
        train_dataset = PizzaDataset(data_train["mlp"], data_train["y"])
        train_loader = DataLoader(
            train_dataset, batch_size=Config.MLP_BATCH_SIZE, shuffle=True, num_workers=0
        )

        val_loader = None
        if data_val:
            val_dataset = PizzaDataset(data_val["mlp"], data_val["y"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.MLP_BATCH_SIZE * 2,
                shuffle=False,
                num_workers=0,
            )

        # Optimization
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.MLP_LEARNING_RATE,
            weight_decay=Config.MLP_WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping Variables
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None

        print(f"Starting MLP Training on {self.device}...")

        for epoch in range(Config.MLP_EPOCHS):
            # --- Training ---
            self.model.train()
            train_loss_sum = 0
            all_preds = []
            all_targets = []

            for batch_data, batch_y in train_loader:
                # Move to device
                req = batch_data["request_emb"].to(self.device)
                hist = batch_data["history_seq"].to(self.device)
                meta = batch_data["metadata"].to(self.device)
                y = batch_y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(req, hist, meta)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * y.size(0)

                # Store for metrics
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(batch_y.numpy())

            avg_train_loss = train_loss_sum / len(train_dataset)
            train_auc = roc_auc_score(all_targets, all_preds)

            # --- Validation ---
            val_auc = 0.0
            if val_loader:
                self.model.eval()
                val_preds = []
                val_targets = []
                val_loss_sum = 0

                with torch.no_grad():
                    for batch_data, batch_y in val_loader:
                        req = batch_data["request_emb"].to(self.device)
                        hist = batch_data["history_seq"].to(self.device)
                        meta = batch_data["metadata"].to(self.device)
                        y = batch_y.to(self.device).unsqueeze(1)

                        logits = self.model(req, hist, meta)
                        loss = criterion(logits, y)
                        val_loss_sum += loss.item() * y.size(0)

                        probs = torch.sigmoid(logits).cpu().numpy()
                        val_preds.extend(probs)
                        val_targets.extend(batch_y.numpy())

                val_auc = roc_auc_score(val_targets, val_preds)
                avg_val_loss = val_loss_sum / len(val_dataset)

                print(
                    f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | "
                    f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc}"
                )

                # Early Stopping Check
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.MLP_PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_auc}"
                    )
                    break
            else:
                print(
                    f"Epoch {epoch+1}/{Config.MLP_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc}"
                )
                # If no validation, just save the last state
                best_model_state = self.model.state_dict()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.save()

    def predict_proba(self, data_dict):
        """
        Args:
            data_dict (dict): {'mlp': {...}}
        Returns:
            np.array: Probabilities
        """
        if self.model is None:
            self.load()

        dataset = PizzaDataset(data_dict["mlp"], y=None)
        loader = DataLoader(
            dataset, batch_size=Config.MLP_BATCH_SIZE * 2, shuffle=False, num_workers=0
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch_data in loader:
                req = batch_data["request_emb"].to(self.device)
                hist = batch_data["history_seq"].to(self.device)
                meta = batch_data["metadata"].to(self.device)

                logits = self.model(req, hist, meta)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs).flatten()

    def save(self):
        if self.model is not None:
            torch.save(self.model, self.model_path)
            print(f"MLP Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.to(self.device)
            print(f"MLP Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"MLP Model not found at {self.model_path}")
