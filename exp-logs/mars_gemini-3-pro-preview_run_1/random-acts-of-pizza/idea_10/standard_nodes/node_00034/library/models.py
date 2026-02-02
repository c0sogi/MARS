import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse
import copy
import os

from library import config, utils

# =============================================================================
# STREAM A: RANDOM FOREST
# =============================================================================


class StreamARandomForest:
    """
    Wrapper for the Topic-Augmented Random Forest (Stream A).
    Handles the combination of sparse TF-IDF features and dense metadata.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=config.RF_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=config.RF_N_JOBS,
            verbose=0,
        )

    def _prepare_data(self, X_tfidf, X_meta):
        """
        Combines sparse TF-IDF matrix and dense metadata array.
        """
        # Ensure X_meta is 2D
        if len(X_meta.shape) == 1:
            X_meta = X_meta.reshape(-1, 1)

        # Convert dense metadata to sparse matrix for efficient stacking
        X_meta_sparse = scipy.sparse.csr_matrix(X_meta)

        # Horizontally stack sparse matrices
        X_combined = scipy.sparse.hstack([X_tfidf, X_meta_sparse])
        return X_combined

    def train(self, X_tfidf, X_meta, y):
        """
        Trains the Random Forest model.
        """
        print("Training Stream A (Random Forest)...")
        X_combined = self._prepare_data(X_tfidf, X_meta)
        self.model.fit(X_combined, y)

        # Optional: Print training score for sanity check
        y_pred = self.model.predict_proba(X_combined)[:, 1]
        score = utils.compute_score(y, y_pred)
        print(f"Stream A Train AUC: {score}")

    def predict_proba(self, X_tfidf, X_meta):
        """
        Generates probability predictions.
        """
        X_combined = self._prepare_data(X_tfidf, X_meta)
        # Return probabilities for the positive class (1)
        return self.model.predict_proba(X_combined)[:, 1]


# =============================================================================
# STREAM B: PYTORCH DATASET
# =============================================================================


class PizzaDataset(Dataset):
    """
    Custom PyTorch Dataset to handle the multi-modal input dictionary.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing 'y' and 'stream_b' features.
        """
        self.y = torch.FloatTensor(data_dict["y"])
        sb = data_dict["stream_b"]
        self.req_emb = torch.FloatTensor(sb["X_request_emb"])
        self.hist_emb = torch.FloatTensor(sb["X_history_emb"])
        self.hist_mask = torch.FloatTensor(sb["X_history_mask"])
        self.meta = torch.FloatTensor(sb["X_meta"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "req_emb": self.req_emb[idx],
            "hist_emb": self.hist_emb[idx],
            "hist_mask": self.hist_mask[idx],
            "meta": self.meta[idx],
            "y": self.y[idx],
        }


# =============================================================================
# STREAM B: CONTEXT-GATED ATTENTION MLP
# =============================================================================


class StreamBContextGatedMLP(nn.Module):
    """
    Neural Network with Context-Gated Attention (Stream B).
    Fuses Request Text, History Context, and Metadata.
    """

    def __init__(self, meta_dim):
        super(StreamBContextGatedMLP, self).__init__()

        self.hidden_dim = config.MLP_HIDDEN_DIM
        text_dim = config.MLP_INPUT_DIM_TEXT

        # --- Branch 1: Request Processing ---
        self.req_proj = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.MLP_DROPOUT),
        )

        # --- Branch 2: History Attention ---
        # Projects inputs to hidden space for attention calculation
        self.att_query = nn.Linear(text_dim, self.hidden_dim)  # Project request
        self.att_key = nn.Linear(text_dim, self.hidden_dim)  # Project history
        self.att_value = nn.Linear(text_dim, self.hidden_dim)  # Project history

        # --- Branch 3: Metadata Processing & Gating ---
        # The semantic vector is concatenation of Request Hidden + Context Hidden
        self.semantic_dim = 2 * self.hidden_dim

        # Gate generator: Metadata -> Sigmoid Gate
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.semantic_dim),
            nn.Sigmoid(),
        )

        # --- Final Classification ---
        # Input: (Gated Semantic Vector) + (Residual Metadata)
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim + meta_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.MLP_DROPOUT),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, req_emb, hist_emb, hist_mask, meta):
        # 1. Process Request
        h_req = self.req_proj(req_emb)  # (B, hidden)

        # 2. Process History (Attention)
        # Q: (B, 1, hidden)
        Q = self.att_query(req_emb).unsqueeze(1)

        # K, V: (B, seq_len, hidden)
        K = self.att_key(hist_emb)
        V = self.att_value(hist_emb)

        # Attention Scores: (B, 1, seq_len)
        # Scaled Dot-Product
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim**0.5)

        # Apply Mask (Set padded positions to -inf)
        mask_expanded = hist_mask.unsqueeze(1)  # (B, 1, seq_len)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Weights
        weights = torch.softmax(scores, dim=-1)

        # Context Vector: (B, hidden)
        ctx = torch.bmm(weights, V).squeeze(1)

        # 3. Fusion & Gating
        h_sem = torch.cat([h_req, ctx], dim=1)  # (B, 2*hidden)
        gate = self.meta_gate(meta)  # (B, 2*hidden)
        h_gated = h_sem * gate  # Element-wise modulation

        # 4. Residual & Output
        h_final = torch.cat([h_gated, meta], dim=1)
        logits = self.classifier(h_final)

        return logits


# =============================================================================
# STREAM B: TRAINER
# =============================================================================


class MLPTrainer:
    """
    Handles training, validation, and inference for the MLP model.
    """

    def __init__(self, meta_dim):
        self.device = config.DEVICE
        self.model = StreamBContextGatedMLP(meta_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.best_model_state = None

    def fit(self, train_data, val_data):
        """
        Trains the model with Early Stopping.
        """
        print("Training Stream B (MLP)...")
        utils.set_seed(config.SEED)  # Ensure deterministic training

        train_dataset = PizzaDataset(train_data)
        val_dataset = PizzaDataset(val_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(config.EPOCHS):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                # Move data to device
                req = batch["req_emb"].to(self.device)
                hist = batch["hist_emb"].to(self.device)
                mask = batch["hist_mask"].to(self.device)
                meta = batch["meta"].to(self.device)
                y = batch["y"].to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                logits = self.model(req, hist, mask, meta)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * req.size(0)

            train_loss /= len(train_dataset)

            # Validation
            val_auc, val_loss = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | Val AUC: {val_auc}"
            )

            # Early Stopping Logic
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with Val AUC: {best_val_auc}")

    def evaluate(self, loader):
        """
        Evaluates the model on a dataloader.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                req = batch["req_emb"].to(self.device)
                hist = batch["hist_emb"].to(self.device)
                mask = batch["hist_mask"].to(self.device)
                meta = batch["meta"].to(self.device)
                y = batch["y"].to(self.device).unsqueeze(1)

                logits = self.model(req, hist, mask, meta)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * req.size(0)

                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        auc = utils.compute_score(
            np.array(all_targets).flatten(), np.array(all_preds).flatten()
        )
        return auc, avg_loss

    def predict_proba(self, test_data):
        """
        Generates predictions for test data.
        """
        test_dataset = PizzaDataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if self.device == "cuda" else False,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                req = batch["req_emb"].to(self.device)
                hist = batch["hist_emb"].to(self.device)
                mask = batch["hist_mask"].to(self.device)
                meta = batch["meta"].to(self.device)

                logits = self.model(req, hist, mask, meta)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds).flatten()
