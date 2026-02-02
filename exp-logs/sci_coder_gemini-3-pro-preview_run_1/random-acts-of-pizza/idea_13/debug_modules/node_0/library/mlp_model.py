import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
import os

from library.config import Config

# Set seeds for reproducibility
torch.manual_seed(Config.RANDOM_STATE)
np.random.seed(Config.RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.RANDOM_STATE)


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request data.
    Handles dictionary-based inputs containing embeddings and metadata.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing tensors:
                              'request_emb': (N, Dim)
                              'history_emb': (N, Seq, Dim)
                              'meta_features': (N, MetaDim)
                              'y': (N,) [Optional]
        """
        self.request_emb = data_dict["request_emb"]
        self.history_emb = data_dict["history_emb"]
        self.meta_features = data_dict["meta_features"]
        self.y = data_dict.get("y", None)

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        sample = {
            "request_emb": self.request_emb[idx],
            "history_emb": self.history_emb[idx],
            "meta_features": self.meta_features[idx],
        }
        if self.y is not None:
            sample["y"] = self.y[idx]
        return sample


class CredibilityGatedNet(nn.Module):
    """
    Neural Network with Credibility-Gated Attention Mechanism.

    Structure:
    1. Request Branch: Encodes request text embedding.
    2. History Branch: Applies Dot-Product Attention (Query=Request, Key=History).
    3. Metadata Branch: Generates a 'Credibility Gate' via Sigmoid.
    4. Fusion: (Request + Attended_History) * Credibility_Gate.
    """

    def __init__(self, input_emb_dim, meta_dim, hidden_dim, dropout_rate):
        super(CredibilityGatedNet, self).__init__()

        # Branch 1: Request Semantics Encoder
        self.req_encoder = nn.Sequential(
            nn.Linear(input_emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Branch 2: History Encoder (Post-Attention projection)
        self.hist_encoder = nn.Sequential(
            nn.Linear(input_emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )

        # Branch 3: Metadata (Credibility Gate)
        # Projects metadata to the size of the concatenated semantic vector (2 * hidden_dim)
        # Ends with Sigmoid to act as a gate [0, 1]
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid(),
        )

        # Classifier Head
        # Takes the gated fused vector and predicts probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # Scaling factor for attention
        self.scale = torch.sqrt(torch.FloatTensor([input_emb_dim])).to(Config.DEVICE)

    def forward(self, request_emb, history_emb, meta_features):
        """
        Args:
            request_emb: (Batch, Dim)
            history_emb: (Batch, Seq, Dim)
            meta_features: (Batch, MetaDim)
        """

        # --- Attention Mechanism ---
        # Query: Request (B, 1, Dim)
        query = request_emb.unsqueeze(1)
        # Key: History (B, Seq, Dim)
        keys = history_emb

        # Scores: (B, 1, Seq) = Q * K^T / sqrt(d)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale

        # Masking: Identify padding in history (where vector is all zeros)
        # history_emb is (B, Seq, Dim). Sum abs over Dim. If approx 0, it's padding.
        # mask: (B, 1, Seq) - True where valid, False where padding
        mask = (history_emb.abs().sum(dim=2) > 1e-6).unsqueeze(1)

        # Apply mask: fill padding positions with large negative value before softmax
        scores = scores.masked_fill(~mask, -1e9)

        # Weights: (B, 1, Seq)
        attn_weights = torch.softmax(scores, dim=2)

        # Context: (B, 1, Dim) = Weights * Values(History)
        # This represents the weighted history relevant to the request
        context = torch.bmm(attn_weights, keys)
        context = context.squeeze(1)

        # --- Encoding ---
        h_req = self.req_encoder(request_emb)  # (B, Hidden)
        h_hist = self.hist_encoder(context)  # (B, Hidden)

        # --- Concatenation ---
        # Combine current request semantics with relevant historical context
        h_sem = torch.cat([h_req, h_hist], dim=1)  # (B, 2*Hidden)

        # --- Gating ---
        # Generate credibility gate from metadata
        gate = self.meta_encoder(meta_features)  # (B, 2*Hidden)

        # Modulate semantics with credibility gate
        # If gate is low (low credibility), the semantic signal is dampened
        h_fused = h_sem * gate

        # --- Classification ---
        logits = self.classifier(h_fused)

        return logits


class MLPTrainer:
    """
    Trainer class for the Credibility-Gated MLP.
    Handles training loop, validation, early stopping, and prediction.
    """

    def __init__(self, input_emb_dim=Config.EMBEDDING_DIM, meta_dim=13, params=None):
        self.params = params if params is not None else Config.MLP_PARAMS
        self.device = Config.DEVICE

        # Initialize Model
        self.model = CredibilityGatedNet(
            input_emb_dim=input_emb_dim,
            meta_dim=meta_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout_rate=self.params["dropout_rate"],
        ).to(self.device)

        # Loss and Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            verbose=False,
        )

    def train(self, train_data, val_data):
        """
        Runs the training loop with early stopping.

        Args:
            train_data (dict): Training data dictionary.
            val_data (dict): Validation data dictionary.

        Returns:
            float: Best Validation ROC AUC.
        """
        train_dataset = PizzaDataset(train_data)
        val_dataset = PizzaDataset(val_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device == "cuda"),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device == "cuda"),
        )

        best_val_auc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(
            f"Starting MLP training for {self.params['epochs']} epochs on {self.device}..."
        )

        for epoch in range(self.params["epochs"]):
            # --- Training Phase ---
            self.model.train()
            train_loss_sum = 0.0
            all_train_probs = []
            all_train_targets = []

            for batch in train_loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_emb"].to(self.device)
                meta = batch["meta_features"].to(self.device)
                y = batch["y"].to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()

                logits = self.model(req, hist, meta)
                loss = self.criterion(logits, y)

                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item() * req.size(0)
                all_train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
                all_train_targets.extend(y.detach().cpu().numpy())

            train_loss = train_loss_sum / len(train_dataset)
            train_auc = roc_auc_score(all_train_targets, all_train_probs)

            # --- Validation Phase ---
            self.model.eval()
            val_loss_sum = 0.0
            all_val_probs = []
            all_val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    req = batch["request_emb"].to(self.device)
                    hist = batch["history_emb"].to(self.device)
                    meta = batch["meta_features"].to(self.device)
                    y = batch["y"].to(self.device).unsqueeze(1)

                    logits = self.model(req, hist, meta)
                    loss = self.criterion(logits, y)

                    val_loss_sum += loss.item() * req.size(0)
                    all_val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    all_val_targets.extend(y.cpu().numpy())

            val_loss = val_loss_sum / len(val_dataset)
            val_auc = roc_auc_score(all_val_targets, all_val_probs)

            # Step Scheduler
            self.scheduler.step(val_auc)

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{self.params['epochs']} - "
                f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc} - "
                f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc}"
            )

            # --- Early Stopping ---
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
        print(f"Best Validation AUC: {best_val_auc}")
        return best_val_auc

    def predict_proba(self, test_data):
        """
        Generates probability predictions for the test set.

        Args:
            test_data (dict): Test data dictionary.

        Returns:
            np.ndarray: Array of probabilities.
        """
        test_dataset = PizzaDataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device == "cuda"),
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_emb"].to(self.device)
                meta = batch["meta_features"].to(self.device)

                logits = self.model(req, hist, meta)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())

        return np.array(all_probs)
