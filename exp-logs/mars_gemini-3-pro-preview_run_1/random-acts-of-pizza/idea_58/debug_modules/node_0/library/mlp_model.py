import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class PizzaDataset(Dataset):
    def __init__(self, feature_dict, labels=None):
        """
        Args:
            feature_dict (dict): Dictionary containing numpy arrays for features.
            labels (np.ndarray, optional): Array of labels.
        """
        self.title_emb = torch.FloatTensor(feature_dict["title_emb"])
        self.body_emb = torch.FloatTensor(feature_dict["body_emb"])
        self.metadata = torch.FloatTensor(feature_dict["metadata"])
        self.top_k = torch.FloatTensor(feature_dict["top_k"])
        self.history_centroid = torch.FloatTensor(feature_dict["history_centroid"])
        self.consistency = torch.FloatTensor(feature_dict["consistency"])
        self.sentiment = torch.FloatTensor(feature_dict["sentiment"])

        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        item = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "metadata": self.metadata[idx],
            "top_k": self.top_k[idx],
            "history_centroid": self.history_centroid[idx],
            "consistency": self.consistency[idx],
            "sentiment": self.sentiment[idx],
        }

        if self.labels is not None:
            return item, self.labels[idx]
        return item


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class DecoupledGatedMLP(nn.Module):
    def __init__(self):
        super(DecoupledGatedMLP, self).__init__()

        # Dimensions from Config
        self.emb_dim = Config.SBERT_EMBEDDING_DIM
        self.meta_dim = 7  # Defined in features.py
        self.top_k_dim = Config.TOP_K_SUBREDDITS
        self.consistency_dim = 2
        self.sentiment_dim = 4

        # Hyperparameters
        self.topic_hidden_dim = Config.MLP_TOPIC_HIDDEN_DIM
        self.narrative_hidden_dim = Config.MLP_NARRATIVE_HIDDEN_DIM
        self.gate_hidden_dim = Config.MLP_GATE_HIDDEN_DIM
        self.final_hidden_dim = Config.MLP_FINAL_HIDDEN_DIM
        self.dropout_emb = Config.MLP_DROPOUT_EMB
        self.dropout_dense = Config.MLP_DROPOUT_DENSE

        # --- Stream 1: Topic (Title + History) ---
        # Input: Title (384) + History (384) = 768
        self.topic_projector = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.topic_hidden_dim),
            nn.BatchNorm1d(self.topic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_emb),
        )

        # --- Stream 2: Narrative (Body + History) ---
        # Input: Body (384) + History (384) = 768
        self.narrative_projector = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.narrative_hidden_dim),
            nn.BatchNorm1d(self.narrative_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_emb),
        )

        # --- Decoupled Gates (Driven by Metadata) ---

        # Topic Gate: Controls how much the Topic stream contributes
        self.topic_gate = nn.Sequential(
            nn.Linear(self.meta_dim, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gate_hidden_dim, self.topic_hidden_dim),
            nn.Sigmoid(),
        )

        # Narrative Gate: Controls how much the Narrative stream contributes
        self.narrative_gate = nn.Sequential(
            nn.Linear(self.meta_dim, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.gate_hidden_dim, self.narrative_hidden_dim),
            nn.Sigmoid(),
        )

        # --- Fusion Layer ---
        # Inputs:
        # 1. Gated Topic (topic_hidden_dim)
        # 2. Gated Narrative (narrative_hidden_dim)
        # 3. Skip Connection: Metadata (meta_dim)
        # 4. Skip Connection: Top-K (top_k_dim)
        # 5. Skip Connection: Consistency (consistency_dim)
        # 6. Skip Connection: Sentiment (sentiment_dim)

        fusion_input_dim = (
            self.topic_hidden_dim
            + self.narrative_hidden_dim
            + self.meta_dim
            + self.top_k_dim
            + self.consistency_dim
            + self.sentiment_dim
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.final_hidden_dim),
            nn.BatchNorm1d(self.final_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_dense),
            nn.Linear(self.final_hidden_dim, 1),
        )

    def forward(self, inputs):
        title = inputs["title_emb"]
        body = inputs["body_emb"]
        meta = inputs["metadata"]
        top_k = inputs["top_k"]
        history = inputs["history_centroid"]
        consistency = inputs["consistency"]
        sentiment = inputs["sentiment"]

        # Stream 1: Topic Processing
        # Concatenate Title and History
        topic_input = torch.cat([title, history], dim=1)
        topic_feat = self.topic_projector(topic_input)

        # Stream 2: Narrative Processing
        # Concatenate Body and History
        narrative_input = torch.cat([body, history], dim=1)
        narrative_feat = self.narrative_projector(narrative_input)

        # Gating Mechanisms
        # Gates are derived strictly from metadata
        g_topic = self.topic_gate(meta)
        g_narrative = self.narrative_gate(meta)

        # Apply Gates (Element-wise multiplication)
        gated_topic = topic_feat * g_topic
        gated_narrative = narrative_feat * g_narrative

        # Fusion with Skip Connections
        fusion_vec = torch.cat(
            [gated_topic, gated_narrative, meta, top_k, consistency, sentiment], dim=1
        )

        logits = self.fusion_layer(fusion_vec)
        return logits.squeeze(1)


# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------
class MLPTrainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = DecoupledGatedMLP().to(self.device)
        self.model_path = Config.MODEL_MLP_PATH

        # Training Config
        self.batch_size = Config.MLP_BATCH_SIZE
        self.epochs = Config.MLP_EPOCHS
        self.patience = Config.MLP_PATIENCE
        self.learning_rate = Config.MLP_LEARNING_RATE
        self.weight_decay = Config.MLP_WEIGHT_DECAY

        set_seed(Config.RANDOM_SEED)

    def train(self, train_data, val_data):
        """
        Args:
            train_data: tuple (feature_dict, labels)
            val_data: tuple (feature_dict, labels)
        """
        # Prepare Datasets
        train_dataset = PizzaDataset(train_data[0], train_data[1])
        val_dataset = PizzaDataset(val_data[0], val_data[1])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Optimizer & Loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early Stopping State
        best_auc = 0.0
        patience_counter = 0

        print(f"Starting MLP training on {self.device}...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_inputs, batch_labels in train_loader:
                # Move to device
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_labels.size(0)

            avg_train_loss = train_loss / len(train_dataset)

            # Validation
            val_auc, val_loss = self._evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                self._save_model()
                # print(f"  -> New best model saved (AUC: {best_auc})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # Load best model for final state
        self._load_model()
        print(f"Training complete. Best Val AUC: {best_auc}")

    def predict(self, feature_dict):
        """
        Args:
            feature_dict: Dictionary of features
        Returns:
            np.ndarray: Probabilities
        """
        if not os.path.exists(self.model_path):
            self._load_model()  # Try loading, might fail if not trained

        dataset = PizzaDataset(feature_dict)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch_inputs in loader:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                logits = self.model(batch_inputs)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        return np.concatenate(all_preds)

    def _evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_inputs, batch_labels in loader:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_labels)

                total_loss += loss.item() * batch_labels.size(0)
                probs = torch.sigmoid(logits)

                all_labels.append(batch_labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        return auc, avg_loss

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            print(f"Warning: Model file not found at {self.model_path}")
