import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import random

import library.config as config
from library.features import FeatureProcessor


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(config.RANDOM_STATE)


# ==========================================
# Dataset
# ==========================================
class PizzaDataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.request_emb = torch.FloatTensor(data_dict["mlp_request_emb"])
        self.history_seq = torch.FloatTensor(data_dict["mlp_history_seq"])
        self.history_mask = torch.FloatTensor(data_dict["mlp_history_mask"])
        self.metadata = torch.FloatTensor(data_dict["mlp_metadata"])

        if mode != "test":
            self.labels = torch.FloatTensor(data_dict["labels"])
        else:
            self.labels = None

    def __len__(self):
        return len(self.request_emb)

    def __getitem__(self, idx):
        sample = {
            "request_emb": self.request_emb[idx],
            "history_seq": self.history_seq[idx],
            "history_mask": self.history_mask[idx],
            "metadata": self.metadata[idx],
        }
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        return sample


# ==========================================
# Model Architecture
# ==========================================
class GatedAttentionNet(nn.Module):
    def __init__(self, metadata_dim):
        super(GatedAttentionNet, self).__init__()

        self.embedding_dim = config.SBERT_EMBEDDING_DIM

        # Branch 3: Metadata Gating
        # The gate will modulate the concatenated semantic vector (Request + Context)
        # Size of semantic vector = embedding_dim (Request) + embedding_dim (Context) = 2 * embedding_dim
        self.semantic_dim = 2 * self.embedding_dim

        self.gate_layer = nn.Sequential(
            nn.Linear(metadata_dim, self.semantic_dim), nn.Sigmoid()
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim, config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.MLP_DROPOUT_RATE),
            nn.Linear(config.MLP_HIDDEN_DIM, 1),
        )

    def forward(self, request_emb, history_seq, history_mask, metadata):
        """
        Args:
            request_emb: (B, 384)
            history_seq: (B, L, 384)
            history_mask: (B, L) - 1 for valid, 0 for padding
            metadata: (B, M)
        """

        # --- Branch 2: Dot-Product Attention ---
        # Query: Request (B, 384) -> (B, 1, 384)
        query = request_emb.unsqueeze(1)

        # Keys: History (B, L, 384)
        keys = history_seq

        # Scores: (B, 1, L)
        # Transpose keys to (B, 384, L) for matmul
        scores = torch.bmm(query, keys.transpose(1, 2))

        # Apply Mask
        # Mask is (B, L). Unsqueeze to (B, 1, L)
        mask = history_mask.unsqueeze(1)
        # Set masked positions to large negative value before softmax
        scores = scores.masked_fill(mask == 0, -1e9)

        # Attention Weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, L)

        # Context Vector: Weighted sum of values (History)
        # (B, 1, L) x (B, L, 384) -> (B, 1, 384)
        context = torch.bmm(attn_weights, history_seq)
        context = context.squeeze(1)  # (B, 384)

        # --- Fusion ---
        # Concatenate Request and Context
        semantic_vector = torch.cat([request_emb, context], dim=1)  # (B, 768)

        # --- Branch 3: Credibility Gating ---
        gate = self.gate_layer(metadata)  # (B, 768)

        # Modulate
        gated_vector = semantic_vector * gate

        # --- Classification ---
        logits = self.classifier(gated_vector)
        return logits


# ==========================================
# Trainer
# ==========================================
class MLPTrainer:
    def __init__(self, metadata_dim, device):
        self.device = device
        self.model = GatedAttentionNet(metadata_dim).to(device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.MLP_LEARNING_RATE,
            weight_decay=config.MLP_WEIGHT_DECAY,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, train_loader, val_loader):
        print(f"Starting MLP training on {self.device}...")
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(config.EPOCHS):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                logits = self.model(req, hist, mask, meta)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * req.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation
            val_auc, val_loss = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc}"
            )

            # Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_auc

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)

                logits = self.model(req, hist, mask, meta)
                preds = torch.sigmoid(logits)

                if "label" in batch:
                    labels = batch["label"].to(self.device).unsqueeze(1)
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item() * req.size(0)
                    all_labels.extend(labels.cpu().numpy())

                all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0

        auc = 0.0
        if len(all_labels) > 0:
            try:
                auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                auc = 0.0

        return auc, avg_loss

    def predict(self, loader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                req = batch["request_emb"].to(self.device)
                hist = batch["history_seq"].to(self.device)
                mask = batch["history_mask"].to(self.device)
                meta = batch["metadata"].to(self.device)

                logits = self.model(req, hist, mask, meta)
                preds = torch.sigmoid(logits)
                all_preds.extend(preds.cpu().numpy().flatten())

        return np.array(all_preds)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


# ==========================================
# Main Execution Stream
# ==========================================
def run_mlp_stream(load_cached_data=True):
    """
    Executes the full MLP pipeline.
    """
    # 1. Load Data
    print("Initializing FeatureProcessor for MLP Stream...")
    processor = FeatureProcessor()
    train_data, val_data, test_data = processor.process(
        load_cached_data=load_cached_data
    )

    # 2. Prepare Datasets and Loaders
    train_dataset = PizzaDataset(train_data, mode="train")
    val_dataset = PizzaDataset(val_data, mode="val")
    test_dataset = PizzaDataset(test_data, mode="test")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Determine metadata dimension
    metadata_dim = train_data["mlp_metadata"].shape[1]

    # 3. Initialize and Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MLPTrainer(metadata_dim, device)

    trainer.train(train_loader, val_loader)

    # 4. Generate Predictions
    print("Generating MLP predictions...")
    val_preds = trainer.predict(val_loader)
    test_preds = trainer.predict(test_loader)

    # 5. Save Artifacts
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    val_preds_path = os.path.join(config.WORKING_DIR, "mlp_val_preds.npy")
    test_preds_path = os.path.join(config.WORKING_DIR, "mlp_test_preds.npy")
    model_path = os.path.join(config.WORKING_DIR, "mlp_model.pth")

    np.save(val_preds_path, val_preds)
    np.save(test_preds_path, test_preds)
    trainer.save(model_path)

    print(f"MLP predictions saved to {val_preds_path} and {test_preds_path}")
    print(f"MLP model saved to {model_path}")

    return val_preds, test_preds
