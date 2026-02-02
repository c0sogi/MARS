import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import os


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DualBranchMLP(nn.Module):
    """
    Dual-Branch MLP architecture.
    Branch 1: Semantic (Text Embeddings) - Strong regularization via Dropout.
    Branch 2: Metadata (Numerical Features) - Weak/No regularization to preserve signal.
    """

    def __init__(self, semantic_input_dim, meta_input_dim, hidden_dim, dropout_rate):
        super(DualBranchMLP, self).__init__()

        # Branch 1: Semantic
        self.semantic_branch = nn.Sequential(
            nn.Linear(semantic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Branch 2: Metadata
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_input_dim, hidden_dim), nn.ReLU()
        )

        # Fusion Head
        # Concatenates the outputs of both branches (hidden_dim + hidden_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_sem, x_meta):
        out_sem = self.semantic_branch(x_sem)
        out_meta = self.meta_branch(x_meta)

        # Concatenate along feature dimension
        combined = torch.cat((out_sem, out_meta), dim=1)

        return self.fusion_head(combined)


class MLPTrainer:
    """
    Trainer class for the DualBranchMLP.
    Handles data conversion, training loop, validation, and early stopping.
    """

    def __init__(self, params):
        self.params = params
        self.device = torch.device(
            params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None

    def train(self, train_data, val_data):
        # Ensure reproducibility
        set_seed(42)

        # Unpack data: (X_sem, X_meta, y)
        X_sem_train, X_meta_train, y_train = train_data
        X_sem_val, X_meta_val, y_val = val_data

        # Convert to Tensors
        X_sem_train_t = torch.FloatTensor(X_sem_train)
        X_meta_train_t = torch.FloatTensor(X_meta_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

        X_sem_val_t = torch.FloatTensor(X_sem_val)
        X_meta_val_t = torch.FloatTensor(X_meta_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        # Create DataLoader
        train_dataset = TensorDataset(X_sem_train_t, X_meta_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )

        # Initialize Model
        meta_dim = X_meta_train.shape[1]
        self.model = DualBranchMLP(
            semantic_input_dim=self.params["semantic_input_dim"],
            meta_input_dim=meta_dim,
            hidden_dim=self.params["hidden_dim"],
            dropout_rate=self.params["dropout_rate"],
        ).to(self.device)

        # Setup Optimization
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.BCELoss()

        # Training Loop
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None

        print(f"Starting training on {self.device}...")

        for epoch in range(self.params["epochs"]):
            self.model.train()
            total_loss = 0.0

            for b_sem, b_meta, b_y in train_loader:
                b_sem, b_meta, b_y = (
                    b_sem.to(self.device),
                    b_meta.to(self.device),
                    b_y.to(self.device),
                )

                optimizer.zero_grad()
                outputs = self.model(b_sem, b_meta)
                loss = criterion(outputs, b_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * b_sem.size(0)

            avg_train_loss = total_loss / len(train_dataset)

            # Validation
            val_auc, val_loss = self.evaluate(
                X_sem_val_t, X_meta_val_t, y_val_t, criterion
            )

            print(
                f"Epoch {epoch+1}/{self.params['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
            )

            # Early Stopping Logic
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.params["patience"]:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_val_auc}"
                )
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def evaluate(self, X_sem, X_meta, y, criterion):
        self.model.eval()
        X_sem = X_sem.to(self.device)
        X_meta = X_meta.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            outputs = self.model(X_sem, X_meta)
            loss = criterion(outputs, y)
            preds = outputs.cpu().numpy()
            targets = y.cpu().numpy()

        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:
            # Fallback for edge cases (e.g., batch has only one class)
            auc = 0.5

        return auc, loss.item()

    def predict(self, test_data):
        """
        Generates predictions for test data.
        test_data is expected to be a tuple: (X_sem, X_meta, ids)
        """
        X_sem, X_meta = test_data[0], test_data[1]

        X_sem_t = torch.FloatTensor(X_sem).to(self.device)
        X_meta_t = torch.FloatTensor(X_meta).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_sem_t, X_meta_t)
            preds = outputs.cpu().numpy().flatten()

        return preds
