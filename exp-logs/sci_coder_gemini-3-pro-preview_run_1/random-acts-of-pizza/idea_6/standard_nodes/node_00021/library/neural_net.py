import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from library import config, utils


class PizzaDataset(Dataset):
    """
    Custom Dataset for the Dual-Branch architecture.
    Handles SBERT embeddings, dense ratio features, and optional targets.
    """

    def __init__(self, embeddings, dense_features, targets=None):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.dense_features = torch.tensor(dense_features, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        dense = self.dense_features[idx]

        if self.targets is not None:
            target = self.targets[idx]
            return embedding, dense, target
        else:
            return embedding, dense


class DualBranchMLP(nn.Module):
    """
    Stabilized Dual-Branch MLP.
    Branch 1: Semantic (Text) - High Dropout
    Branch 2: Ratio-Metadata (Tabular) - Batch Norm, Low Dropout
    """

    def __init__(self, dense_input_dim, mlp_config=config.MLP_CONFIG):
        super(DualBranchMLP, self).__init__()

        # --- Branch 1: Semantic (SBERT Embeddings) ---
        self.semantic_branch = nn.Sequential(
            nn.Linear(mlp_config["embedding_dim"], mlp_config["semantic_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(mlp_config["semantic_dropout"]),
        )

        # --- Branch 2: Ratio-Metadata (Dense Features) ---
        self.ratio_branch = nn.Sequential(
            nn.Linear(dense_input_dim, mlp_config["ratio_hidden_dim"]),
            nn.BatchNorm1d(mlp_config["ratio_hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(mlp_config["ratio_dropout"]),
        )

        # --- Fusion ---
        fusion_input_dim = (
            mlp_config["semantic_hidden_dim"] + mlp_config["ratio_hidden_dim"]
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_config["fusion_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(mlp_config["fusion_hidden_dim"], mlp_config["output_dim"]),
        )

    def forward(self, embedding, dense):
        # Process branches
        sem_out = self.semantic_branch(embedding)
        ratio_out = self.ratio_branch(dense)

        # Concatenate
        combined = torch.cat((sem_out, ratio_out), dim=1)

        # Final prediction (logits)
        logits = self.fusion_layer(combined)
        return logits


def train_model(train_data, val_data, dense_input_dim):
    """
    Trains the DualBranchMLP model with Early Stopping.

    Args:
        train_data (dict): {'embedding': np.array, 'dense': np.array, 'y': np.array}
        val_data (dict): {'embedding': np.array, 'dense': np.array, 'y': np.array}
        dense_input_dim (int): Number of dense features.

    Returns:
        model: The trained PyTorch model (best state).
    """
    # Set seed for reproducibility
    utils.set_seed()

    device = torch.device(config.MLP_CONFIG["device"])

    # Prepare Datasets and Loaders
    train_dataset = PizzaDataset(
        train_data["embedding"], train_data["dense"], train_data["y"]
    )
    val_dataset = PizzaDataset(val_data["embedding"], val_data["dense"], val_data["y"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.MLP_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Validation loader can be larger since no backprop
    val_loader = DataLoader(
        val_dataset, batch_size=config.MLP_CONFIG["batch_size"] * 2, shuffle=False
    )

    # Initialize Model
    model = DualBranchMLP(dense_input_dim).to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.MLP_CONFIG["learning_rate"],
        weight_decay=config.MLP_CONFIG["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(config.MLP_CONFIG["epochs"]):
        model.train()
        train_loss = 0.0

        for embeddings, dense, targets in train_loader:
            embeddings = embeddings.to(device)
            dense = dense.to(device)
            targets = targets.to(device).unsqueeze(1)  # Match output shape (B, 1)

            optimizer.zero_grad()
            logits = model(embeddings, dense)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * embeddings.size(0)

        train_loss /= len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for embeddings, dense, targets in val_loader:
                embeddings = embeddings.to(device)
                dense = dense.to(device)
                targets = targets.to(device).unsqueeze(1)

                logits = model(embeddings, dense)
                loss = criterion(logits, targets)

                val_loss += loss.item() * embeddings.size(0)
                probs = torch.sigmoid(logits)

                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(all_targets, all_probs)

        print(
            f"Epoch {epoch+1}/{config.MLP_CONFIG['epochs']} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.MLP_CONFIG["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val AUC: {best_val_auc}")

    return model


def predict_model(model, test_data):
    """
    Generates predictions for the test set.

    Args:
        model: Trained PyTorch model.
        test_data (dict): {'embedding': np.array, 'dense': np.array}

    Returns:
        np.array: Probabilities of class 1.
    """
    device = torch.device(config.MLP_CONFIG["device"])
    model.eval()

    dataset = PizzaDataset(test_data["embedding"], test_data["dense"], targets=None)
    loader = DataLoader(
        dataset, batch_size=config.MLP_CONFIG["batch_size"] * 2, shuffle=False
    )

    all_probs = []

    with torch.no_grad():
        for embeddings, dense in loader:
            embeddings = embeddings.to(device)
            dense = dense.to(device)

            logits = model(embeddings, dense)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs).flatten()
