import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, compute_auc


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Multi-Branch MLP.
    Handles Semantic, Reliability, and Community feature sets.
    """

    def __init__(self, sem_feats, rel_feats, comm_feats, labels=None):
        self.sem_feats = torch.FloatTensor(sem_feats)
        self.rel_feats = torch.FloatTensor(rel_feats)
        self.comm_feats = torch.FloatTensor(comm_feats)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.sem_feats)

    def __getitem__(self, idx):
        sem = self.sem_feats[idx]
        rel = self.rel_feats[idx]
        comm = self.comm_feats[idx]

        if self.labels is not None:
            return sem, rel, comm, self.labels[idx]
        return sem, rel, comm


class OrthogonalSkipGatedMLP(nn.Module):
    """
    Non-Linear Orthogonal Skip-Gated MLP.

    Architecture:
    - Branch 1 (Semantic): Processes text/history embeddings.
    - Branch 2 (Reliability Gate): Non-linear MLP on metadata to gate Branch 1.
    - Branch 3 (Community Skip): Additive path using community flags + metadata.

    Fusion:
    - Gated Semantic = Branch 1 * Sigmoid(Branch 2)
    - Output = Linear(Concat(Gated Semantic, Branch 3))
    """

    def __init__(
        self,
        sem_dim,
        rel_dim,
        comm_dim,
        hidden_dim,
        gate_hidden_dim,
        dropout_emb,
        dropout_dense,
    ):
        super(OrthogonalSkipGatedMLP, self).__init__()

        # Branch 1: Semantic Content
        # Applies dropout to input embeddings, then projects to hidden space
        self.sem_net = nn.Sequential(
            nn.Dropout(p=dropout_emb),
            nn.Linear(sem_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_dense),
        )

        # Branch 2: Reliability Control (Non-Linear Gate)
        # Projects metadata to a gate vector via a bottleneck
        self.gate_net = nn.Sequential(
            nn.Linear(rel_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, hidden_dim),
            nn.Sigmoid(),  # Output in [0, 1] for gating
        )

        # Branch 3: Community Bias (Skip Connection)
        # Direct projection of community info to hidden space
        self.comm_net = nn.Sequential(nn.Linear(comm_dim, hidden_dim), nn.ReLU())

        # Final Fusion Head
        # Concatenates the gated semantic signal and the community bias
        self.final_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, sem, rel, comm):
        # 1. Process Semantic Content
        h_sem = self.sem_net(sem)

        # 2. Generate Reliability Gate
        g = self.gate_net(rel)

        # 3. Apply Orthogonal Gating (Element-wise multiplication)
        # This allows the model to suppress semantic signals if reliability is low
        h_gated = h_sem * g

        # 4. Process Community Bias (Skip Connection)
        h_comm = self.comm_net(comm)

        # 5. Fusion
        # Concatenate preserved semantic signals with direct community signals
        combined = torch.cat([h_gated, h_comm], dim=1)

        # 6. Prediction
        logits = self.final_head(combined)
        return logits


def train_mlp(
    X_sem,
    X_rel,
    X_comm,
    y,
    X_val_sem=None,
    X_val_rel=None,
    X_val_comm=None,
    y_val=None,
    save_path=None,
):
    """
    Trains the OrthogonalSkipGatedMLP model with Early Stopping.
    """
    set_seed(Config.SEED)

    # Dimensions
    sem_dim = X_sem.shape[1]
    rel_dim = X_rel.shape[1]
    comm_dim = X_comm.shape[1]

    # Hyperparameters from Config
    hidden_dim = Config.MLP_ARCH_PARAMS["hidden_dim"]
    gate_hidden_dim = Config.MLP_ARCH_PARAMS["gate_hidden_dim"]
    dropout_emb = Config.MLP_ARCH_PARAMS["dropout_emb"]
    dropout_dense = Config.MLP_ARCH_PARAMS["dropout_dense"]

    batch_size = Config.BATCH_SIZE
    lr = Config.LEARNING_RATE
    weight_decay = Config.WEIGHT_DECAY
    epochs = Config.EPOCHS
    patience = Config.PATIENCE

    # DataLoaders
    train_dataset = PizzaDataset(X_sem, X_rel, X_comm, y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = None
    if X_val_sem is not None:
        val_dataset = PizzaDataset(X_val_sem, X_val_rel, X_val_comm, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

    # Model Initialization
    model = OrthogonalSkipGatedMLP(
        sem_dim,
        rel_dim,
        comm_dim,
        hidden_dim,
        gate_hidden_dim,
        dropout_emb,
        dropout_dense,
    ).to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting MLP training on {Config.DEVICE}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sem_batch, rel_batch, comm_batch, y_batch in train_loader:
            sem_batch = sem_batch.to(Config.DEVICE)
            rel_batch = rel_batch.to(Config.DEVICE)
            comm_batch = comm_batch.to(Config.DEVICE)
            y_batch = y_batch.to(Config.DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(sem_batch, rel_batch, comm_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sem_batch.size(0)

        train_loss /= len(train_dataset)

        # Validation
        if val_loader:
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0

            with torch.no_grad():
                for sem_batch, rel_batch, comm_batch, y_batch in val_loader:
                    sem_batch = sem_batch.to(Config.DEVICE)
                    rel_batch = rel_batch.to(Config.DEVICE)
                    comm_batch = comm_batch.to(Config.DEVICE)
                    y_batch = y_batch.to(Config.DEVICE).unsqueeze(1)

                    logits = model(sem_batch, rel_batch, comm_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * sem_batch.size(0)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(y_batch.cpu().numpy())

            val_loss /= len(val_dataset)
            val_auc = compute_auc(val_targets, val_preds)

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}")
            best_model_state = model.state_dict()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    if save_path is None:
        save_path = os.path.join(Config.WORKING_DIR, "best_mlp.pth")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, {"auc": best_auc}


def predict_mlp(model, X_sem, X_rel, X_comm):
    """
    Generates predictions using the trained MLP model.
    """
    model.eval()
    dataset = PizzaDataset(X_sem, X_rel, X_comm, None)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_probs = []

    with torch.no_grad():
        for sem_batch, rel_batch, comm_batch in loader:
            sem_batch = sem_batch.to(Config.DEVICE)
            rel_batch = rel_batch.to(Config.DEVICE)
            comm_batch = comm_batch.to(Config.DEVICE)

            logits = model(sem_batch, rel_batch, comm_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs).flatten()


def run_mlp_pipeline(processed_data):
    """
    Orchestrates the MLP pipeline using the dictionary output from FeatureProcessor.
    """
    # Unpack Data
    X_train_sem = processed_data["train_mlp_sem"]
    X_train_rel = processed_data["train_mlp_rel"]
    X_train_comm = processed_data["train_mlp_comm"]
    y_train = processed_data["train_y"]

    X_val_sem = processed_data["val_mlp_sem"]
    X_val_rel = processed_data["val_mlp_rel"]
    X_val_comm = processed_data["val_mlp_comm"]
    y_val = processed_data["val_y"] if "val_y" in processed_data else None

    X_test_sem = processed_data["test_mlp_sem"]
    X_test_rel = processed_data["test_mlp_rel"]
    X_test_comm = processed_data["test_mlp_comm"]

    # Train
    model, metrics = train_mlp(
        X_train_sem,
        X_train_rel,
        X_train_comm,
        y_train,
        X_val_sem,
        X_val_rel,
        X_val_comm,
        y_val,
    )

    # Inference
    val_probs = None
    if X_val_sem is not None:
        val_probs = predict_mlp(model, X_val_sem, X_val_rel, X_val_comm)

    test_probs = predict_mlp(model, X_test_sem, X_test_rel, X_test_comm)

    return model, val_probs, test_probs
