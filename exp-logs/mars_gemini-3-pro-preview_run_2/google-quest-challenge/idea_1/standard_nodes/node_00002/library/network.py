import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config


class DualGRUNet(nn.Module):
    """
    Dual-Stream GRU Network.

    Architecture:
    1. Shared Embedding Layer.
    2. Bidirectional GRU for Q and A streams.
    3. Global Average Pooling (masked) on GRU outputs.
    4. Interaction Layer: Concatenates Q, A, |Q-A|, and Q*A.
    5. MLP Head.
    """

    def __init__(self):
        super(DualGRUNet, self).__init__()

        # Hyperparameters
        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_prob = Config.DROPOUT
        self.num_targets = len(Config.TARGET_COLS)

        # Shared Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Bidirectional GRU
        self.gru = nn.GRU(
            self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )

        # GRU output dimension (bidirectional)
        self.gru_out_dim = self.hidden_dim * 2

        # Interaction Layer Dimension
        # We concatenate: q_pool, a_pool, abs(q-a), q*a
        # Each has dimension gru_out_dim, so total is 4 * gru_out_dim
        self.input_dim = 4 * self.gru_out_dim

        # MLP Head
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_targets)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_indices, a_indices):
        # 1. Embedding
        q_emb = self.embedding(q_indices)  # (batch, seq, dim)
        a_emb = self.embedding(a_indices)  # (batch, seq, dim)

        # 2. GRU Encoding
        q_out, _ = self.gru(q_emb)  # (batch, seq, 2*hidden)
        a_out, _ = self.gru(a_emb)  # (batch, seq, 2*hidden)

        # 3. Masked Global Average Pooling
        # Create masks (batch, seq, 1)
        q_mask = (q_indices != 0).unsqueeze(2).float()
        a_mask = (a_indices != 0).unsqueeze(2).float()

        # Sum masked outputs and divide by valid lengths
        q_pool = (q_out * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-9)
        a_pool = (a_out * a_mask).sum(dim=1) / a_mask.sum(dim=1).clamp(min=1e-9)

        # 4. Interaction Layer
        diff = torch.abs(q_pool - a_pool)
        prod = q_pool * a_pool

        # Concatenate features
        features = torch.cat([q_pool, a_pool, diff, prod], dim=1)

        # 5. MLP Head
        x = self.fc1(features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = self.sigmoid(logits)

        return probs


class DualAveNet(nn.Module):
    """
    Dual-Stream Averaged Embedding Network (Dual-AveNet).
    Kept for reference or fallback.
    """

    def __init__(self):
        super(DualAveNet, self).__init__()

        # Hyperparameters
        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_prob = Config.DROPOUT
        self.num_targets = len(Config.TARGET_COLS)

        # Shared Embedding Layer
        # padding_idx=0 ensures the padding token vector remains 0
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Interaction Layer Dimension
        # We concatenate: q_pool, a_pool, abs(q-a), q*a
        # Each has dimension embed_dim, so total is 4 * embed_dim
        self.input_dim = 4 * self.embed_dim

        # MLP Head
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_targets)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_indices, a_indices):
        """
        Forward pass of the network.
        Args:
            q_indices: Tensor of shape (batch_size, seq_len)
            a_indices: Tensor of shape (batch_size, seq_len)
        Returns:
            probs: Tensor of shape (batch_size, num_targets) in range [0,1]
        """
        # 1. Embedding
        q_emb = self.embedding(q_indices)  # (batch, seq, dim)
        a_emb = self.embedding(a_indices)  # (batch, seq, dim)

        # 2. Global Average Pooling
        # Computes the semantic centroid.
        # Note: Simple mean includes padding zeros, effectively diluting the vector for shorter sequences.
        q_pool = q_emb.mean(dim=1)  # (batch, dim)
        a_pool = a_emb.mean(dim=1)  # (batch, dim)

        # 3. Interaction Layer
        diff = torch.abs(q_pool - a_pool)
        prod = q_pool * a_pool

        # Concatenate features
        features = torch.cat([q_pool, a_pool, diff, prod], dim=1)  # (batch, 4*dim)

        # 4. MLP Head
        x = self.fc1(features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = self.sigmoid(logits)

        return probs


def train_model(model, train_loader, val_loader):
    """
    Trains the DualAveNet model using the configuration specified in library.config.
    Implements Adam optimizer, BCE Loss, and Early Stopping.

    Args:
        model: Instance of DualAveNet
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data

    Returns:
        model: The trained model (with best weights loaded)
    """
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    device = Config.DEVICE
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Early Stopping tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for q, a, y in train_loader:
            q, a, y = q.to(device), a.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(q, a)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * q.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for q, a, y in val_loader:
                q, a, y = q.to(device), a.to(device), y.to(device)

                outputs = model(q, a)
                loss = criterion(outputs, y)
                val_loss += loss.item() * q.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict_submission(model, test_loader, test_ids):
    """
    Generates predictions for the test set and saves them to the submission file.

    Args:
        model: Trained DualAveNet model
        test_loader: DataLoader for test data
        test_ids: Array of qa_ids corresponding to the test set
    """
    device = Config.DEVICE
    model.to(device)
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for q, a, _ in test_loader:
            # Note: test_loader returns dummy targets, which we ignore
            q, a = q.to(device), a.to(device)
            outputs = model(q, a)
            all_preds.append(outputs.cpu().numpy())

    # Stack predictions
    predictions = np.vstack(all_preds)

    # Create submission DataFrame
    submission = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_ids)

    # Save to disk
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
