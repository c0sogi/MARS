import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Computes a weighted average of sequence embeddings using a learnable attention mechanism.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        # x: (batch, seq, dim)
        # mask: (batch, seq) - 1 for valid, 0 for pad
        scores = self.attention_weights(x)  # (batch, seq, 1)

        if mask is not None:
            # Mask padding by setting scores to -inf
            # Ensure we don't mask everything (avoid NaN), though data should be valid.
            # We unsqueeze mask to match scores shape
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -float("inf"))

        weights = self.softmax(scores)  # (batch, seq, 1)

        # Weighted sum
        # If all scores were -inf (empty seq), weights would be nan.
        # But we assume valid inputs or handle it by ensuring mask has at least one 1.
        context = torch.sum(weights * x, dim=1)  # (batch, dim)
        return context


class DualAveNet(nn.Module):
    """
    Dual-Stream Network with Attention Pooling.

    Architecture:
    1. Independent Question and Answer streams sharing an Embedding layer.
    2. Attention Pooling to obtain weighted fixed-size representations.
    3. Interaction Layer: Concatenates Q, A, |Q-A|, and Q*A.
    4. MLP Head: Dense -> BN -> ReLU -> Dropout -> Dense -> Sigmoid.
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
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)

        # Attention Pooling Layers
        self.q_attention = AttentionPooling(self.embed_dim)
        self.a_attention = AttentionPooling(self.embed_dim)

        # Interaction Layer Dimension
        # We concatenate: q_pool, a_pool, abs(q-a), q*a
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
        """
        # Create masks (1 for valid, 0 for pad)
        # Ensure at least one token is valid to prevent NaNs in softmax
        q_mask = (q_indices != 0).float()
        q_mask[:, 0] = 1.0  # Force first token valid (pad embedding is 0 anyway)

        a_mask = (a_indices != 0).float()
        a_mask[:, 0] = 1.0

        # 1. Embedding
        q_emb = self.embedding(q_indices)  # (batch, seq, dim)
        a_emb = self.embedding(a_indices)  # (batch, seq, dim)

        # 2. Attention Pooling
        q_pool = self.q_attention(q_emb, q_mask)  # (batch, dim)
        a_pool = self.a_attention(a_emb, a_mask)  # (batch, dim)

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
