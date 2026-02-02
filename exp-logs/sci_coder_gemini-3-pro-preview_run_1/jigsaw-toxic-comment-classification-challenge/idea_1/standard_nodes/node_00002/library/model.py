import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, load_embeddings
from library.data_loader import get_dataloaders


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops entire channels (embedding dimensions) across the sequence.
    """

    def __init__(self, p=0.3):
        super(SpatialDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        if not self.training or self.p == 0:
            return x

        # Create a mask of shape (batch_size, 1, embed_dim)
        # We sample a Bernoulli mask for the embedding dimensions and broadcast it across the sequence length.
        mask = (
            x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p).div_(1 - self.p)
        )
        return x * mask


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        scores = self.attn(x).squeeze(2)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq_len, 1)
        return (x * weights).sum(dim=1)


class BiGRU_Pool_Net(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim,
        embedding_matrix=None,
        dropout=0.3,
    ):
        super(BiGRU_Pool_Net, self).__init__()

        # Initialize Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Load pre-trained weights if provided
        if embedding_matrix is not None:
            # Ensure the matrix is a float tensor
            weights = torch.tensor(embedding_matrix, dtype=torch.float32)

            # Handle potential shape mismatch between vocab_size and matrix
            # If matrix is smaller/larger, we create a new one and copy what fits
            if weights.size(0) != vocab_size:
                new_weights = torch.normal(0, 1, (vocab_size, embed_dim))
                min_rows = min(vocab_size, weights.size(0))
                new_weights[:min_rows] = weights[:min_rows]
                weights = new_weights

            self.embedding.weight = nn.Parameter(weights)
            # Allow fine-tuning of embeddings
            self.embedding.weight.requires_grad = True

        self.spatial_dropout = SpatialDropout(dropout)

        # Bidirectional GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Attention Layer
        self.attention = Attention(hidden_dim * 2)

        # Concatenated pooling: (Attention + Max) * 2 (Bidirectional)
        linear_input_dim = hidden_dim * 2 * 2

        self.linear1 = nn.Linear(linear_input_dim, linear_input_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_input_dim // 2, output_dim)

    def forward(self, x):
        # x: (batch, seq_len)

        # Embedding & Spatial Dropout
        h_emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        h_emb = self.spatial_dropout(h_emb)

        # GRU
        h_gru, _ = self.gru(h_emb)  # (batch, seq_len, hidden_dim * 2)

        # Pooling
        # Attention Pooling
        attn_pool = self.attention(h_gru)  # (batch, hidden_dim * 2)
        # Global Max Pooling (Cite solution_lesson_node_00001)
        # torch.max returns (values, indices), we need values
        max_pool, _ = torch.max(h_gru, 1)  # (batch, hidden_dim * 2)

        # Concatenate
        pool_concat = torch.cat((attn_pool, max_pool), 1)  # (batch, hidden_dim * 4)

        # Classification Head
        x = self.linear1(pool_concat)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions and targets for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate Mean Column-wise ROC AUC
    auc_scores = []
    for i in range(all_targets.shape[1]):
        try:
            # Handle edge cases where a batch might have only one class present (though unlikely in full epoch)
            if len(np.unique(all_targets[:, i])) > 1:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
            else:
                score = 0.5
            auc_scores.append(score)
        except ValueError:
            auc_scores.append(0.5)

    epoch_auc = np.mean(auc_scores)
    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc_scores = []
    for i in range(all_targets.shape[1]):
        try:
            if len(np.unique(all_targets[:, i])) > 1:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
            else:
                score = 0.5
            auc_scores.append(score)
        except ValueError:
            auc_scores.append(0.5)

    epoch_auc = np.mean(auc_scores)
    return epoch_loss, epoch_auc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def run_pipeline(debug=Config.DEBUG):
    """
    Orchestrates the training and submission generation process.
    """
    seed_everything(Config.SEED)

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, word_index = get_dataloaders(debug=debug)

    # Define vocab_size. Config.MAX_FEATURES is the upper bound.
    vocab_size = Config.MAX_FEATURES

    print("Loading Embeddings...")
    # Attempt to load embeddings if a file exists (e.g., GloVe)
    # We check common paths or default to None (random init)
    embedding_path = os.path.join(Config.INPUT_DIR, "glove.840B.300d.txt")
    if not os.path.exists(embedding_path):
        # Fallback or check other common names if needed, or just proceed with random init
        embedding_path = None

    embedding_matrix = load_embeddings(embedding_path, word_index, Config.EMBED_DIM)

    print("Initializing Model...")
    device = Config.DEVICE
    model = BiGRU_Pool_Net(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.NUM_CLASSES,
        embedding_matrix=embedding_matrix,
        dropout=Config.DROPOUT,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Time: {elapsed:.0f}s - "
            f"Train Loss: {train_loss} - Train AUC: {train_auc} - "
            f"Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Best Validation AUC: {best_val_auc}")

    # Prediction on Test Set
    print("Generating Predictions...")
    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    test_preds = predict(model, test_loader, device)

    # Create Submission File
    submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    submission[Config.LABEL_COLS] = test_preds

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
