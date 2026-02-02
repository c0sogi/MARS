import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    EMBED_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY,
    CACHE_DIR,
)


class Attention(nn.Module):
    """
    Attention mechanism for aggregating LSTM states.
    """

    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # lstm_output: (batch, seq_len, hidden_dim)
        # mask: (batch, seq_len)

        # Calculate attention energies
        energy = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Mask padding positions with a large negative value
        energy = energy.masked_fill(mask == 0, -1e9)

        # Calculate weights
        weights = torch.softmax(energy, dim=1)  # (batch, seq_len)

        # Weighted sum of LSTM outputs
        # weights: (batch, 1, seq_len)
        # lstm_output: (batch, seq_len, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(
            1
        )  # (batch, hidden_dim)

        return context


class InsultModel(nn.Module):
    """
    Bidirectional LSTM classifier with Attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
    ):
        super(InsultModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.2,
        )
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # text: (batch, seq_len)
        embedded = self.embedding(text)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Create mask for attention (0 is padding_idx)
        mask = text != 0

        # Apply Attention
        attn_output = self.attention(output, mask)

        x = self.dropout(attn_output)
        x = self.fc(x)
        return self.sigmoid(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for text, lengths, labels in dataloader:
        text = text.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(text, lengths)
        # Flatten outputs to match labels shape (batch_size,)
        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        # Store predictions and labels for AUC calculation
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate AUC, handling potential edge cases (e.g., only one class in batch)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, lengths, labels in dataloader:
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(text, lengths)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str = DEVICE,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    save_path: str = os.path.join(CACHE_DIR, "best_model.pth"),
):
    """
    Main training loop with Early Stopping.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    # Adam optimizer works well with EmbeddingBag sparse gradients
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Print metrics without rounding as requested
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss} Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss} Val AUC: {val_auc}")

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save best model
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Training complete. Best Val AUC: {best_val_auc}")

    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str = DEVICE
) -> np.ndarray:
    """
    Generates predictions for the given dataloader.
    Returns:
        np.ndarray: Array of probability scores.
    """
    model.eval()
    model = model.to(device)
    predictions = []

    with torch.no_grad():
        for text, lengths, _ in dataloader:
            text = text.to(device)

            outputs = model(text, lengths)
            outputs = outputs.view(-1)

            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)
