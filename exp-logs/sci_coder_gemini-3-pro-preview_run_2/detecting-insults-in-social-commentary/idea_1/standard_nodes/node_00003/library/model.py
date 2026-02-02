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


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM classifier.
    Captures sequential dependencies for insult detection.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        pad_idx: int = 0,
    ):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # FC layer: input is hidden_dim * 2 (forward + backward)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Final activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: (batch_size, seq_len)
            lengths: (batch_size)
        """
        # embedded: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(text)

        # Pack padded sequence for efficiency and ignoring padding
        # enforce_sorted=False allows unsorted batch
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        # output is packed, hidden is (num_layers*num_directions, batch, hidden_dim)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Concatenate the final forward and backward hidden states
        # hidden[-2, :, :] is the last hidden state of the forward LSTM
        # hidden[-1, :, :] is the last hidden state of the backward LSTM
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # FC and Sigmoid
        return self.sigmoid(self.fc(hidden))


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
        lengths = lengths.to(device)
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
            lengths = lengths.to(device)
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
            lengths = lengths.to(device)

            outputs = model(text, lengths)
            outputs = outputs.view(-1)

            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)
