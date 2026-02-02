import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, save_submission
from library.dataset import get_datasets


class BiGRUClassifier(nn.Module):
    """
    Bi-directional Recurrent Spectral Network for Right Whale Detection.

    Architecture:
    1. Input: Log-Mel Spectrogram (Batch, 1, F, T)
    2. Reshape: (Batch, T, F)
    3. Bi-Directional GRU: Captures temporal context in both directions.
    4. Global Max Pooling: Extracts the strongest activation across time.
    5. Fully Connected Layer: Maps features to logit score.
    """

    def __init__(self):
        super(BiGRUClassifier, self).__init__()

        # GRU Layer
        # Input size is the number of Mel frequency bins (F)
        self.gru = nn.GRU(
            input_size=Config.N_MELS,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0,
        )

        # Classification Head
        # Input dim is Hidden Size * 2 (for bidirectional)
        self.fc = nn.Linear(Config.HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, 1, F, T)
        Returns:
            logits (Tensor): Output logits of shape (Batch, 1)
        """
        # Remove channel dimension: (Batch, 1, F, T) -> (Batch, F, T)
        x = x.squeeze(1)

        # Permute to (Batch, Time, Features) for GRU
        # T is sequence length, F is input size
        x = x.permute(0, 2, 1)

        # Pass through Bi-GRU
        # out shape: (Batch, T, Hidden_Size * 2)
        out, _ = self.gru(x)

        # Global Max Pooling over the time dimension (dim 1)
        # This makes the model invariant to the position of the call in the clip
        # out shape: (Batch, Hidden_Size * 2)
        out, _ = torch.max(out, dim=1)

        # Classification layer
        logits = self.fc(out)

        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Match shape (Batch, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in the batch/epoch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training(load_cached_data=True, limit=None):
    """
    Main function to run the training pipeline.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
        limit (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # get_datasets handles caching and normalization internally
    train_dataset, val_dataset, test_dataset = get_datasets(
        load_cached_data=load_cached_data, limit=limit
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 3. Model Initialization
    model = BiGRUClassifier().to(device)

    # 4. Loss and Optimizer
    # Handle class imbalance with pos_weight
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"Validation AUC improved. Model saved to {Config.MODEL_CHECKPOINT}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # 6. Inference on Test Set
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    predictions = []
    test_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, clip_ids in test_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            test_ids.extend(clip_ids)

    # 7. Save Submission
    save_submission(predictions, test_ids)
