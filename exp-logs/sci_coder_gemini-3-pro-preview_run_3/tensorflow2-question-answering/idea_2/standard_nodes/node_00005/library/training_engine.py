import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from library.config import Config
from library.datasets import prepare_ranker_data, prepare_reader_data
from library.networks import SiameseTextCNN, AttentionMLPReader

# Set fixed random seed for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_ranker(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    load_cached_data=True,
    sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Trains the SiameseTextCNN Ranker model.

    Args:
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        sample_size (int or None): Number of samples to use (for debugging).

    Returns:
        SiameseTextCNN: The trained model with the best validation weights loaded.
    """
    print(f"Starting Ranker training on device: {get_device()}")

    # 1. Prepare Data
    train_dataset = prepare_ranker_data(
        split="train", load_cached_data=load_cached_data, sample_size=sample_size
    )
    val_dataset = prepare_ranker_data(
        split="val", load_cached_data=load_cached_data, sample_size=sample_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2. Initialize Model
    model = SiameseTextCNN().to(get_device())

    # 3. Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training Ranker for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            question = batch["question"].to(get_device())
            paragraph = batch["paragraph"].to(get_device())
            labels = batch["label"].to(get_device())

            optimizer.zero_grad()
            outputs = model(question, paragraph)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                question = batch["question"].to(get_device())
                paragraph = batch["paragraph"].to(get_device())
                labels = batch["label"].to(get_device())

                outputs = model(question, paragraph)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        epoch_duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {avg_val_loss} | Val Acc: {val_acc} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
            print(
                f"Validation loss improved. Model saved to {Config.RANKER_MODEL_PATH}"
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model
    if os.path.exists(Config.RANKER_MODEL_PATH):
        model.load_state_dict(torch.load(Config.RANKER_MODEL_PATH))

    return model


def train_reader(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    load_cached_data=True,
    sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Trains the AttentionMLPReader model.

    Args:
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        sample_size (int or None): Number of samples to use (for debugging).

    Returns:
        AttentionMLPReader: The trained model with the best validation weights loaded.
    """
    print(f"Starting Reader training on device: {get_device()}")

    # 1. Prepare Data
    train_dataset = prepare_reader_data(
        split="train", load_cached_data=load_cached_data, sample_size=sample_size
    )
    val_dataset = prepare_reader_data(
        split="val", load_cached_data=load_cached_data, sample_size=sample_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2. Initialize Model
    model = AttentionMLPReader().to(get_device())

    # 3. Setup Optimization
    # We use CrossEntropyLoss which expects class indices (0 to L-1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training Reader for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            question = batch["question"].to(get_device())
            paragraph = batch["paragraph"].to(get_device())
            start_targets = batch["start_idx"].to(get_device())
            end_targets = batch["end_idx"].to(get_device())

            optimizer.zero_grad()
            start_logits, end_logits = model(question, paragraph)

            # Calculate loss for both start and end positions
            loss_start = criterion(start_logits, start_targets)
            loss_end = criterion(end_logits, end_targets)
            loss = loss_start + loss_end

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * start_targets.size(0)

            # Calculate Exact Match Accuracy
            # Correct if both start and end predictions match targets
            pred_start = torch.argmax(start_logits, dim=1)
            pred_end = torch.argmax(end_logits, dim=1)

            match = (pred_start == start_targets) & (pred_end == end_targets)
            correct_train += match.sum().item()
            total_train += start_targets.size(0)

        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                question = batch["question"].to(get_device())
                paragraph = batch["paragraph"].to(get_device())
                start_targets = batch["start_idx"].to(get_device())
                end_targets = batch["end_idx"].to(get_device())

                start_logits, end_logits = model(question, paragraph)

                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                loss = loss_start + loss_end

                val_loss += loss.item() * start_targets.size(0)

                pred_start = torch.argmax(start_logits, dim=1)
                pred_end = torch.argmax(end_logits, dim=1)

                match = (pred_start == start_targets) & (pred_end == end_targets)
                correct_val += match.sum().item()
                total_val += start_targets.size(0)

        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        epoch_duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss} | Train EM Acc: {train_acc} | "
            f"Val Loss: {avg_val_loss} | Val EM Acc: {val_acc} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.READER_MODEL_PATH)
            print(
                f"Validation loss improved. Model saved to {Config.READER_MODEL_PATH}"
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model
    if os.path.exists(Config.READER_MODEL_PATH):
        model.load_state_dict(torch.load(Config.READER_MODEL_PATH))

    return model
