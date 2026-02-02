import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os

from library.config import Config
from library.text_utils import Vocab
from library.ranker_net import InteractionRanker
from library.reader_net import UNetReader
from library.data_loader import get_ranker_loaders, get_reader_loaders, load_vocab


def train_ranker(
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    train_sample_size=None,
    val_sample_size=None,
):
    """
    Trains the InteractionRanker model using Binary Cross-Entropy loss.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        train_sample_size (int, optional): Number of training samples to use (for debugging).
        val_sample_size (int, optional): Number of validation samples to use.

    Returns:
        model: The trained InteractionRanker model.
    """
    print(
        f"Starting Ranker training: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}"
    )

    # Load Vocabulary
    vocab = load_vocab()

    # Get DataLoaders
    train_loader, val_loader = get_ranker_loaders(
        batch_size=batch_size,
        train_sample_size=train_sample_size,
        val_sample_size=val_sample_size,
    )

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionRanker(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_embeddings=vocab.embedding_matrix,
    ).to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            q_ids = batch["q_ids"].to(device)
            p_ids = batch["p_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(q_ids, p_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * q_ids.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["q_ids"].to(device)
                p_ids = batch["p_ids"].to(device)
                labels = batch["label"].to(device)

                outputs = model(q_ids, p_ids)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * q_ids.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Metrics
        preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
        val_acc = accuracy_score(all_labels, preds_binary)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss} - Val Loss: {epoch_val_loss} - Val Acc: {val_acc}"
        )

        # Early Stopping and Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model


def train_reader(
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    train_sample_size=None,
    val_sample_size=None,
):
    """
    Trains the UNetReader model using Categorical Cross-Entropy loss.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        train_sample_size (int, optional): Number of training samples to use.
        val_sample_size (int, optional): Number of validation samples to use.

    Returns:
        model: The trained UNetReader model.
    """
    print(
        f"Starting Reader training: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}"
    )

    # Load Vocabulary
    vocab = load_vocab()

    # Get DataLoaders
    train_loader, val_loader = get_reader_loaders(
        batch_size=batch_size,
        train_sample_size=train_sample_size,
        val_sample_size=val_sample_size,
    )

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetReader(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_embeddings=vocab.embedding_matrix,
    ).to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            start_targets = batch["start_target"].to(device)
            end_targets = batch["end_target"].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids)

            loss = criterion(start_logits, start_targets) + criterion(
                end_logits, end_targets
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_start = 0
        correct_end = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                start_targets = batch["start_target"].to(device)
                end_targets = batch["end_target"].to(device)

                start_logits, end_logits = model(input_ids)
                loss = criterion(start_logits, start_targets) + criterion(
                    end_logits, end_targets
                )

                running_val_loss += loss.item()

                pred_start = torch.argmax(start_logits, dim=1)
                pred_end = torch.argmax(end_logits, dim=1)

                correct_start += (pred_start == start_targets).sum().item()
                correct_end += (pred_end == end_targets).sum().item()
                total_samples += input_ids.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_acc_start = correct_start / total_samples
        val_acc_end = correct_end / total_samples

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss} - Val Loss: {epoch_val_loss} - Start Acc: {val_acc_start} - End Acc: {val_acc_end}"
        )

        # Early Stopping and Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.READER_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model
