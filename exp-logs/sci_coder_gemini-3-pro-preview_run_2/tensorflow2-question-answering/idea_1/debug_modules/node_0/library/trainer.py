import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

from library.config import Config
from library.data_utils import build_tokenizer, collate_fn
from library.dataset import NQTrainDataset
from library.embeddings import create_embedding_matrix
from library.model import BoERanker


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(load_cached_data=True, limit=None):
    """
    Trains the Neural BoE Ranker model.

    Args:
        load_cached_data (bool): Whether to use cached artifacts (vocab, embeddings, offsets).
        limit (int, optional): Limit the number of samples for debugging.

    Returns:
        BoERanker: The trained model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data Artifacts
    print("Preparing tokenizer and embeddings...")
    tokenizer = build_tokenizer(
        load_cached_data=load_cached_data,
        data_path=Config.TRAIN_DATA_PATH,
        sample_size=limit,
    )

    # We don't have a specific GloVe path in Config provided in the prompt,
    # so we pass None to initialize randomly or rely on what create_embedding_matrix does.
    # Assuming standard behavior where if path is None it initializes randomly.
    embedding_matrix = create_embedding_matrix(
        tokenizer,
        glove_path=None,
        embedding_dim=Config.EMBEDDING_DIM,
        load_cached_data=load_cached_data,
    )

    # 2. Initialize Model
    print("Initializing model...")
    model = BoERanker(embedding_matrix)
    model.to(device)

    # 3. Prepare Datasets and Loaders
    print("Preparing datasets...")
    train_dataset = NQTrainDataset(
        metadata_path=Config.TRAIN_META_PATH,
        data_path=Config.TRAIN_DATA_PATH,
        tokenizer=tokenizer,
        limit=limit,
        load_cached_data=load_cached_data,
    )

    val_dataset = NQTrainDataset(
        metadata_path=Config.VAL_META_PATH,
        data_path=Config.TRAIN_DATA_PATH,
        tokenizer=tokenizer,
        limit=limit,
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Simple single-process loading to avoid complexity
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 4. Setup Optimization
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if batch is None:
                continue

            q_seqs = batch["q_seqs"].to(device)
            c_seqs = batch["c_seqs"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(q_seqs, c_seqs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            # Accuracy calculation
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                q_seqs = batch["q_seqs"].to(device)
                c_seqs = batch["c_seqs"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(q_seqs, c_seqs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val Acc: {val_acc}")

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"Validation loss improved. Saving model to {Config.MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{Config.PATIENCE}"
            )
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")

    # Load best model state before returning
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    return model
