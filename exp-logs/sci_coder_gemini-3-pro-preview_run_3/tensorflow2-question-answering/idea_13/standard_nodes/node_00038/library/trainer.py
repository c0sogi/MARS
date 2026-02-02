import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from library.config import Config
from library.ranker_model import DIPNRanker
from library.reader_model import QIRNReader
from library.text_utils import get_vocab_and_matrix
from library.data_loader import get_data_loaders


class ModelTrainer:
    """
    Orchestrates the training of the Ranker and Reader models.
    """

    def __init__(self):
        self.device = Config.DEVICE
        # Ensure working directory exists for saving models
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def train_ranker(self, train_loader, val_loader, embedding_matrix):
        """
        Trains the DIPNRanker model.
        """
        print("\n--- Initializing Ranker (DIPN) ---")
        model = DIPNRanker(embedding_matrix).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting Ranker Training...")
        for epoch in range(Config.NUM_EPOCHS):
            # Training Phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                q_indices = batch[0].to(self.device)
                p_indices = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                optimizer.zero_grad()
                logits = model(q_indices, p_indices)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = train_loss / total if total > 0 else 0.0
            train_acc = correct / total if total > 0 else 0.0

            # Validation Phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    q_indices = batch[0].to(self.device)
                    p_indices = batch[1].to(self.device)
                    labels = batch[2].to(self.device)

                    logits = model(q_indices, p_indices)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            avg_val_loss = val_loss / total_val if total_val > 0 else 0.0
            val_acc = correct_val / total_val if total_val > 0 else 0.0

            print(
                f"Ranker Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {avg_train_loss}, Train Acc: {train_acc}, "
                f"Val Loss: {avg_val_loss}, Val Acc: {val_acc}"
            )

            # Checkpoint & Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
                print(f"Ranker model saved to {Config.RANKER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered for Ranker.")
                    break

        return model

    def train_reader(self, train_loader, val_loader, embedding_matrix):
        """
        Trains the QIRNReader model.
        """
        print("\n--- Initializing Reader (QIRN) ---")
        model = QIRNReader(embedding_matrix).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting Reader Training...")
        for epoch in range(Config.NUM_EPOCHS):
            # Training Phase
            model.train()
            train_loss = 0.0
            total = 0

            for batch in train_loader:
                q_indices = batch[0].to(self.device)
                p_indices = batch[1].to(self.device)
                start_targets = batch[2].to(self.device)
                end_targets = batch[3].to(self.device)

                optimizer.zero_grad()
                start_logits, end_logits = model(q_indices, p_indices)

                # Calculate loss for both start and end tokens
                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                loss = loss_start + loss_end

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * q_indices.size(0)
                total += q_indices.size(0)

            avg_train_loss = train_loss / total if total > 0 else 0.0

            # Validation Phase
            model.eval()
            val_loss = 0.0
            total_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    q_indices = batch[0].to(self.device)
                    p_indices = batch[1].to(self.device)
                    start_targets = batch[2].to(self.device)
                    end_targets = batch[3].to(self.device)

                    start_logits, end_logits = model(q_indices, p_indices)

                    loss_start = criterion(start_logits, start_targets)
                    loss_end = criterion(end_logits, end_targets)
                    loss = loss_start + loss_end

                    val_loss += loss.item() * q_indices.size(0)
                    total_val += q_indices.size(0)

            avg_val_loss = val_loss / total_val if total_val > 0 else 0.0

            print(
                f"Reader Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}"
            )

            # Checkpoint & Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.READER_MODEL_PATH)
                print(f"Reader model saved to {Config.READER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered for Reader.")
                    break

        return model

    def run(self, load_cached_data=True):
        """
        Orchestrates the loading of data and training of both models.

        Args:
            load_cached_data (bool): If True, attempts to load processed data and embeddings from disk.
        """
        print("Preparing Vocabulary and Embeddings...")
        # Note: In this environment, we assume texts are not needed if cache exists.
        # If cache is missing, this call might fail if texts=None, but we rely on pre-existing cache
        # or external handling of raw text reading for vocab building in the main pipeline.
        try:
            vocab, embedding_matrix = get_vocab_and_matrix(
                texts=None, load_cached_data=load_cached_data
            )
        except ValueError as e:
            print(
                f"Error loading vocab: {e}. Ensure vocab.parquet exists or text is provided."
            )
            return

        print("Loading DataLoaders...")
        loaders = get_data_loaders(vocab, load_cached_data=load_cached_data)
        (
            ranker_train_loader,
            ranker_val_loader,
            reader_train_loader,
            reader_val_loader,
        ) = loaders

        # Train Ranker
        self.train_ranker(ranker_train_loader, ranker_val_loader, embedding_matrix)

        # Train Reader
        self.train_reader(reader_train_loader, reader_val_loader, embedding_matrix)

        print("\nAll training tasks completed.")
