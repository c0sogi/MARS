import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.models import EarlyFusionRanker, DynamicKernelReader
from library.data_utils import (
    get_ranker_datasets,
    get_reader_datasets,
    load_embeddings,
    Vocabulary,
)


class Trainer:
    """
    Handles the training loops for the Early-Fusion Ranker and Dynamic Kernel Reader.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(Config.SEED)
        Config.ensure_directories()

    def set_seed(self, seed):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_embeddings(self, load_cached_data):
        """Helper to load vocabulary and embeddings."""
        vocab = Vocabulary()
        # Ensure vocab exists (prepare_data in data_utils handles this, but we load here)
        if os.path.exists(Config.VOCAB_PATH):
            vocab.load(Config.VOCAB_PATH)
        else:
            # Fallback if not yet created, though get_datasets usually handles it
            print("Vocabulary not found during model init, ensure data is prepared.")
            return None

        embeddings = load_embeddings(
            vocab, Config.EMBEDDING_DIM, load_cached_data=load_cached_data
        )
        return embeddings

    def train_ranker(self, load_cached_data=True):
        """
        Trains the EarlyFusionRanker model.
        """
        print("Initializing Ranker Training...")

        # Load Data
        train_dataset, val_dataset = get_ranker_datasets(
            load_cached_data=load_cached_data
        )
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Initialize Model
        embeddings = self._get_embeddings(load_cached_data)
        model = EarlyFusionRanker(embedding_matrix=embeddings).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * input_ids.size(0)

                # Accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(input_ids)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * input_ids.size(0)

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {avg_train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {avg_val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
                print(f"New best model saved to {Config.RANKER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print("Ranker training completed.")

    def train_reader(self, load_cached_data=True):
        """
        Trains the DynamicKernelReader model.
        """
        print("Initializing Reader Training...")

        # Load Data
        train_dataset, val_dataset = get_reader_datasets(
            load_cached_data=load_cached_data
        )
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Initialize Model
        embeddings = self._get_embeddings(load_cached_data)
        model = DynamicKernelReader(embedding_matrix=embeddings).to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            # Training Phase
            model.train()
            train_loss = 0.0
            train_correct_em = 0  # Exact Match
            train_total = 0

            for batch in train_loader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                ctx_input_ids = batch["ctx_input_ids"].to(self.device)
                start_targets = batch["start_idx"].to(self.device)
                end_targets = batch["end_idx"].to(self.device)

                optimizer.zero_grad()
                start_logits, end_logits = model(q_input_ids, ctx_input_ids)

                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                loss = loss_start + loss_end

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * q_input_ids.size(0)

                # Exact Match Calculation
                pred_start = torch.argmax(start_logits, dim=1)
                pred_end = torch.argmax(end_logits, dim=1)
                exact_matches = (
                    (pred_start == start_targets) & (pred_end == end_targets)
                ).float()
                train_correct_em += exact_matches.sum().item()
                train_total += q_input_ids.size(0)

            avg_train_loss = train_loss / train_total
            train_em = train_correct_em / train_total

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct_em = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    q_input_ids = batch["q_input_ids"].to(self.device)
                    ctx_input_ids = batch["ctx_input_ids"].to(self.device)
                    start_targets = batch["start_idx"].to(self.device)
                    end_targets = batch["end_idx"].to(self.device)

                    start_logits, end_logits = model(q_input_ids, ctx_input_ids)

                    loss_start = criterion(start_logits, start_targets)
                    loss_end = criterion(end_logits, end_targets)
                    loss = loss_start + loss_end

                    val_loss += loss.item() * q_input_ids.size(0)

                    pred_start = torch.argmax(start_logits, dim=1)
                    pred_end = torch.argmax(end_logits, dim=1)
                    exact_matches = (
                        (pred_start == start_targets) & (pred_end == end_targets)
                    ).float()
                    val_correct_em += exact_matches.sum().item()
                    val_total += q_input_ids.size(0)

            avg_val_loss = val_loss / val_total
            val_em = val_correct_em / val_total

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {avg_train_loss} | Train EM: {train_em} | "
                f"Val Loss: {avg_val_loss} | Val EM: {val_em}"
            )

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.READER_MODEL_PATH)
                print(f"New best model saved to {Config.READER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print("Reader training completed.")
