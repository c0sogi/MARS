import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.data_utils import Tokenizer, build_embedding_matrix
from library.dataset import NQDataset
from library.model import GlobalContextPointwiseNet, calculate_loss


class Trainer:
    def __init__(self, load_cached_data=True):
        """
        Initializes the Trainer with model, optimizer, and resources.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # Load Tokenizer
        self.tokenizer = Tokenizer()
        if os.path.exists(Config.VOCAB_CACHE_FILE) and load_cached_data:
            self.tokenizer.load(Config.VOCAB_CACHE_FILE)
        else:
            # If vocab doesn't exist, we assume the dataset processing will handle it
            # or it should have been pre-calculated. For this module, we warn.
            print(
                f"Warning: Vocab file not found at {Config.VOCAB_CACHE_FILE}. Model may not initialize correctly if not built."
            )

        # Build/Load Embeddings
        self.embedding_matrix = build_embedding_matrix(
            self.tokenizer.word_index,
            embedding_dim=Config.EMBEDDING_DIM,
            load_cached_data=load_cached_data,
        )

        # Initialize Model
        self.model = GlobalContextPointwiseNet(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            dropout_rate=Config.DROPOUT_RATE,
            embedding_matrix=self.embedding_matrix,
        ).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def fit(self, load_cached_data=True):
        """
        Executes the training loop with validation and early stopping.
        """
        # Prepare DataLoaders
        print("Preparing datasets...")
        train_dataset = NQDataset(
            metadata_path=Config.TRAIN_META_PATH,
            raw_data_path=Config.TRAIN_DATA_PATH,
            tokenizer=self.tokenizer,
            is_train=True,
            load_cached_data=load_cached_data,
        )
        val_dataset = NQDataset(
            metadata_path=Config.VAL_META_PATH,
            raw_data_path=Config.TRAIN_DATA_PATH,
            tokenizer=self.tokenizer,
            is_train=False,
            load_cached_data=load_cached_data,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )

        # Training State
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Fix for Stale Checkpoints (Cite debug_lesson_7)
        if os.path.exists(best_model_path):
            print(f"Removing stale checkpoint: {best_model_path}")
            os.remove(best_model_path)

        print("Starting training...")

        for epoch in range(Config.NUM_EPOCHS):
            # --- Training Phase ---
            self.model.train()
            train_loss_sum = 0.0
            num_train_batches = len(train_loader)

            for batch in train_loader:
                # Move batch to device
                q_seq = batch["q_seq"].to(self.device)
                c_seq = batch["c_seq"].to(self.device)
                long_labels = batch["long_label"].to(self.device)
                start_labels = batch["short_start"].to(self.device)
                end_labels = batch["short_end"].to(self.device)
                yn_labels = batch["yes_no_label"].to(self.device)

                # Forward Pass
                self.optimizer.zero_grad()
                l_logits, s_logits, e_logits, yn_logits = self.model(q_seq, c_seq)

                # Loss Calculation
                loss, _, _, _ = calculate_loss(
                    l_logits,
                    s_logits,
                    e_logits,
                    yn_logits,
                    long_labels,
                    start_labels,
                    end_labels,
                    yn_labels,
                )

                # Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / num_train_batches

            # --- Validation Phase ---
            self.model.eval()
            val_loss_sum = 0.0
            correct_long = 0
            total_long = 0
            num_val_batches = len(val_loader)

            with torch.no_grad():
                for batch in val_loader:
                    q_seq = batch["q_seq"].to(self.device)
                    c_seq = batch["c_seq"].to(self.device)
                    long_labels = batch["long_label"].to(self.device)
                    start_labels = batch["short_start"].to(self.device)
                    end_labels = batch["short_end"].to(self.device)
                    yn_labels = batch["yes_no_label"].to(self.device)

                    l_logits, s_logits, e_logits, yn_logits = self.model(q_seq, c_seq)

                    loss, _, _, _ = calculate_loss(
                        l_logits,
                        s_logits,
                        e_logits,
                        yn_logits,
                        long_labels,
                        start_labels,
                        end_labels,
                        yn_labels,
                    )
                    val_loss_sum += loss.item()

                    # Calculate Long Answer Accuracy (Binary Classification)
                    preds = (torch.sigmoid(l_logits) > 0.5).float()
                    correct_long += (preds == long_labels).sum().item()
                    total_long += long_labels.size(0)

            avg_val_loss = val_loss_sum / num_val_batches
            val_acc = correct_long / total_long if total_long > 0 else 0.0

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {avg_train_loss} - "
                f"Val Loss: {avg_val_loss} - "
                f"Val Long Acc: {val_acc}"
            )

            # --- Early Stopping & Checkpointing ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                # Not printing "Saved model" to keep logs clean as per instructions,
                # but logic is here.
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break
