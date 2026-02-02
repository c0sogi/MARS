import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.configuration import Config
from library.text_utils import build_vocab
from library.data_loader import get_reader_loaders


class StackedBiGRUReader(nn.Module):
    """
    A Stacked Bi-Directional GRU model for Extractive Question Answering.

    Architecture:
    1. Embedding Layer: Maps token indices to dense vectors.
    2. Bi-GRU Layers: Processes the concatenated Question + Context sequence.
    3. Linear Heads: Two separate linear layers to predict start and end token logits
       from the Context portion of the GRU output.
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx
    ):
        super(StackedBiGRUReader, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Input to linear layers is hidden_dim * 2 because of bidirectionality
        self.start_fc = nn.Linear(hidden_dim * 2, 1)
        self.end_fc = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_indices, ctx_indices):
        """
        Args:
            q_indices: (batch_size, max_q_len)
            ctx_indices: (batch_size, max_ctx_len)

        Returns:
            start_logits: (batch_size, max_ctx_len)
            end_logits: (batch_size, max_ctx_len)
        """
        # 1. Embeddings
        q_emb = self.embedding(q_indices)  # (B, Q_Len, Emb)
        ctx_emb = self.embedding(ctx_indices)  # (B, Ctx_Len, Emb)

        # 2. Concatenate Question and Context along sequence dimension
        # Shape: (B, Q_Len + Ctx_Len, Emb)
        inputs = torch.cat([q_emb, ctx_emb], dim=1)
        inputs = self.dropout(inputs)

        # 3. Pass through GRU
        # outputs: (B, Seq_Len, Hidden*2)
        outputs, _ = self.gru(inputs)

        # 4. Slice out the Context part
        # We assume the context starts after Q_Len.
        # Note: This relies on fixed Q_Len padding from the dataset.
        q_len = q_indices.size(1)
        ctx_outputs = outputs[:, q_len:, :]  # (B, Ctx_Len, Hidden*2)

        # 5. Predict Start and End Logits
        # Squeeze the last dimension (size 1) after linear projection
        start_logits = self.start_fc(ctx_outputs).squeeze(-1)  # (B, Ctx_Len)
        end_logits = self.end_fc(ctx_outputs).squeeze(-1)  # (B, Ctx_Len)

        return start_logits, end_logits


class ReaderTrainer:
    """
    Manages the training and evaluation of the StackedBiGRUReader.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model_path = Config.READER_MODEL_PATH
        self.vocab = None
        self.model = None

    def load_vocab(self, load_cached_data=True):
        """
        Loads or builds the vocabulary.
        """
        self.vocab = build_vocab(
            load_cached_data=load_cached_data, cache_path=Config.VOCAB_CACHE_PATH
        )
        return self.vocab

    def train(self, load_cached_data=True, sample_size=None):
        """
        Main training loop with early stopping.
        """
        print("Initializing Reader Training...")

        # Ensure vocab is loaded
        if self.vocab is None:
            self.load_vocab(load_cached_data=load_cached_data)

        pad_idx = self.vocab[Config.PAD_TOKEN]

        # Initialize Model
        self.model = StackedBiGRUReader(
            vocab_size=len(self.vocab),
            embedding_dim=Config.READER_EMBEDDING_DIM,
            hidden_dim=Config.READER_HIDDEN_DIM,
            num_layers=Config.READER_NUM_LAYERS,
            dropout=Config.READER_DROPOUT,
            pad_idx=pad_idx,
        ).to(self.device)

        # Data Loaders
        train_loader, val_loader = get_reader_loaders(
            self.vocab,
            batch_size=Config.READER_BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            sample_size=sample_size,
        )

        # Optimization
        criterion = nn.CrossEntropyLoss(
            ignore_index=-1
        )  # -1 is not used here, but good practice
        optimizer = optim.Adam(self.model.parameters(), lr=Config.READER_LEARNING_RATE)

        # Early Stopping State
        best_val_loss = float("inf")
        patience_counter = 0

        print(
            f"Starting training on {self.device} for {Config.READER_EPOCHS} epochs..."
        )

        for epoch in range(Config.READER_EPOCHS):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                q, ctx, start_target, end_target = [b.to(self.device) for b in batch]

                optimizer.zero_grad()

                start_logits, end_logits = self.model(q, ctx)

                loss_start = criterion(start_logits, start_target)
                loss_end = criterion(end_logits, end_target)
                loss = loss_start + loss_end

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.READER_GRAD_CLIP
                )

                optimizer.step()

                train_loss += loss.item() * q.size(0)

                # Calculate simple accuracy (Exact Match of indices)
                pred_start = torch.argmax(start_logits, dim=1)
                pred_end = torch.argmax(end_logits, dim=1)
                train_correct += (
                    ((pred_start == start_target) & (pred_end == end_target))
                    .sum()
                    .item()
                )
                train_total += q.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    q, ctx, start_target, end_target = [
                        b.to(self.device) for b in batch
                    ]

                    start_logits, end_logits = self.model(q, ctx)

                    loss_start = criterion(start_logits, start_target)
                    loss_end = criterion(end_logits, end_target)
                    val_loss += (loss_start + loss_end).item() * q.size(0)

                    pred_start = torch.argmax(start_logits, dim=1)
                    pred_end = torch.argmax(end_logits, dim=1)
                    val_correct += (
                        ((pred_start == start_target) & (pred_end == end_target))
                        .sum()
                        .item()
                    )
                    val_total += q.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{Config.READER_EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | Train EM: {train_acc:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | Val EM: {val_acc:.6f}"
            )

            # --- Early Stopping Check ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model()
                print("  New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= Config.READER_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print("Reader training completed.")

    def save_model(self):
        """Saves the model state dict."""
        if self.model:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        """Loads the model state dict."""
        if self.vocab is None:
            self.load_vocab()

        pad_idx = self.vocab[Config.PAD_TOKEN]

        self.model = StackedBiGRUReader(
            vocab_size=len(self.vocab),
            embedding_dim=Config.READER_EMBEDDING_DIM,
            hidden_dim=Config.READER_HIDDEN_DIM,
            num_layers=Config.READER_NUM_LAYERS,
            dropout=Config.READER_DROPOUT,
            pad_idx=pad_idx,
        ).to(self.device)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            print(f"Reader model loaded from {self.model_path}")
        else:
            print(f"Warning: No reader model found at {self.model_path}")

    def predict(self, q_indices, ctx_indices):
        """
        Inference method.
        Args:
            q_indices: (batch_size, max_q_len)
            ctx_indices: (batch_size, max_ctx_len)
        Returns:
            start_probs: (batch_size, max_ctx_len)
            end_probs: (batch_size, max_ctx_len)
        """
        if self.model is None:
            self.load_model()

        self.model.eval()
        with torch.no_grad():
            q_tensor = torch.tensor(q_indices, dtype=torch.long).to(self.device)
            ctx_tensor = torch.tensor(ctx_indices, dtype=torch.long).to(self.device)

            start_logits, end_logits = self.model(q_tensor, ctx_tensor)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

        return start_probs, end_probs
