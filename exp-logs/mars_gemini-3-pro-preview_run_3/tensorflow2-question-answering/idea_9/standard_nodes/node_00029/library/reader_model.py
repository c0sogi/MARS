import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from library.config import Config


class GatedConvBlock(nn.Module):
    """
    A Gated Convolutional Block consisting of a 1D convolution,
    GLU activation, and a residual connection.
    """

    def __init__(self, hidden_dim, kernel_size, dropout):
        super(GatedConvBlock, self).__init__()
        # Output channels is 2 * hidden_dim because GLU halves the dimension
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Hidden_Dim, Seq_Len)
        residual = x
        out = self.conv(x)
        out = F.glu(out, dim=1)  # (Batch, Hidden_Dim, Seq_Len)
        out = self.dropout(out)
        return out + residual


class GatedConvReader(nn.Module):
    """
    Gated Convolutional Network for Reading Comprehension.
    Predicts start and end token probabilities.
    """

    def __init__(self, embedding_matrix=None):
        super(GatedConvReader, self).__init__()

        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.hidden_dim = Config.READER_FILTERS
        self.kernel_size = Config.READER_KERNEL_SIZE
        self.num_layers = Config.READER_LAYERS
        self.dropout_prob = Config.READER_DROPOUT

        # 1. Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )

        # 2. Projection to hidden dimension
        # Input to Conv1d needs to be (Batch, Channel, Length)
        # We project embedding dim to filter dim
        self.input_proj = nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size=1)

        # 3. Gated Convolutional Encoder
        self.layers = nn.ModuleList(
            [
                GatedConvBlock(self.hidden_dim, self.kernel_size, self.dropout_prob)
                for _ in range(self.num_layers)
            ]
        )

        # 4. Output Heads
        self.start_head = nn.Linear(self.hidden_dim, 1)
        self.end_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_indices):
        """
        Args:
            input_indices: (Batch, Seq_Len)
        Returns:
            start_logits: (Batch, Seq_Len)
            end_logits: (Batch, Seq_Len)
        """
        # Create mask for padding (1 for valid, 0 for pad)
        mask = input_indices != 0  # (B, L)

        # Embed
        x = self.embedding(input_indices)  # (B, L, E)

        # Transpose for Conv1d: (B, E, L)
        x = x.transpose(1, 2)

        # Project to hidden dim
        x = self.input_proj(x)  # (B, H, L)

        # Apply Gated Conv Blocks
        for layer in self.layers:
            x = layer(x)

        # Transpose back for Linear layers: (B, L, H)
        x = x.transpose(1, 2)

        # Compute logits
        start_logits = self.start_head(x).squeeze(-1)  # (B, L)
        end_logits = self.end_head(x).squeeze(-1)  # (B, L)

        # Mask padding positions with large negative value to prevent selection
        start_logits = start_logits.masked_fill(~mask, -1e9)
        end_logits = end_logits.masked_fill(~mask, -1e9)

        return start_logits, end_logits


def train_reader(train_loader, val_loader, embedding_matrix, device):
    """
    Trains the Reader model with Early Stopping.
    """
    model = GatedConvReader(embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting Reader training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_indices = batch["input_indices"].to(device)
            start_targets = batch["start_positions"].to(device)
            end_targets = batch["end_positions"].to(device)

            optimizer.zero_grad()

            start_logits, end_logits = model(input_indices)

            # Calculate loss for both start and end
            loss_start = criterion(start_logits, start_targets)
            loss_end = criterion(end_logits, end_targets)
            loss = loss_start + loss_end

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # Metrics
            batch_size = input_indices.size(0)
            train_loss += loss.item() * batch_size

            # Exact Match Accuracy
            _, pred_start = torch.max(start_logits, dim=1)
            _, pred_end = torch.max(end_logits, dim=1)

            correct_mask = (pred_start == start_targets) & (pred_end == end_targets)
            train_correct += correct_mask.sum().item()
            train_total += batch_size

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_indices = batch["input_indices"].to(device)
                start_targets = batch["start_positions"].to(device)
                end_targets = batch["end_positions"].to(device)

                start_logits, end_logits = model(input_indices)

                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                loss = loss_start + loss_end

                batch_size = input_indices.size(0)
                val_loss += loss.item() * batch_size

                _, pred_start = torch.max(start_logits, dim=1)
                _, pred_end = torch.max(end_logits, dim=1)

                correct_mask = (pred_start == start_targets) & (pred_end == end_targets)
                val_correct += correct_mask.sum().item()
                val_total += batch_size

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {avg_train_loss} | Train EM Acc: {train_acc} | "
            f"Val Loss: {avg_val_loss} | Val EM Acc: {val_acc}"
        )

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.READER_MODEL_PATH)
            print(f"  New best model saved to {Config.READER_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return model
