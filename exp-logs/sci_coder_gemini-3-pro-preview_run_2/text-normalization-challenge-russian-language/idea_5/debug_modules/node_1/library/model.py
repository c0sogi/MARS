import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from library.config import Config
from library.utils import set_seed


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # Slice pe to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CharToSubwordTransformer(nn.Module):
    """
    Heterogeneous Transformer Architecture.
    Encoder: Character-level (high granularity for parsing numbers/symbols).
    Decoder: Subword-level (medium granularity for generating language).
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        d_model=Config.ENC_EMB_DIM,
        nhead=Config.ENC_HEADS,
        num_encoder_layers=Config.ENC_LAYERS,
        num_decoder_layers=Config.DEC_LAYERS,
        dim_feedforward=Config.ENC_HIDDEN_DIM,
        dropout=Config.DROPOUT,
        max_len=Config.MAX_SEQ_LEN,
    ):
        super(CharToSubwordTransformer, self).__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)

        # Transformer
        # batch_first=True ensures input/output tensors are (Batch, Seq, Feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection to target vocabulary
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier Uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates a causal mask for the decoder to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt, device):
        """Creates source and target masks for padding and causal attention."""
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # Causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        # No causal mask needed for encoder
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(
            torch.bool
        )

        # Padding masks (True where padding exists)
        src_padding_mask = src == self.src_pad_idx
        tgt_padding_mask = tgt == self.tgt_pad_idx

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        device = src.device

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt, device
        )

        # Apply Embeddings and Positional Encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Pass through Transformer
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.fc_out(outs)

    def encode(self, src):
        """Helper for inference: Encodes the source sequence."""
        src_padding_mask = src == self.src_pad_idx
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, memory_key_padding_mask=None):
        """Helper for inference: Decodes one step given memory."""
        device = tgt.device
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        tgt_padding_mask = tgt == self.tgt_pad_idx

        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Executes the training loop with Early Stopping and Metric Logging.

    Args:
        model: The initialized CharToSubwordTransformer.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.
        epochs: Maximum number of epochs.
        patience: Patience for early stopping.

    Returns:
        The trained model (loaded with best weights).
    """
    set_seed()
    model = model.to(device)

    # Loss Function: Cross Entropy ignoring padding
    # Label smoothing helps generalization on ambiguous numbers
    criterion = nn.CrossEntropyLoss(
        ignore_index=model.tgt_pad_idx, label_smoothing=Config.LABEL_SMOOTHING
    )

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: OneCycleLR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            # Teacher Forcing:
            # Input to Decoder: <s> Token1 Token2 ...
            # Target for Loss:  Token1 Token2 ... </s>
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            output = model(src, tgt_input)

            # Reshape output and target for Loss calculation
            # Output: (Batch * SeqLen, VocabSize)
            # Target: (Batch * SeqLen)
            loss = criterion(
                output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1)
            )

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ==========================
        # Validation Loop
        # ==========================
        model.eval()
        val_loss = 0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = model(src, tgt_input)

                loss = criterion(
                    output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1)
                )
                val_loss += loss.item()

                # Calculate Accuracy (Exact Token Match)
                preds = torch.argmax(output, dim=-1)

                # Create mask to ignore padding in accuracy calculation
                mask = tgt_output != model.tgt_pad_idx
                correct_tokens += ((preds == tgt_output) & mask).sum().item()
                total_tokens += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.10f} | "
            f"Val Loss: {avg_val_loss:.10f} | "
            f"Val Acc: {val_accuracy:.10f}"
        )

        # ==========================
        # Early Stopping & Checkpointing
        # ==========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Ensure directory exists
            os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model weights before returning
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    return model
