import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from library.config import Config
from library.utils import get_device
from library.data_processor import CharTokenizer


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # pe is [max_len, 1, d_model], transpose to [1, max_len, d_model] for batch_first addition
        # actually pe is [max_len, 1, d_model], x is [batch, seq_len, d_model]
        # We slice pe to [seq_len, 1, d_model] and transpose to [1, seq_len, d_model]
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class CharTransformer(nn.Module):
    """
    Character-level Encoder-Decoder Transformer for text normalization.
    """

    def __init__(self, config: Config, tokenizer: CharTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        vocab_size = len(tokenizer)

        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, config.d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(
            config.d_model, config.dropout, max_len=5000
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output projection
        self.fc_out = nn.Linear(config.d_model, vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_key_padding_mask: [batch_size, src_len] (True for pad)
            tgt_key_padding_mask: [batch_size, tgt_len] (True for pad)
            tgt_mask: [tgt_len, tgt_len] (Causal mask)
        """
        # Embed and add position info
        # Scale embeddings by sqrt(d_model) as per Attention Is All You Need
        src_emb = self.pos_encoder(
            self.src_embedding(src) * math.sqrt(self.config.d_model)
        )
        tgt_emb = self.pos_encoder(
            self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        )

        # Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        # Project to vocab
        logits = self.fc_out(output)
        return logits

    def generate(
        self, src: torch.Tensor, max_len: int = 128, device: torch.device = None
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.

        Args:
            src: [batch_size, src_len]
            max_len: Maximum length of generated sequence
            device: torch device

        Returns:
            Tensor of shape [batch_size, generated_len] containing token IDs
        """
        if device is None:
            device = src.device

        batch_size = src.size(0)

        # Prepare Encoder Input
        src_key_padding_mask = (src == self.tokenizer.pad_token_id).to(device)
        src_emb = self.pos_encoder(
            self.src_embedding(src) * math.sqrt(self.config.d_model)
        )

        # Encode
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # Initialize Decoder Input with SOS
        tgt_input = torch.full(
            (batch_size, 1),
            self.tokenizer.sos_token_id,
            dtype=torch.long,
            device=device,
        )

        # Generation Loop
        for _ in range(max_len):
            # Embed Decoder Input
            tgt_emb = self.pos_encoder(
                self.tgt_embedding(tgt_input) * math.sqrt(self.config.d_model)
            )

            # Create Causal Mask
            tgt_len = tgt_input.size(1)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(
                device
            )

            # Decode
            output = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Get logits for the last token
            last_token_logits = self.fc_out(output[:, -1, :])

            # Greedy prediction
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)

            # Append to input
            tgt_input = torch.cat([tgt_input, next_token], dim=1)

            # Check if all sequences have generated EOS (optional optimization, skip for simplicity or check per batch)
            # For batch processing, we usually continue until max_len or handle individually.
            # Here we just run to max_len or until a break condition if we were doing single sample.
            # Since it's batched, we continue.

        return tgt_input


class NeuralTrainer:
    """
    Wrapper for training and evaluating the CharTransformer.
    """

    def __init__(self, config: Config, tokenizer: CharTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = get_device()

        self.model = CharTransformer(config, tokenizer).to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=0.01
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        # Loss Function
        # Ignore padding, use label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
        )

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            # Move to device
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_pad_mask = batch["src_pad_mask"].to(self.device)
            tgt_pad_mask = batch["tgt_pad_mask"].to(self.device)

            # Prepare inputs/targets for Teacher Forcing
            # Input to decoder: <sos> ... token_n
            # Target for loss:  token_1 ... <eos>
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Adjust mask for shifted target
            tgt_pad_mask_input = tgt_pad_mask[:, :-1]

            # Causal Mask
            seq_len = tgt_input.size(1)
            tgt_mask = self.model.transformer.generate_square_subsequent_mask(
                seq_len
            ).to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask_input,
                tgt_mask=tgt_mask,
            )

            # Calculate Loss
            # Flatten: [batch * seq_len, vocab_size]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                src_pad_mask = batch["src_pad_mask"].to(self.device)
                tgt_pad_mask = batch["tgt_pad_mask"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_pad_mask_input = tgt_pad_mask[:, :-1]

                seq_len = tgt_input.size(1)
                tgt_mask = self.model.transformer.generate_square_subsequent_mask(
                    seq_len
                ).to(self.device)

                logits = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask_input,
                    tgt_mask=tgt_mask,
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader):
        print(f"Starting training on {self.device}...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(self.config.model_checkpoint_path)
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")
