import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        # Shape: (1, max_len, d_model) for batch_first broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    Transformer-based Sequence-to-Sequence model for character transduction.
    """

    def __init__(
        self,
        vocab_size,
        pad_token_id,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
    ):
        super(TransformerSeq2Seq, self).__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=Config.MAX_CHAR_LEN + 50, dropout=dropout
        )

        # Transformer
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Used to prevent the decoder from looking ahead.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt):
        """
        Creates padding masks for src and tgt, and look-ahead mask for tgt.
        """
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(
            torch.bool
        )

        # Padding masks: True where value is pad_token_id
        src_padding_mask = src == self.pad_token_id
        tgt_padding_mask = tgt == self.pad_token_id

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        """
        Forward pass for training.
        src: (batch, src_len)
        tgt: (batch, tgt_len) - includes SOS, excludes EOS usually for input,
             but standard practice is: input=tgt[:-1], target=tgt[1:]
        """
        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt
        )

        # Embed and add position encoding
        # Multiply by sqrt(d_model) as per Attention Is All You Need
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # Transformer pass
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

    def predict(self, src, tokenizer, max_len=Config.MAX_CHAR_LEN):
        """
        Greedy decoding for inference.
        src: (batch, src_len)
        """
        self.eval()
        device = src.device
        batch_size = src.shape[0]

        # Encode source
        src_padding_mask = src == self.pad_token_id
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )

        # Initialize decoder input with SOS
        sos_id = tokenizer.sos_token_id
        eos_id = tokenizer.eos_token_id

        # (batch, 1)
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Embed decoder input
            tgt_emb = self.pos_encoder(self.embedding(ys) * math.sqrt(self.d_model))

            # Create target mask (look-ahead)
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)

            # Decoder forward pass
            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,  # Not strictly needed for generation if we don't pad right
                memory_key_padding_mask=src_padding_mask,
            )

            # Project to vocab
            prob = self.fc_out(out[:, -1])  # Take last token output

            # Greedy choice
            _, next_word = torch.max(prob, dim=1)

            # Append to sequence
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Update finished status
            finished |= next_word == eos_id

            # Stop if all finished
            if finished.all():
                break

        return ys


class NeuralTrainer:
    """
    Manager for training and evaluating the Transformer model.
    """

    def __init__(self, model, tokenizer, device=Config.DEVICE):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=Config.LABEL_SMOOTHING
        )

        self.scheduler = None  # Initialized in train()

    def train(self, train_loader, val_loader, epochs=Config.EPOCHS):
        print(f"Starting training on {self.device} for {epochs} epochs...")

        # Initialize Scheduler (Linear Warmup + Cosine Decay)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Training Phase
            self.model.train()
            total_train_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                # Unpack batch
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)

                # Prepare inputs and targets for teacher forcing
                # Input: [SOS, t1, t2, ...]
                # Target: [t1, t2, ..., EOS]
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                self.optimizer.zero_grad()

                logits = self.model(src, tgt_input)

                # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len)
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
                )

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.GRAD_CLIP
                )

                self.optimizer.step()
                self.scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation Phase
            avg_val_loss = self.evaluate(val_loader)

            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch}/{epochs} | Time: {epoch_time:.1f}s | "
                f"Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}"
            )

            # Checkpointing & Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save(Config.MODEL_PATH)
                patience_counter = 0
                print(f"  -> New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        # Load best model before returning
        self.load(Config.MODEL_PATH)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(
                torch.load(path, map_location=torch.device("cpu"))
            )
