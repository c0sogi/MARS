import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import ensure_dir


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

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Slice pe to [seq_len, d_model] and broadcast add
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    Transformer-based Sequence-to-Sequence model for text normalization.
    """

    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.emb_size = emb_size

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        # Embed and add position encoding
        src_emb = self.positional_encoding(
            self.src_tok_emb(src) * math.sqrt(self.emb_size)
        )
        tgt_emb = self.positional_encoding(
            self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size)
        )

        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size)),
            src_mask,
        )

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size)),
            memory,
            tgt_mask,
        )


class NeuralNormalizer:
    """
    Wrapper class to handle training, evaluation, and inference of the Seq2SeqTransformer.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device

        # Initialize Model
        self.model = Seq2SeqTransformer(
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            emb_size=config.embedding_dim,
            nhead=config.nhead,
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        ).to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss Function
        # We need to identify the padding index
        pad_token = config.pad_token
        self.pad_idx = tokenizer.char_to_id.get(pad_token, 0)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, label_smoothing=config.label_smoothing
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )

        src_padding_mask = src == self.pad_idx
        tgt_padding_mask = tgt == self.pad_idx

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def train(self, train_loader, val_loader):
        """
        Trains the model with Early Stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                # Prepare inputs for transformer
                # tgt_input is tgt shifted by 1 (exclude last token)
                # tgt_out is tgt shifted by 1 (exclude first token/BOS)
                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                    self._create_mask(src, tgt_input)
                )

                self.optimizer.zero_grad()

                logits = self.model(
                    src,
                    tgt_input,
                    src_mask,
                    tgt_mask,
                    src_padding_mask,
                    tgt_padding_mask,
                    src_padding_mask,
                )

                # Reshape for loss calculation
                # logits: [batch, seq_len, vocab] -> [batch*seq_len, vocab]
                # tgt_out: [batch, seq_len] -> [batch*seq_len]
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad_norm
                )

                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            avg_val_loss = self.evaluate(val_loader)
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"    Train Loss: {avg_train_loss:.8f}")
            print(f"    Val Loss:   {avg_val_loss:.8f}")

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                ensure_dir(self.config.model_best_path)
                torch.save(self.model.state_dict(), self.config.model_best_path)
                print("    -> Best model saved.")
            else:
                patience_counter += 1
                print(
                    f"    -> No improvement. Patience: {patience_counter}/{self.config.early_stopping_patience}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Load best model for future use
        self.load(self.config.model_best_path)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                    self._create_mask(src, tgt_input)
                )

                logits = self.model(
                    src,
                    tgt_input,
                    src_mask,
                    tgt_mask,
                    src_padding_mask,
                    tgt_padding_mask,
                    src_padding_mask,
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        else:
            print(f"Warning: Model file not found at {path}")

    def predict(self, dataloader):
        """
        Runs inference on the provided dataloader using Greedy Decoding.
        Returns a dictionary mapping ID -> Predicted Text.
        """
        self.model.eval()
        predictions = {}

        # Special Tokens
        bos_idx = self.tokenizer.char_to_id[self.config.bos_token]
        eos_idx = self.tokenizer.char_to_id[self.config.eos_token]

        # Max generation length (safe upper bound)
        max_gen_len = 128

        with torch.no_grad():
            for batch in dataloader:
                src = batch["src"].to(self.device)
                ids = batch["id"]
                batch_size = src.shape[0]

                # Create src mask
                src_seq_len = src.shape[1]
                src_mask = torch.zeros(
                    (src_seq_len, src_seq_len), device=self.device
                ).type(torch.bool)
                src_padding_mask = src == self.pad_idx

                # Encode
                memory = self.model.encode(src, src_mask)

                # Initialize decoder input with BOS
                ys = torch.fill_(
                    torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
                    bos_idx,
                )

                # Keep track of finished sequences
                finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

                for i in range(max_gen_len):
                    tgt_mask = self._generate_square_subsequent_mask(ys.size(1))

                    # Decode
                    out = self.model.decode(ys, memory, tgt_mask)

                    # Project to vocab
                    # Take the last token output
                    prob = self.model.generator(out[:, -1])
                    _, next_word = torch.max(prob, dim=1)
                    next_word = next_word.unsqueeze(1)  # [batch, 1]

                    # Append to sequence
                    ys = torch.cat([ys, next_word], dim=1)

                    # Check for EOS
                    is_eos = next_word.squeeze() == eos_idx
                    finished = finished | is_eos

                    if finished.all():
                        break

                # Convert IDs to Text
                ys_cpu = ys.cpu().tolist()
                for idx, seq_ids in enumerate(ys_cpu):
                    # Decode using tokenizer (removes special tokens like BOS/EOS/PAD)
                    pred_text = self.tokenizer.decode(
                        seq_ids, remove_special_tokens=True
                    )
                    predictions[ids[idx]] = pred_text

        return predictions
