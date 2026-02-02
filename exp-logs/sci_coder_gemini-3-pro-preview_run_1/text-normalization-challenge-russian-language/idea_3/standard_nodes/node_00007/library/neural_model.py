import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import numpy as np
from library.config import Config
from library.utils import get_artifact_path, seed_everything

# --- Model Definition ---


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # pe: [max_len, d_model] -> slice to [seq_len, d_model]
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class CharSeq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=128,
        pad_token_id=0,
        sos_token_id=1,
        eos_token_id=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output Head
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt, device):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(
            torch.bool
        )

        src_padding_mask = src == self.pad_token_id
        tgt_padding_mask = tgt == self.pad_token_id

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]

        device = src.device
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt, device
        )

        # Embeddings + Positional Encoding
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # Transformer Pass
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

    def predict(self, src, max_len=None):
        """
        Greedy decoding for inference.
        """
        if max_len is None:
            max_len = self.max_seq_len

        device = src.device
        batch_size = src.shape[0]

        # Encode
        src_mask = torch.zeros((src.shape[1], src.shape[1]), device=device).type(
            torch.bool
        )
        src_padding_mask = src == self.pad_token_id

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask
        )

        # Decode
        ys = (
            torch.ones(batch_size, 1)
            .fill_(self.sos_token_id)
            .type(torch.long)
            .to(device)
        )

        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for i in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1), device)
            tgt_emb = self.pos_encoder(self.embedding(ys) * math.sqrt(self.d_model))

            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            prob = self.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Update finished status
            finished |= next_word == self.eos_token_id

            # Append next word
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if finished.all():
                break

        return ys


# --- Trainer ---


class NeuralTrainer:
    def __init__(self, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = CharSeq2SeqTransformer(
            vocab_size=Config.VOCAB_SIZE,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            max_seq_len=Config.MAX_SEQ_LEN,
            pad_token_id=tokenizer.pad_token_id,
            sos_token_id=tokenizer.char_to_id.get("<sos>", 1),
            eos_token_id=tokenizer.char_to_id.get("<eos>", 2),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train(self, train_loader, val_loader):
        best_val_loss = float("inf")
        patience_counter = 0
        model_save_path = get_artifact_path("neural_normalizer_best.pt")

        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                src = batch["input_ids"].to(self.device)
                tgt = batch["target_ids"].to(self.device)

                # Target Input: <sos> ... <eos> -> <sos> ...
                # Target Output: <sos> ... <eos> -> ... <eos>
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                self.optimizer.zero_grad()
                logits = self.model(src, tgt_input)

                # Reshape for loss: [batch * seq_len, vocab_size]
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
                )
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            val_loss = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_save_path)
                # print(f"  Saved best model to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model for future use
        if os.path.exists(model_save_path):
            self.model.load_state_dict(
                torch.load(model_save_path, map_location=self.device)
            )

        return model_save_path

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["input_ids"].to(self.device)
                tgt = batch["target_ids"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            return True
        return False
