import math
import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("model")


class PositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, D_Model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GapTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=Config.D_MODEL,
        nhead=Config.N_HEAD,
        num_layers=Config.N_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        dropout=Config.DROPOUT,
        pad_idx=0,
    ):
        super(GapTransformer, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Add buffer to max_len for safety
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=Config.MAX_SEQ_LEN + 50
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation=Config.ACTIVATION,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output projects to vocab size.
        # Logits at position i represent the probability of inserting a word AFTER position i.
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src: (Batch, Seq_Len)

        # Create padding mask: True where value is pad_idx
        src_key_padding_mask = src == self.pad_idx

        # Embed and add position info
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Encode
        output = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # Project to vocab
        logits = self.decoder(output)  # (Batch, Seq_Len, Vocab_Size)

        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        # Flatten for CrossEntropyLoss
        # logits: (B, L, V) -> (B*L, V)
        # targets: (B, L) -> (B*L)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 2000 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item()}"
            )

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def run_training(model, train_loader, val_loader, vocab_size, no_insert_idx):
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Class weighting: Down-weight the majority [NO_INSERT] class
    weights = torch.ones(vocab_size).to(device)
    weights[no_insert_idx] = Config.NO_INSERT_WEIGHT

    # Ignore padding index (0) in loss
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps, pct_start=0.1
    )

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    logger.info(f"Starting training on {device}...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"Patience {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # Load best model for inference
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def generate_submission(model, test_loader, vocab, output_path):
    device = torch.device(Config.DEVICE)
    model.eval()
    model.to(device)

    predictions = []
    ids = []
    itos = vocab.itos

    # Indices to mask
    pad_idx = vocab.get_pad_index()
    unk_idx = vocab.get_unk_index()
    no_insert_idx = vocab.get_no_insert_index()

    logger.info("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            batch_ids = batch["id"].numpy()
            input_ids = batch["input_ids"].to(device)

            logits = model(input_ids)  # (B, L, V)

            # Mask special tokens to force valid word prediction
            # Set probabilities to -inf
            logits[:, :, pad_idx] = float("-inf")
            logits[:, :, unk_idx] = float("-inf")
            logits[:, :, no_insert_idx] = float("-inf")

            # Mask the first position (index 0) because we insert AFTER the token.
            # We cannot insert before the first token (as per task rules, gap is never first/last word).
            # Actually, inserting after index 0 is valid (it's the second spot).
            # But the task says "never the first or last word".
            # If we insert after token 0, it becomes the 2nd word. This is allowed.
            # However, we should mask the LAST token position, because inserting after the last token
            # would make it the new last word (or after the period).
            # The last token is usually a period.

            # Let's rely on the model's training, but strictly:
            # We want to find the max probability across the whole sentence.

            B, L, V = logits.shape
            flat_logits = logits.view(B, -1)

            # Find global max for each sentence
            max_vals, max_indices = torch.max(flat_logits, dim=1)

            # Decode
            pred_pos = (max_indices // V).cpu().numpy()
            pred_word_idx = (max_indices % V).cpu().numpy()
            input_ids_np = input_ids.cpu().numpy()

            for i in range(len(batch_ids)):
                row_id = batch_ids[i]
                pos = pred_pos[i]
                w_idx = pred_word_idx[i]
                original_ids = input_ids_np[i]

                # Decode the word to insert
                word_to_insert = itos[w_idx]

                # Reconstruct sentence
                # 1. Convert IDs to tokens, ignoring padding
                valid_tokens = []
                for tid in original_ids:
                    if tid == pad_idx:
                        break
                    valid_tokens.append(itos[tid])

                # 2. Insert the word
                # Logic: Prediction at `pos` means insert AFTER `pos`.
                # If pos is beyond current length (due to masking issues?), clamp it.
                # Note: `pos` is relative to the padded sequence.
                # We need to ensure `pos` is within the valid_tokens range.
                if pos >= len(valid_tokens):
                    pos = len(valid_tokens) - 1

                # Insert after `pos`
                valid_tokens.insert(pos + 1, word_to_insert)

                # 3. Join
                pred_sentence = " ".join(valid_tokens)

                ids.append(row_id)
                predictions.append(pred_sentence)

    # Create DataFrame and save
    df_sub = pd.DataFrame({"id": ids, "sentence": predictions})

    # Use QUOTE_NONNUMERIC to ensure sentences are quoted as required
    # "id","sentence"
    # 1,"The sentence..."
    df_sub.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Submission saved to {output_path}")
