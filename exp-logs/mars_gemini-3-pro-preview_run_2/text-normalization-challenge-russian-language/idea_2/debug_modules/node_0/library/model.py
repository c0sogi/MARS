import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import save_data, ensure_dir
from library.vocab import CharTokenizer
from library.dataset import DigitSeq2SeqDataset


class PositionalEncoding(nn.Module):
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

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ContextAwareTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=Config.EMBED_DIM,
        n_heads=Config.N_HEADS,
        hidden_dim=Config.HIDDEN_DIM,
        n_layers=Config.N_LAYERS,
        dropout=Config.DROPOUT,
        pad_idx=Config.PAD_IDX,
        device=Config.DEVICE,
    ):
        super(ContextAwareTransformer, self).__init__()
        self.pad_idx = pad_idx
        self.device = device

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(
            embed_dim, max_len=Config.MAX_SEQ_LEN, dropout=dropout
        )

        # Transformer
        # batch_first=True expects [Batch, Seq, Feature]
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Initialize parameters
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(self.device)

    def create_masks(self, src, tgt):
        # src: [Batch, Src_Len]
        # tgt: [Batch, Tgt_Len]

        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        src_padding_mask = src == self.pad_idx
        tgt_padding_mask = tgt == self.pad_idx

        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # src: [Batch, Src_Len]
        # tgt: [Batch, Tgt_Len]

        tgt_mask, src_padding_mask, tgt_padding_mask = self.create_masks(src, tgt)

        src_emb = self.pos_encoder(
            self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        )
        tgt_emb = self.pos_encoder(
            self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        )

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.fc_out(outs)

    def encode(self, src):
        # Helper for inference
        src_seq_len = src.shape[1]
        src_padding_mask = src == self.pad_idx
        src_emb = self.pos_encoder(
            self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        )
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask):
        # Helper for inference
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        tgt_padding_mask = tgt == self.pad_idx

        tgt_emb = self.pos_encoder(
            self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        )

        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.fc_out(out)


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # tgt_input: [SOS, ..., token_n]
        # tgt_output: [token_1, ..., EOS]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        output = model(src, tgt_input)

        # Reshape for loss
        # output: [Batch, Seq, Vocab] -> [Batch*Seq, Vocab]
        # tgt_output: [Batch, Seq] -> [Batch*Seq]
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(
    tokenizer,
    train_dataset=None,
    val_dataset=None,
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    lr=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    save_path=Config.MODEL_CHECKPOINT,
):
    """
    Main training function.
    """
    device = Config.DEVICE
    print(f"Training on device: {device}")

    # Datasets
    if train_dataset is None:
        train_dataset = DigitSeq2SeqDataset(mode="train", tokenizer=tokenizer)
    if val_dataset is None:
        val_dataset = DigitSeq2SeqDataset(mode="val", tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    vocab_size = len(tokenizer.token2idx)
    model = ContextAwareTransformer(vocab_size=vocab_size, device=device)

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, Config.CLIP_GRAD
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {train_loss:.8f}")
        print(f"\t Val. Loss: {val_loss:.8f}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ensure_dir(save_path)
            torch.save(model.state_dict(), save_path)
            print(f"\tNew best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"\tNo improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model


def predict_beam(
    model,
    tokenizer,
    src_tensor,
    beam_width=Config.BEAM_WIDTH,
    max_len=Config.MAX_SEQ_LEN,
):
    """
    Performs beam search inference for a single source sequence.
    """
    model.eval()
    device = model.device
    src = src_tensor.unsqueeze(0).to(device)  # [1, Seq]

    with torch.no_grad():
        memory, memory_key_padding_mask = model.encode(src)

        # Start with SOS
        # List of tuples: (sequence_tensor, score)
        beams = [(torch.tensor([[Config.SOS_IDX]], device=device), 0.0)]

        completed_beams = []

        for _ in range(max_len):
            new_beams = []

            for seq, score in beams:
                if seq[0, -1].item() == Config.EOS_IDX:
                    completed_beams.append((seq, score))
                    continue

                # Decode
                out = model.decode(seq, memory, memory_key_padding_mask)
                # Get logits for last token: [1, Seq, Vocab] -> [1, Vocab]
                logits = out[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                # Top k
                topk_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                for k in range(beam_width):
                    idx = topk_indices[0, k].unsqueeze(0).unsqueeze(0)  # [1, 1]
                    prob = topk_probs[0, k].item()

                    new_seq = torch.cat([seq, idx], dim=1)
                    new_score = score + prob
                    new_beams.append((new_seq, new_score))

            # Prune beams
            # Sort by score (descending)
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

            # If all beams are completed, stop
            if all(b[0][0, -1].item() == Config.EOS_IDX for b in beams):
                completed_beams.extend(beams)
                break

        # If no beams completed naturally (max_len reached), treat current beams as completed
        if not completed_beams:
            completed_beams = beams

        # Select best
        completed_beams.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed_beams[0][0]

        # Decode to string
        # remove_special_tokens=True in tokenizer handles SOS/EOS removal
        decoded_str = tokenizer.decode(best_seq.squeeze(0))
        return decoded_str


def load_model(model_path, tokenizer):
    """
    Loads a trained model from checkpoint.
    """
    device = Config.DEVICE
    vocab_size = len(tokenizer.token2idx)
    model = ContextAwareTransformer(vocab_size=vocab_size, device=device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")

    model.eval()
    return model
