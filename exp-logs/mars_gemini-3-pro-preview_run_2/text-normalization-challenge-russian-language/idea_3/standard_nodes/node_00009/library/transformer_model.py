import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.utils import save_checkpoint, load_checkpoint


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

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # pe shape: (1, Max_Len, Dim) -> slice to (1, Seq_Len, Dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Scale embeddings by sqrt(d_model) as per Attention Is All You Need
        return self.embedding(x) * math.sqrt(self.d_model)


class Seq2SeqTransformer(nn.Module):
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

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )  # Important: batch_first=True

        self.generator = nn.Linear(emb_size, tgt_vocab_size)

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
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

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
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, pad_idx, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = src == pad_idx
    tgt_padding_mask = tgt == pad_idx

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class Trainer:
    def __init__(self, model, train_loader, val_loader, vocab_size, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.device = device

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=Config.PAD_IDX, label_smoothing=Config.LABEL_SMOOTHING
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # OneCycleLR scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=steps_per_epoch,
            epochs=Config.EPOCHS,
            pct_start=0.1,  # Warmup for first 10%
        )

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            src = batch["input_ids"].to(self.device)
            tgt = batch["labels"].to(self.device)

            # Prepare inputs for Transformer
            # tgt_input: Exclude last token (EOS/PAD)
            # tgt_out: Exclude first token (SOS) - this is what we predict
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input, Config.PAD_IDX, self.device
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

            self.optimizer.zero_grad()

            # Reshape for loss calculation: (Batch * Seq_Len, Vocab_Size) vs (Batch * Seq_Len)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch_idx} | Train Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["input_ids"].to(self.device)
                tgt = batch["labels"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt_input, Config.PAD_IDX, self.device
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

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.10f}")
        return avg_loss

    def fit(self):
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"New best model found! Saving to {Config.MODEL_CHECKPOINT}")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": best_val_loss,
                    },
                    Config.MODEL_CHECKPOINT,
                )
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break


class Generator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(device)

    def greedy_decode(
        self,
        src,
        max_len=Config.MAX_TARGET_LEN,
        start_symbol=Config.SOS_IDX,
        end_symbol=Config.EOS_IDX,
    ):
        """
        Performs greedy decoding for a batch of source sequences.
        """
        batch_size = src.shape[0]
        src = src.to(self.device)
        src_mask = torch.zeros((src.shape[1], src.shape[1]), device=self.device).type(
            torch.bool
        )
        src_padding_mask = src == Config.PAD_IDX

        memory = self.model.encode(src, src_mask)

        # Initialize decoder input with SOS token
        ys = torch.fill_(
            torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
            start_symbol,
        )

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), self.device)

            # The decoder expects memory_key_padding_mask to match the source padding
            # memory_key_padding_mask is essentially src_padding_mask
            out = self.model.decode(ys, memory, tgt_mask)

            # Project to vocab
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)

            # Update ys
            ys = torch.cat([ys, next_word], dim=1)

            # Check for EOS
            is_eos = next_word.squeeze() == end_symbol
            finished = finished | is_eos

            if finished.all():
                break

        return ys

    def predict_batch(self, batch_input_ids):
        """
        Takes a batch of input IDs and returns a list of decoded strings.
        """
        with torch.no_grad():
            output_tokens = self.greedy_decode(batch_input_ids)

        decoded_texts = []
        for tokens in output_tokens:
            # Convert tensor to list
            token_list = tokens.tolist()
            # Decode using tokenizer
            text = self.tokenizer.decode(token_list, remove_special_tokens=True)
            decoded_texts.append(text)

        return decoded_texts
