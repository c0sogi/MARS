import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os
import pandas as pd
import numpy as np
import time
from library.config import Config
from library.utils import set_seed

# ==========================================
# Model Architecture
# ==========================================


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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CharToBPETransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super(CharToBPETransformer, self).__init__()

        self.d_model = Config.D_MODEL

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(
            self.d_model, Config.DROPOUT, max_len=Config.MAX_ENC_LEN + 50
        )
        self.pos_decoder = PositionalEncoding(
            self.d_model, Config.DROPOUT, max_len=Config.MAX_DEC_LEN + 50
        )

        # Transformer
        # batch_first=True is available in torch >= 1.9. Environment has 2.8.0.
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # Output projection
        self.fc_out = nn.Linear(self.d_model, tgt_vocab_size)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None,
    ):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)

        # Embed and Add Position
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


# ==========================================
# Dataset
# ==========================================


class NormalizationDataset(Dataset):
    def __init__(self, df, tokenizer, hfbb_model=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.hfbb = hfbb_model
        self.is_train = is_train

        # Pre-compute weights if training and HFBB provided
        self.weights = None
        if self.is_train and self.hfbb:
            self.weights = self._compute_weights()

    def _compute_weights(self):
        weights = []
        # Extract necessary columns as list of dicts for speed
        records = self.df[["before", "prev", "next", "after"]].to_dict("records")

        for row in records:
            before = str(row["before"])
            prev = str(row.get("prev", "<START>"))
            nxt = str(row.get("next", "<END>"))
            target = str(row["after"])

            # Query HFBB
            pred, conf, level = self.hfbb.query(before, prev, nxt)

            # Logic:
            # If Tier 1 is correct and high confidence -> Anchor (Low Weight)
            # Else -> Residual (High Weight)

            if pred == target and conf > Config.HFBB_CONFIDENCE_THRESHOLD:
                weights.append(Config.WEIGHT_ANCHOR)
            else:
                weights.append(Config.WEIGHT_RESIDUAL)

        return np.array(weights, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Context
        before = str(row["before"])
        prev = str(row.get("prev", "<START>"))
        nxt = str(row.get("next", "<END>"))

        # Encode Source
        src_tensor = self.tokenizer.encode_src(prev, before, nxt)

        item = {"src": src_tensor}

        if self.is_train:
            target = str(row["after"])
            tgt_tensor = self.tokenizer.encode_tgt(target)
            item["tgt"] = tgt_tensor

            if self.weights is not None:
                item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float)
            else:
                item["weight"] = torch.tensor(1.0, dtype=torch.float)

        return item


# ==========================================
# Trainer / Manager
# ==========================================


class TransformerTrainer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = torch.device(Config.DEVICE)

        # Get vocab sizes
        src_vocab_size, tgt_vocab_size = self.tokenizer.get_vocab_sizes()

        # Initialize Model
        self.model = CharToBPETransformer(src_vocab_size, tgt_vocab_size).to(
            self.device
        )

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=Config.LABEL_SMOOTHING,
            reduction="none",  # To apply sample weights
        )

    def _prepare_data(self, df, hfbb_model=None, is_train=True):
        # Context generation if not present
        if "prev" not in df.columns or "next" not in df.columns:
            # Assumes df is sorted by sentence_id, token_id
            df["prev"] = df["before"].shift(1).fillna("<START>")
            df["next"] = df["before"].shift(-1).fillna("<END>")

            # Fix boundaries
            if "sentence_id" in df.columns:
                start_mask = df["sentence_id"] != df["sentence_id"].shift(1)
                end_mask = df["sentence_id"] != df["sentence_id"].shift(-1)
                df.loc[start_mask, "prev"] = "<START>"
                df.loc[end_mask, "next"] = "<END>"

        # Filtering for "Density-Maximized Semiotics"
        if is_train:
            # Regex for digits or latin
            semiotic_mask = (
                df["before"].astype(str).str.contains(r"\d|[a-zA-Z]", regex=True)
            )
            df_filtered = df[semiotic_mask].copy()

            # Upsampling rare classes
            if "class" in df_filtered.columns:
                # Simple heuristic: upsample MONEY, DECIMAL, TELEPHONE
                rare_classes = ["MONEY", "DECIMAL", "TELEPHONE", "ELECTRONIC", "DIGIT"]
                dfs = [df_filtered]
                for cls in rare_classes:
                    subset = df_filtered[df_filtered["class"] == cls]
                    if len(subset) > 0:
                        # Upsample 5x
                        dfs.append(subset)
                        dfs.append(subset)
                        dfs.append(subset)
                        dfs.append(subset)
                        dfs.append(subset)
                df_filtered = pd.concat(dfs)

            # Shuffle
            df_filtered = df_filtered.sample(
                frac=1.0, random_state=Config.SEED
            ).reset_index(drop=True)

            if Config.DEBUG:
                df_filtered = df_filtered.head(Config.DEBUG_SIZE)

            return df_filtered

        return df

    def train(self, hfbb_model=None):
        set_seed(Config.SEED)

        # Load Data
        print("Loading training data for Transformer...")
        df_train = pd.read_csv(Config.TRAIN_DATA)
        df_val = pd.read_csv(Config.VAL_DATA)

        # Prepare Data (Filter/Upsample)
        df_train_proc = self._prepare_data(df_train, hfbb_model, is_train=True)
        # Validation set: Use a subset of original validation, but focus on semiotic for metric relevance
        df_val_proc = self._prepare_data(df_val, hfbb_model, is_train=True)
        df_val_proc = df_val_proc.drop_duplicates(subset=["sentence_id", "token_id"])

        print(
            f"Training on {len(df_train_proc)} samples. Validation on {len(df_val_proc)} samples."
        )

        # Datasets
        train_dataset = NormalizationDataset(
            df_train_proc, self.tokenizer, hfbb_model, is_train=True
        )
        val_dataset = NormalizationDataset(
            df_val_proc, self.tokenizer, hfbb_model, is_train=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting training...")

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                weights = batch["weight"].to(self.device)

                # tgt input: <SOS> ... <last>
                tgt_input = tgt[:, :-1]
                # tgt output: ... <last> <EOS>
                tgt_output = tgt[:, 1:]

                # Masks
                src_pad_mask = src == self.tokenizer.char2id[self.tokenizer.PAD_TOKEN]
                tgt_pad_mask = tgt_input == self.tokenizer.pad_id
                tgt_mask = self.model.generate_square_subsequent_mask(
                    tgt_input.size(1)
                ).to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(
                    src,
                    tgt_input,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    tgt_mask=tgt_mask,
                )

                # Reshape for loss
                loss_per_token = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

                # Apply weights: broadcast (N) -> (N, T) -> (N*T)
                weights_expanded = (
                    weights.unsqueeze(1).expand_as(tgt_output).reshape(-1)
                )
                weighted_loss = loss_per_token * weights_expanded

                # Mean over non-pad tokens
                non_pad_mask = tgt_output.reshape(-1) != self.tokenizer.pad_id
                if non_pad_mask.sum() > 0:
                    loss = weighted_loss[non_pad_mask].mean()
                else:
                    loss = weighted_loss.mean()  # Fallback

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            avg_val_loss = self.evaluate(val_loader)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {elapsed:.0f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  New best model saved to {Config.BEST_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model
        if os.path.exists(Config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_pad_mask = src == self.tokenizer.char2id[self.tokenizer.PAD_TOKEN]
                tgt_pad_mask = tgt_input == self.tokenizer.pad_id
                tgt_mask = self.model.generate_square_subsequent_mask(
                    tgt_input.size(1)
                ).to(self.device)

                logits = self.model(
                    src,
                    tgt_input,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    tgt_mask=tgt_mask,
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

                # Filter pads
                non_pad_mask = tgt_output.reshape(-1) != self.tokenizer.pad_id
                if non_pad_mask.sum() > 0:
                    total_loss += loss[non_pad_mask].mean().item()

        return total_loss / len(val_loader)

    def predict(self, src_tensor):
        # Greedy decoding
        self.model.eval()
        src = src_tensor.to(self.device)
        batch_size = src.size(0)

        # Start token
        tgt_indices = torch.full(
            (batch_size, 1), self.tokenizer.bos_id, dtype=torch.long, device=self.device
        )

        # Encoder output
        src_pad_mask = src == self.tokenizer.char2id[self.tokenizer.PAD_TOKEN]
        src_emb = self.model.pos_encoder(
            self.model.src_embedding(src) * math.sqrt(self.model.d_model)
        )
        memory = self.model.transformer.encoder(
            src_emb, src_key_padding_mask=src_pad_mask
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(Config.MAX_DEC_LEN):
            tgt_emb = self.model.pos_decoder(
                self.model.tgt_embedding(tgt_indices) * math.sqrt(self.model.d_model)
            )
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_indices.size(1)
            ).to(self.device)

            output = self.model.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask
            )

            # Get last token logits
            last_token_logits = self.model.fc_out(output[:, -1, :])
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(1)

            tgt_indices = torch.cat([tgt_indices, next_token], dim=1)

            # Check EOS
            eos_mask = next_token.squeeze(1) == self.tokenizer.eos_id
            finished = finished | eos_mask

            if finished.all():
                break

        return tgt_indices
