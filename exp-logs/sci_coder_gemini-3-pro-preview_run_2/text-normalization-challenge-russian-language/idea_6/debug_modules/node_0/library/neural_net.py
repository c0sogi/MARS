import os
import re
import math
import time
import random
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict, Optional

from library.config import Config
from library.utils import set_seed, is_semiotic, save_parquet_cache, load_parquet_cache
from library.hfbb_engine import HFBB

# ==========================================
# 1. Tokenizers
# ==========================================


class CharTokenizer:
    """
    Character-level tokenizer for the input source.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.specials = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<SEP>"]

    def fit(self, texts: List[str]):
        """Builds vocabulary from a list of strings."""
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Create mappings
        self.char2idx = {token: idx for idx, token in enumerate(self.specials)}
        start_idx = len(self.specials)

        for i, char in enumerate(sorted_chars):
            self.char2idx[char] = start_idx + i

        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

        if self.vocab_size > Config.CHAR_VOCAB_SIZE:
            print(
                f"Warning: Char vocab size {self.vocab_size} exceeds limit {Config.CHAR_VOCAB_SIZE}."
            )

    def encode(self, text: str) -> List[int]:
        """Converts string to list of IDs."""
        return [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in str(text)]

    def decode(self, ids: List[int]) -> str:
        """Converts list of IDs to string."""
        return "".join(
            [
                self.idx2char.get(idx, "")
                for idx in ids
                if idx
                not in [
                    self.char2idx["<PAD>"],
                    self.char2idx["<SOS>"],
                    self.char2idx["<EOS>"],
                    self.char2idx["<SEP>"],
                ]
            ]
        )

    @property
    def pad_token_id(self):
        return self.char2idx["<PAD>"]

    @property
    def sep_token_id(self):
        return self.char2idx["<SEP>"]


class TargetBPETokenizer:
    """
    Wrapper for SentencePiece BPE tokenizer for the target text.
    """

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.model_prefix = Config.TOKENIZER_PREFIX
        self.model_path = f"{self.model_prefix}.model"

    def train(self, texts: List[str], vocab_size: int = Config.TARGET_VOCAB_SIZE):
        """Trains the SentencePiece model."""
        if os.path.exists(self.model_path):
            print(f"BPE model already exists at {self.model_path}. Loading...")
            self.sp.Load(self.model_path)
            return

        print(f"Training BPE model on {len(texts)} samples...")

        # Save texts to temp file for spm training
        temp_file = os.path.join(Config.WORKING_DIR, "temp_bpe_train.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(str(text) + "\n")

        # Train
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=self.model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
        )

        if os.path.exists(temp_file):
            os.remove(temp_file)
        self.sp.Load(self.model_path)
        print("BPE training complete.")

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BPE model not found at {self.model_path}")
        self.sp.Load(self.model_path)

    def encode(self, text: str) -> List[int]:
        return self.sp.encode_as_ids(str(text))

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode_ids(ids)

    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def pad_id(self):
        return 0

    @property
    def unk_id(self):
        return 1

    @property
    def bos_id(self):
        return 2

    @property
    def eos_id(self):
        return 3


# ==========================================
# 2. Data Processing (Residual Generation)
# ==========================================


class ResidualGenerator:
    """
    Handles the generation of the residual dataset via Jackknifing (for train)
    and standard inference (for val).
    """

    @staticmethod
    def _vectorized_predict(hfbb: HFBB, df: pd.DataFrame) -> pd.Series:
        """
        Optimized prediction using vectorized map lookups.
        """
        # Prepare context columns if not present
        if "prev" not in df.columns or "next" not in df.columns:
            df = df.copy()
            df["before"] = df["before"].astype(str)
            df["prev"] = df["before"].shift(1).fillna("<START>")
            df["next"] = df["before"].shift(-1).fillna("<END>")

            # Mask boundaries
            is_start = df["sentence_id"] != df["sentence_id"].shift(1)
            df.loc[is_start, "prev"] = "<START>"
            is_end = df["sentence_id"] != df["sentence_id"].shift(-1)
            df.loc[is_end, "next"] = "<END>"

        # Create keys
        trigram_keys = list(zip(df["prev"], df["before"], df["next"]))
        bigram_prev_keys = list(zip(df["prev"], df["before"]))
        bigram_next_keys = list(zip(df["before"], df["next"]))
        unigram_keys = df["before"].tolist()

        # 1. Trigram
        preds = pd.Series(trigram_keys).map(hfbb.trigram_map)

        # 2. Bigram Prev
        mask = preds.isna()
        if mask.any():
            preds.loc[mask] = pd.Series(bigram_prev_keys)[mask].map(
                hfbb.bigram_prev_map
            )

        # 3. Bigram Next
        mask = preds.isna()
        if mask.any():
            preds.loc[mask] = pd.Series(bigram_next_keys)[mask].map(
                hfbb.bigram_next_map
            )

        # 4. Unigram
        mask = preds.isna()
        if mask.any():
            preds.loc[mask] = pd.Series(unigram_keys)[mask].map(hfbb.unigram_map)

        return preds

    @staticmethod
    def get_train_residuals(load_cached_data: bool = True) -> pd.DataFrame:
        """
        Generates training residuals using K-Fold Jackknifing.
        """
        cache_path = Config.RESIDUAL_TRAIN_CACHE
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading train residuals from {cache_path}")
            return load_parquet_cache(cache_path)

        print("Generating training residuals (Jackknifing)...")
        df_train = pd.read_csv(Config.TRAIN_FILE)

        if Config.DEBUG:
            df_train = df_train.iloc[: Config.DEBUG_SIZE].copy()
            print(f"DEBUG: Reduced train size to {len(df_train)}")

        kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.SEED)
        sentences = df_train["sentence_id"].unique()

        residuals_list = []

        for fold, (train_sent_idx, val_sent_idx) in enumerate(kf.split(sentences)):
            print(f"  Processing Fold {fold + 1}/{Config.K_FOLDS}...")

            train_sents = sentences[train_sent_idx]
            val_sents = sentences[val_sent_idx]

            fold_train_df = df_train[df_train["sentence_id"].isin(train_sents)].copy()
            fold_val_df = df_train[df_train["sentence_id"].isin(val_sents)].copy()

            # Train temporary HFBB
            temp_hfbb = HFBB()
            temp_dir = os.path.join(Config.WORKING_DIR, f"temp_fold_{fold}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_hfbb.cache_paths = {
                k: os.path.join(temp_dir, os.path.basename(v))
                for k, v in temp_hfbb.cache_paths.items()
            }

            temp_hfbb.fit(fold_train_df, load_cached_data=False)

            preds = ResidualGenerator._vectorized_predict(temp_hfbb, fold_val_df)

            fold_val_df["hfbb_pred"] = preds.fillna("<MISSING>")
            is_mismatch = fold_val_df["hfbb_pred"] != fold_val_df["after"]
            is_semiotic_mask = fold_val_df["before"].astype(str).apply(is_semiotic)

            fold_residuals = fold_val_df[is_mismatch & is_semiotic_mask].copy()
            residuals_list.append(fold_residuals)

            shutil.rmtree(temp_dir)

        if residuals_list:
            all_residuals = pd.concat(residuals_list, ignore_index=True)
        else:
            all_residuals = pd.DataFrame(columns=df_train.columns)

        print(f"Generated {len(all_residuals)} training residuals.")
        save_parquet_cache(all_residuals, cache_path)
        return all_residuals

    @staticmethod
    def get_val_residuals(load_cached_data: bool = True) -> pd.DataFrame:
        """
        Generates validation residuals by applying full HFBB to validation set.
        """
        cache_path = Config.RESIDUAL_VAL_CACHE
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading val residuals from {cache_path}")
            return load_parquet_cache(cache_path)

        print("Generating validation residuals...")
        df_train = pd.read_csv(Config.TRAIN_FILE)
        df_val = pd.read_csv(Config.VAL_FILE)

        if Config.DEBUG:
            df_train = df_train.iloc[: Config.DEBUG_SIZE]
            df_val = df_val.iloc[: Config.DEBUG_SIZE]

        hfbb = HFBB()
        hfbb.fit(df_train, load_cached_data=True)

        preds = ResidualGenerator._vectorized_predict(hfbb, df_val)

        df_val["hfbb_pred"] = preds.fillna("<MISSING>")
        is_mismatch = df_val["hfbb_pred"] != df_val["after"]
        is_semiotic_mask = df_val["before"].astype(str).apply(is_semiotic)

        residuals = df_val[is_mismatch & is_semiotic_mask].copy()

        print(f"Generated {len(residuals)} validation residuals.")
        save_parquet_cache(residuals, cache_path)
        return residuals


# ==========================================
# 3. Dataset
# ==========================================


class ResidualDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        char_tokenizer: CharTokenizer,
        bpe_tokenizer: TargetBPETokenizer,
        train_mode: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.char_tokenizer = char_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.train_mode = train_mode
        self.context_window = Config.CONTEXT_WINDOW_CHARS

        if "prev" not in self.df.columns:
            self.df["prev"] = self.df["before"].shift(1).fillna("<START>")
            self.df["next"] = self.df["before"].shift(-1).fillna("<END>")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        prev_ctx = str(row["prev"])[-self.context_window :]
        curr_tok = str(row["before"])
        next_ctx = str(row["next"])[: self.context_window]

        input_str = prev_ctx + "<SEP>" + curr_tok + "<SEP>" + next_ctx
        src_ids = self.char_tokenizer.encode(input_str)

        if self.train_mode:
            target_str = str(row["after"])
            tgt_ids = self.bpe_tokenizer.encode(target_str)
            tgt_ids = (
                [self.bpe_tokenizer.bos_id] + tgt_ids + [self.bpe_tokenizer.eos_id]
            )
            return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
                tgt_ids, dtype=torch.long
            )
        else:
            return torch.tensor(src_ids, dtype=torch.long)


# ==========================================
# 4. Model Architecture
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CharToSubwordTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.NHEAD,
        num_encoder_layers: int = Config.NUM_ENCODER_LAYERS,
        num_decoder_layers: int = Config.NUM_DECODER_LAYERS,
        dim_feedforward: int = Config.DIM_FEEDFORWARD,
        dropout: float = Config.DROPOUT,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_key_padding_mask = src == self.pad_idx
        tgt_key_padding_mask = tgt == self.pad_idx
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(
            src.device
        )

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(output)

    def predict(self, src, max_len: int, bos_id: int, eos_id: int):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_key_padding_mask = src == self.pad_idx
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        ys = torch.ones(batch_size, 1).fill_(bos_id).type(torch.long).to(device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for i in range(max_len - 1):
            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))

            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            prob = self.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)

            ys = torch.cat([ys, next_word], dim=1)
            finished |= next_word.squeeze(1) == eos_id
            if finished.all():
                break
        return ys


# ==========================================
# 5. Training Loop
# ==========================================


class ResidualTrainer:
    def __init__(self):
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

    def run(self):
        df_train_res = ResidualGenerator.get_train_residuals()
        df_val_res = ResidualGenerator.get_val_residuals()

        if len(df_train_res) == 0:
            print("No training residuals found! Skipping training.")
            return

        char_tokenizer = CharTokenizer()
        char_tokenizer.fit(
            df_train_res["before"].tolist()
            + df_train_res["prev"].tolist()
            + df_train_res["next"].tolist()
        )

        bpe_tokenizer = TargetBPETokenizer()
        bpe_tokenizer.train(df_train_res["after"].tolist())

        train_dataset = ResidualDataset(
            df_train_res, char_tokenizer, bpe_tokenizer, train_mode=True
        )
        val_dataset = ResidualDataset(
            df_val_res, char_tokenizer, bpe_tokenizer, train_mode=True
        )

        def collate_wrapper(batch):
            src_batch = [item[0] for item in batch]
            tgt_batch = [item[1] for item in batch]
            src_padded = pad_sequence(
                src_batch, batch_first=True, padding_value=char_tokenizer.pad_token_id
            )
            tgt_padded = pad_sequence(
                tgt_batch, batch_first=True, padding_value=bpe_tokenizer.pad_id
            )
            return src_padded, tgt_padded

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_wrapper,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_wrapper,
            pin_memory=True,
        )

        model = CharToSubwordTransformer(
            src_vocab_size=char_tokenizer.vocab_size,
            tgt_vocab_size=bpe_tokenizer.vocab_size(),
            pad_idx=char_tokenizer.pad_token_id,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(
            ignore_index=bpe_tokenizer.pad_id, label_smoothing=Config.LABEL_SMOOTHING
        )

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {len(df_train_res)} samples...")

        for epoch in range(Config.NUM_EPOCHS):
            model.train()
            total_loss = 0

            for src, tgt in train_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                optimizer.zero_grad()
                logits = model(src, tgt_input)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), Config.GRADIENT_CLIP_VAL
                )
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    logits = model(src, tgt_input)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                    )
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "char_vocab": char_tokenizer.char2idx,
                        "config": {
                            "src_vocab_size": char_tokenizer.vocab_size,
                            "tgt_vocab_size": bpe_tokenizer.vocab_size(),
                        },
                    },
                    Config.TRANSFORMER_CHECKPOINT,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
        print("Training complete.")
