import pandas as pd
import numpy as np
import os
import gc
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import (
    setup_logger,
    load_or_process,
    calculate_accuracy,
    format_submission_id,
)
from library.data_processor import get_data

logger = setup_logger("NgramModel")


class CharSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        self.decoder_embedding = nn.Embedding(output_dim, embed_dim)
        self.decoder = nn.GRU(embed_dim, hidden_dim * 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)

        # Encoder
        embedded = self.embedding(src)
        _, hidden = self.encoder(embedded)

        # Reshape encoder hidden to match decoder (combine bidirectional)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)

        # Decoder
        input_step = trg[:, 0]

        for t in range(1, max_len):
            embedded_dec = self.decoder_embedding(input_step.unsqueeze(1))
            output, hidden = self.decoder(embedded_dec, hidden)
            prediction = self.fc(output.squeeze(1))
            outputs[:, t, :] = prediction

            top1 = prediction.argmax(1)
            input_step = (
                trg[:, t] if np.random.random() < teacher_forcing_ratio else top1
            )

        return outputs

    def predict_step(self, src, max_len, sos_idx, eos_idx):
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            embedded = self.embedding(src)
            _, hidden = self.encoder(embedded)
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)

            input_step = torch.full(
                (batch_size,), sos_idx, dtype=torch.long, device=self.device
            )
            predictions = []

            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for _ in range(max_len):
                embedded_dec = self.decoder_embedding(input_step.unsqueeze(1))
                output, hidden = self.decoder(embedded_dec, hidden)
                pred = self.fc(output.squeeze(1)).argmax(1)

                predictions.append(pred)
                input_step = pred

                finished |= pred == eos_idx
                if finished.all():
                    break

            return torch.stack(predictions, dim=1)


class TextDataset(Dataset):
    def __init__(self, data, char2idx, max_len):
        self.data = data
        self.char2idx = char2idx
        self.max_len = max_len
        self.sos_idx = char2idx["<SOS>"]
        self.eos_idx = char2idx["<EOS>"]
        self.pad_idx = char2idx["<PAD>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt, trg_txt = self.data[idx]
        return self.encode(src_txt), self.encode(trg_txt)

    def encode(self, text):
        indices = (
            [self.sos_idx]
            + [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in text]
            + [self.eos_idx]
        )
        if len(indices) < self.max_len:
            indices += [self.pad_idx] * (self.max_len - len(indices))
        else:
            indices = indices[: self.max_len - 1] + [self.eos_idx]
        return torch.tensor(indices, dtype=torch.long)


class HierarchicalLookupModel:
    """
    A Non-Parametric Hierarchical N-gram model for text normalization.
    It memorizes the most frequent mapping for tokens based on varying context window sizes.
    """

    def __init__(self):
        self.l1_dict = {}
        self.l2_left_dict = {}
        self.l2_right_dict = {}
        self.l3_dict = {}

        # Neural Fallback components
        self.neural_model = None
        self.char2idx = None
        self.idx2char = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, df: pd.DataFrame):
        logger.info("Fitting HierarchicalLookupModel...")
        df[Config.INPUT_COL] = df[Config.INPUT_COL].astype(str)
        df[Config.TARGET_COL] = df[Config.TARGET_COL].astype(str)

        logger.info("Generating context columns...")
        df["prev"] = df[Config.INPUT_COL].shift(1).fillna(Config.BOS_TOKEN)
        df["next"] = df[Config.INPUT_COL].shift(-1).fillna(Config.EOS_TOKEN)

        mask = (df[Config.INPUT_COL] != Config.BOS_TOKEN) & (
            df[Config.INPUT_COL] != Config.EOS_TOKEN
        )
        df_clean = df[mask].copy()
        gc.collect()

        # --- N-gram Lookup Fitting ---
        def get_mode_map(data, keys, target):
            logger.info(f"Computing mode map for keys: {keys}")
            counts = data.groupby(keys + [target]).size().reset_index(name="count")
            counts = counts.sort_values("count", ascending=False)
            best = counts.drop_duplicates(subset=keys)
            if len(keys) == 1:
                return dict(zip(best[keys[0]], best[target]))
            else:
                return dict(zip(zip(*[best[k] for k in keys]), best[target]))

        self.l3_dict = get_mode_map(
            df_clean, ["prev", Config.INPUT_COL, "next"], Config.TARGET_COL
        )
        self.l2_left_dict = get_mode_map(
            df_clean, ["prev", Config.INPUT_COL], Config.TARGET_COL
        )
        self.l2_right_dict = get_mode_map(
            df_clean, [Config.INPUT_COL, "next"], Config.TARGET_COL
        )
        self.l1_dict = get_mode_map(df_clean, [Config.INPUT_COL], Config.TARGET_COL)

        logger.info(f"Lookup tables fitted.")

        # --- Neural Fallback Fitting ---
        logger.info(
            "Fitting Neural Fallback Model (Cite solution_lesson_node_00001)..."
        )

        # Select training data: tokens that changed (normalization happened)
        # This focuses the model on the "tail" problem (numbers, dates, etc)
        change_mask = df_clean[Config.INPUT_COL] != df_clean[Config.TARGET_COL]

        df_neural = df_clean[change_mask].drop_duplicates(
            subset=[Config.INPUT_COL, Config.TARGET_COL]
        )
        logger.info(f"Neural training samples: {len(df_neural)}")

        # Build Vocabulary
        all_text = "".join(
            df_neural[Config.INPUT_COL].tolist() + df_neural[Config.TARGET_COL].tolist()
        )
        unique_chars = sorted(list(set(all_text)))
        self.char2idx = {c: i + 4 for i, c in enumerate(unique_chars)}
        self.char2idx["<PAD>"] = 0
        self.char2idx["<SOS>"] = 1
        self.char2idx["<EOS>"] = 2
        self.char2idx["<UNK>"] = 3
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        # Dataset & Loader
        train_data = list(
            zip(df_neural[Config.INPUT_COL], df_neural[Config.TARGET_COL])
        )
        dataset = TextDataset(train_data, self.char2idx, Config.NEURAL_MAX_LEN)
        dataloader = DataLoader(
            dataset, batch_size=Config.NEURAL_BATCH_SIZE, shuffle=True, drop_last=True
        )

        # Initialize Model
        self.neural_model = CharSeq2Seq(
            len(self.char2idx),
            len(self.char2idx),
            Config.NEURAL_EMBED_DIM,
            Config.NEURAL_HIDDEN_DIM,
            self.device,
        ).to(self.device)

        optimizer = optim.Adam(self.neural_model.parameters(), lr=Config.NEURAL_LR)
        criterion = nn.CrossEntropyLoss(ignore_index=self.char2idx["<PAD>"])

        self.neural_model.train()
        for epoch in range(Config.NEURAL_EPOCHS):
            total_loss = 0
            for src, trg in dataloader:
                src, trg = src.to(self.device), trg.to(self.device)
                optimizer.zero_grad()

                output = self.neural_model(src, trg)
                # output: [batch, len, vocab], trg: [batch, len]
                output_dim = output.shape[-1]
                loss = criterion(
                    output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1)
                )

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(
                f"Epoch {epoch+1}/{Config.NEURAL_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}"
            )

    def predict(self, df: pd.DataFrame) -> list:
        logger.info("Predicting with HierarchicalLookupModel...")
        df[Config.INPUT_COL] = df[Config.INPUT_COL].astype(str)

        df["prev"] = df[Config.INPUT_COL].shift(1).fillna(Config.BOS_TOKEN)
        df["next"] = df[Config.INPUT_COL].shift(-1).fillna(Config.EOS_TOKEN)

        mask = (df[Config.INPUT_COL] != Config.BOS_TOKEN) & (
            df[Config.INPUT_COL] != Config.EOS_TOKEN
        )
        df_clean = df[mask].copy()

        # --- Hierarchical Lookup ---
        logger.info("Mapping L3 (Trigram)...")
        l3_keys = list(
            zip(df_clean["prev"], df_clean[Config.INPUT_COL], df_clean["next"])
        )
        preds = pd.Series(l3_keys, index=df_clean.index).map(self.l3_dict)

        logger.info("Mapping L2 Left (Bigram)...")
        l2_left_keys = list(zip(df_clean["prev"], df_clean[Config.INPUT_COL]))
        preds = preds.combine_first(
            pd.Series(l2_left_keys, index=df_clean.index).map(self.l2_left_dict)
        )

        logger.info("Mapping L2 Right (Bigram)...")
        l2_right_keys = list(zip(df_clean[Config.INPUT_COL], df_clean["next"]))
        preds = preds.combine_first(
            pd.Series(l2_right_keys, index=df_clean.index).map(self.l2_right_dict)
        )

        logger.info("Mapping L1 (Unigram)...")
        preds = preds.combine_first(df_clean[Config.INPUT_COL].map(self.l1_dict))

        # --- Neural Fallback ---
        # Identify remaining NaNs that look like they need normalization (contain digits)
        missing_mask = preds.isna()
        # Regex: contains digit
        digit_mask = df_clean[Config.INPUT_COL].str.contains(r"\d")
        fallback_mask = missing_mask & digit_mask

        if fallback_mask.any() and self.neural_model is not None:
            logger.info(f"Applying Neural Fallback to {fallback_mask.sum()} tokens...")
            tokens_to_predict = df_clean.loc[fallback_mask, Config.INPUT_COL].unique()

            # Predict in batches
            token_map = {}
            batch_size = Config.NEURAL_BATCH_SIZE

            # Helper to encode
            def encode(text):
                indices = (
                    [self.char2idx["<SOS>"]]
                    + [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in text]
                    + [self.char2idx["<EOS>"]]
                )
                if len(indices) < Config.NEURAL_MAX_LEN:
                    indices += [self.char2idx["<PAD>"]] * (
                        Config.NEURAL_MAX_LEN - len(indices)
                    )
                else:
                    indices = indices[: Config.NEURAL_MAX_LEN - 1] + [
                        self.char2idx["<EOS>"]
                    ]
                return indices

            for i in range(0, len(tokens_to_predict), batch_size):
                batch_tokens = tokens_to_predict[i : i + batch_size]
                batch_indices = [encode(t) for t in batch_tokens]
                src = torch.tensor(batch_indices, dtype=torch.long).to(self.device)

                output = self.neural_model.predict_step(
                    src,
                    Config.NEURAL_MAX_LEN,
                    self.char2idx["<SOS>"],
                    self.char2idx["<EOS>"],
                )

                # Decode
                for j, token in enumerate(batch_tokens):
                    pred_indices = output[j].cpu().numpy()
                    pred_chars = []
                    for idx in pred_indices:
                        if idx == self.char2idx["<EOS>"]:
                            break
                        if idx not in [self.char2idx["<SOS>"], self.char2idx["<PAD>"]]:
                            pred_chars.append(self.idx2char.get(idx, ""))
                    token_map[token] = "".join(pred_chars)

            # Apply predictions
            preds_neural = df_clean.loc[fallback_mask, Config.INPUT_COL].map(token_map)
            preds = preds.combine_first(preds_neural)

        logger.info("Applying Identity fallback...")
        preds = preds.fillna(df_clean[Config.INPUT_COL])

        return preds.tolist()

    def get_stats(self):
        """Returns the learned dictionaries and neural model state."""
        stats = {
            "l1": self.l1_dict,
            "l2_left": self.l2_left_dict,
            "l2_right": self.l2_right_dict,
            "l3": self.l3_dict,
        }
        if self.neural_model is not None:
            stats["neural_state"] = self.neural_model.state_dict()
            stats["char2idx"] = self.char2idx
            stats["idx2char"] = self.idx2char
        return stats

    def load_stats(self, stats):
        """Loads dictionaries from a stats object."""
        self.l1_dict = stats["l1"]
        self.l2_left_dict = stats["l2_left"]
        self.l2_right_dict = stats["l2_right"]
        self.l3_dict = stats["l3"]

        if "neural_state" in stats:
            self.char2idx = stats["char2idx"]
            self.idx2char = stats["idx2char"]
            self.neural_model = CharSeq2Seq(
                len(self.char2idx),
                len(self.char2idx),
                Config.NEURAL_EMBED_DIM,
                Config.NEURAL_HIDDEN_DIM,
                self.device,
            ).to(self.device)
            self.neural_model.load_state_dict(stats["neural_state"])
            self.neural_model.eval()


def _compute_stats_wrapper():
    """
    Internal wrapper to load training data and fit the model.
    Used by load_or_process for caching.
    """
    # Load processed training data (with BOS/EOS)
    df_train = get_data("train", load_cached_data=True)

    model = HierarchicalLookupModel()
    model.fit(df_train)

    return model.get_stats()


def train_model(load_cached_data=True):
    """
    Trains the HierarchicalLookupModel or loads cached statistics.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed stats.

    Returns:
        HierarchicalLookupModel: The trained model instance.
    """
    # Use load_or_process to handle caching of the model statistics (dictionaries)
    stats = load_or_process(
        Config.MODEL_STATS_PATH,
        _compute_stats_wrapper,
        load_cached_data=load_cached_data,
    )

    model = HierarchicalLookupModel()
    model.load_stats(stats)
    return model


def evaluate_model(model, df_val=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The trained model.
        df_val: Optional validation dataframe. If None, loads from config.

    Returns:
        float: Accuracy score.
    """
    if df_val is None:
        df_val = get_data("val", load_cached_data=True)

    logger.info("Evaluating model on validation set...")
    y_pred = model.predict(df_val)

    # Extract ground truth
    # Filter boundary tokens to match prediction output
    mask = (df_val[Config.INPUT_COL] != Config.BOS_TOKEN) & (
        df_val[Config.INPUT_COL] != Config.EOS_TOKEN
    )
    y_true = df_val.loc[mask, Config.TARGET_COL].astype(str).tolist()

    acc = calculate_accuracy(y_true, y_pred)
    logger.info(f"Validation Accuracy: {acc}")
    return acc


def generate_submission(model):
    """
    Generates predictions for the test set and saves the submission file.
    """
    logger.info("Generating submission...")
    df_test = get_data("test", load_cached_data=True)

    # Generate predictions
    y_pred = model.predict(df_test)

    # Prepare submission dataframe
    # Filter test df to get IDs (excluding boundaries)
    mask = (df_test[Config.INPUT_COL] != Config.BOS_TOKEN) & (
        df_test[Config.INPUT_COL] != Config.EOS_TOKEN
    )
    df_sub = df_test.loc[mask].copy()

    # Create submission ID: sentence_id + "_" + token_id
    # Using vectorized string concatenation for speed
    df_sub[Config.SUBMISSION_ID_COL] = (
        df_sub[Config.SENTENCE_ID_COL].astype(str)
        + "_"
        + df_sub[Config.TOKEN_ID_COL].astype(str)
    )

    # Assign predictions
    df_sub[Config.TARGET_COL] = y_pred

    # Select required columns
    submission = df_sub[[Config.SUBMISSION_ID_COL, Config.TARGET_COL]]

    # Save to CSV
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
