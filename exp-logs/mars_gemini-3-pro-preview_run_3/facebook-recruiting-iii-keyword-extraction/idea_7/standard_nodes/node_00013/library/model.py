import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# Import configuration and utilities
from library.config import Config
from library.utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    calculate_f1_samples,
    optimize_f1_threshold,
)

# ==========================================
# 1. Model Architecture
# ==========================================


class DilatedWideAndDeep(nn.Module):
    """
    Subword-Level Dilated Wide-and-Deep Network.

    Consists of:
    1. Wide Stream: EmbeddingBag(sum) -> Linear. Captures keyword memorization.
    2. Deep Stream: Embedding -> Dilated CNNs -> MaxPool -> Linear. Captures n-gram context.
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim,
        num_filters,
        kernel_size,
        dilation_rates,
        dropout,
    ):
        super(DilatedWideAndDeep, self).__init__()

        # --- Wide Stream ---
        # Maps subwords directly to a hidden representation which is then projected to classes
        # This acts like a learnable TF-IDF / BoW model
        self.wide_embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="sum")
        self.wide_linear = nn.Linear(embed_dim, num_classes)

        # --- Deep Stream ---
        self.deep_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Parallel Dilated Convolutions
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    dilation=rate,
                    padding=(kernel_size - 1)
                    * rate
                    // 2,  # Same padding logic for dilated conv
                )
                for rate in dilation_rates
            ]
        )

        # Projection for Deep Stream
        # Input dim is num_filters * number of parallel branches
        self.deep_linear = nn.Linear(num_filters * len(dilation_rates), num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)

        # --- Wide Path ---
        # EmbeddingBag ignores padding index 0 effectively if trained well,
        # or we can mask it. Here we rely on the learned embedding for 0 to be negligible or handled.
        wide_out = self.wide_embedding(x)  # (Batch, Embed_Dim)
        wide_logits = self.wide_linear(wide_out)  # (Batch, Num_Classes)

        # --- Deep Path ---
        deep_emb = self.deep_embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        deep_emb = self.dropout(deep_emb)

        # Permute for Conv1d: (Batch, Embed_Dim, Seq_Len)
        deep_emb = deep_emb.permute(0, 2, 1)

        # Apply parallel convolutions and Global Max Pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv output: (Batch, Num_Filters, Seq_Len_Out)
            c = F.relu(conv(deep_emb))
            # Global Max Pooling: (Batch, Num_Filters)
            c = F.max_pool1d(c, c.shape[2]).squeeze(2)
            conv_outputs.append(c)

        # Concatenate features from all dilation rates
        deep_features = torch.cat(
            conv_outputs, dim=1
        )  # (Batch, Num_Filters * Num_Rates)
        deep_features = self.dropout(deep_features)

        deep_logits = self.deep_linear(deep_features)  # (Batch, Num_Classes)

        # --- Fusion ---
        # Element-wise sum of logits
        total_logits = wide_logits + deep_logits

        return total_logits


# ==========================================
# 2. Dataset
# ==========================================


class StackExchangeDataset(Dataset):
    def __init__(self, tokens, labels=None):
        """
        Args:
            tokens (np.ndarray): Tokenized inputs (N, Max_Len)
            labels (np.ndarray, optional): Binary labels (N, Num_Classes)
        """
        self.tokens = torch.from_numpy(tokens).long()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.tokens[idx], self.labels[idx]
        return self.tokens[idx]


# ==========================================
# 3. Data Processing Pipeline
# ==========================================


def train_tokenizer(texts, vocab_size, save_path):
    """Trains a BPE tokenizer on the provided texts."""
    print("Training BPE Tokenizer...")
    # Initialize ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Save texts to a temp file for tokenizer training (memory efficient)
    temp_text_path = os.path.join(Config.WORKING_DIR, "temp_train_text.txt")
    with open(temp_text_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    # Train
    tokenizer.train(
        files=[temp_text_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )

    # Save
    tokenizer.save(save_path)
    os.remove(temp_text_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer


def process_data(load_cached_data=True):
    """
    Loads raw data, tokenizes, encodes labels, and creates numpy arrays.
    Implements caching to avoid re-processing.
    """
    set_seed()

    # Paths for cached files
    files = {
        "train_tokens": Config.TRAIN_TOKENS_PATH,
        "train_labels": Config.TRAIN_LABELS_PATH,
        "val_tokens": Config.VAL_TOKENS_PATH,
        "val_labels": Config.VAL_LABELS_PATH,
        "test_tokens": Config.TEST_TOKENS_PATH,
        "test_ids": Config.TEST_IDS_PATH,
        "mlb_classes": os.path.join(Config.WORKING_DIR, "mlb_classes.npy"),
    }

    # Check cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in files.values())
        if all_exist and os.path.exists(Config.TOKENIZER_PATH):
            print("Loading cached data...")
            train_tokens = np.load(files["train_tokens"])
            train_labels = np.load(files["train_labels"])
            val_tokens = np.load(files["val_tokens"])
            val_labels = np.load(files["val_labels"])
            test_tokens = np.load(files["test_tokens"])
            test_ids = np.load(files["test_ids"])

            mlb = MultiLabelBinarizer()
            mlb.classes_ = np.load(files["mlb_classes"], allow_pickle=True)
            return (
                train_tokens,
                train_labels,
                val_tokens,
                val_labels,
                test_tokens,
                test_ids,
                mlb,
            )

    print("Processing data from scratch...")

    # 1. Load Metadata
    print("Loading metadata...")
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Debugging subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Using {Config.DEBUG_SIZE} samples.")
        df_train_meta = df_train_meta.iloc[: Config.DEBUG_SIZE]
        df_val_meta = df_val_meta.iloc[: int(Config.DEBUG_SIZE * 0.2)]
        df_test_meta = df_test_meta.iloc[: int(Config.DEBUG_SIZE * 0.2)]

    # 2. Load Raw Data
    print("Loading raw text data...")
    # We read the full raw files. Pandas is efficient enough for 5M rows on 220GB RAM.
    # Note: Body contains newlines, so we rely on standard CSV parsing.
    df_raw_train = pd.read_csv(
        Config.RAW_TRAIN_PATH, usecols=["Id", "Title", "Body", "Tags"]
    )
    df_raw_test = pd.read_csv(Config.RAW_TEST_PATH, usecols=["Id", "Title", "Body"])

    # Handle NaNs
    df_raw_train.fillna("", inplace=True)
    df_raw_test.fillna("", inplace=True)

    # Merge with metadata to get splits
    train_df = pd.merge(df_train_meta, df_raw_train, on="Id", how="inner")
    val_df = pd.merge(df_val_meta, df_raw_train, on="Id", how="inner")
    test_df = pd.merge(df_test_meta, df_raw_test, on="Id", how="inner")

    # Free memory
    del df_raw_train, df_raw_test, df_train_meta, df_val_meta, df_test_meta
    gc.collect()

    # 3. Text Preprocessing (Concat Title + Body)
    print("Concatenating text...")
    train_text = (train_df["Title"] + " " + train_df["Body"]).tolist()
    val_text = (val_df["Title"] + " " + val_df["Body"]).tolist()
    test_text = (test_df["Title"] + " " + test_df["Body"]).tolist()

    # 4. Tokenizer
    if os.path.exists(Config.TOKENIZER_PATH):
        print("Loading existing tokenizer...")
        tokenizer = ByteLevelBPETokenizer(
            os.path.join(Config.WORKING_DIR, "vocab.json"),
            os.path.join(Config.WORKING_DIR, "merges.txt"),
        )
        # Re-load using the json file wrapper if needed, but usually we just use the file
        # Actually ByteLevelBPETokenizer.from_file is better
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(Config.TOKENIZER_PATH)
    else:
        tokenizer = train_tokenizer(
            train_text, Config.VOCAB_SIZE, Config.TOKENIZER_PATH
        )
        # Reload as generic Tokenizer object for consistent API
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(Config.TOKENIZER_PATH)

    # Configure padding/truncation
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>", length=Config.MAX_LEN)
    tokenizer.enable_truncation(max_length=Config.MAX_LEN)

    def encode_batch(texts):
        # Batch encoding is faster
        encodings = tokenizer.encode_batch(texts)
        return np.array([e.ids for e in encodings], dtype=np.int32)

    print("Tokenizing Train...")
    train_tokens = encode_batch(train_text)
    print("Tokenizing Val...")
    val_tokens = encode_batch(val_text)
    print("Tokenizing Test...")
    test_tokens = encode_batch(test_text)

    # 5. Label Encoding
    print("Encoding Labels...")
    # Parse tags
    train_tags = train_df["Tags_y"].apply(lambda x: str(x).split()).tolist()
    val_tags = val_df["Tags_y"].apply(lambda x: str(x).split()).tolist()

    # Select Top K tags based on training data
    from collections import Counter

    all_tags = [t for tags in train_tags for t in tags]
    top_tags = [t for t, c in Counter(all_tags).most_common(Config.TOP_K_TAGS)]

    mlb = MultiLabelBinarizer(classes=top_tags)
    mlb.fit(
        train_tags
    )  # Fit on provided tags (will only keep classes provided in init)

    # Transform
    # Note: We use float32 for BCEWithLogitsLoss
    train_labels = mlb.transform(train_tags).astype(np.float32)
    val_labels = mlb.transform(val_tags).astype(np.float32)

    test_ids = test_df["Id"].values.astype(np.int32)

    # 6. Save to Cache
    print("Saving to cache...")
    np.save(files["train_tokens"], train_tokens)
    np.save(files["train_labels"], train_labels)
    np.save(files["val_tokens"], val_tokens)
    np.save(files["val_labels"], val_labels)
    np.save(files["test_tokens"], test_tokens)
    np.save(files["test_ids"], test_ids)
    np.save(files["mlb_classes"], mlb.classes_)

    return (
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        test_ids,
        mlb,
    )


# ==========================================
# 4. Training Loop
# ==========================================


def train_model(train_tokens, train_labels, val_tokens, val_labels):
    print("Initializing Model...")
    device = get_device()

    # Dataset & Loader
    train_ds = StackExchangeDataset(train_tokens, train_labels)
    val_ds = StackExchangeDataset(val_tokens, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = DilatedWideAndDeep(
        vocab_size=Config.VOCAB_SIZE,
        num_classes=len(train_labels[0]),
        embed_dim=Config.EMBED_DIM,
        num_filters=Config.NUM_FILTERS,
        kernel_size=Config.KERNEL_SIZE,
        dilation_rates=Config.DILATION_RATES,
        dropout=Config.DROPOUT,
    ).to(device)

    # Optimization
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_f1 = 0.0
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                logits = model(tokens)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                with autocast():
                    logits = model(tokens)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).float().cpu())
                all_targets.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)

        # Calculate F1
        y_pred_prob = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        # Optimize threshold on validation set
        best_thr, val_f1 = optimize_f1_threshold(y_true, y_pred_prob)

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {dt:.1f}s | "
            f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | "
            f"Val F1: {val_f1:.5f} (Thr: {best_thr:.2f})"
        )

        # Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_f1, Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! F1: {best_f1:.5f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    return best_f1, best_thr


# ==========================================
# 5. Inference
# ==========================================


def generate_submission(test_tokens, test_ids, mlb, threshold):
    print("Generating Submission...")
    device = get_device()

    # Load best model
    # We need to re-initialize the model structure to load weights
    model = DilatedWideAndDeep(
        vocab_size=Config.VOCAB_SIZE,
        num_classes=len(mlb.classes_),
        embed_dim=Config.EMBED_DIM,
        num_filters=Config.NUM_FILTERS,
        kernel_size=Config.KERNEL_SIZE,
        dilation_rates=Config.DILATION_RATES,
        dropout=Config.DROPOUT,
    ).to(device)

    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    model.eval()
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with F1 {checkpoint['score']:.5f}"
    )

    test_ds = StackExchangeDataset(test_tokens)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []

    with torch.no_grad():
        for tokens in test_loader:
            tokens = tokens.to(device)
            with autocast():
                logits = model(tokens)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Apply threshold
            binary_preds = (probs >= threshold).astype(int)

            # Convert to tags
            # mlb.inverse_transform returns list of tuples of tags
            batch_tags = mlb.inverse_transform(binary_preds)
            all_preds.extend([" ".join(tags) for tags in batch_tags])

    # Create DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Tags": all_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# ==========================================
# 6. Main Execution
# ==========================================


def run_pipeline():
    # 1. Prepare Data
    train_tokens, train_labels, val_tokens, val_labels, test_tokens, test_ids, mlb = (
        process_data(load_cached_data=True)
    )

    # 2. Train
    best_f1, best_thr = train_model(train_tokens, train_labels, val_tokens, val_labels)

    # 3. Predict
    generate_submission(test_tokens, test_ids, mlb, best_thr)
