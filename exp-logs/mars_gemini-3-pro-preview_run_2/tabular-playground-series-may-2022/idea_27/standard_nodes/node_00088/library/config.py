import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from timm.layers import DropPath


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    CACHE_FILE = "processed_data.npz"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = "./submission/submission.csv"

    # --------------------------------------------------------------------------
    # Data Hyperparameters
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    SEQ_LEN = 10  # Length of f_27 string

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stream 1: Sequence
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "gelu"

    # Backbone
    STEM_DIM = 512
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    BLOCK_DROPOUT = 0.35  # Dropout inside residual block (before DropPath)
    DROP_PATH_MAX = 0.2  # Max stochastic depth rate

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 1024
    EPOCHS = 35
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler: Aggressive Step
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------
# Data Processing & Dataset
# ------------------------------------------------------------------------------
def load_and_process_data(config, load_cached_data=True):
    """
    Loads raw data, processes features (f_27 decomposition, scaling),
    and caches the result.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(config.WORKING_DIR, config.CACHE_FILE)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "X_seq_train": data["X_seq_train"],
            "X_cont_train": data["X_cont_train"],
            "y_train": data["y_train"],
            "X_seq_val": data["X_seq_val"],
            "X_cont_val": data["X_cont_val"],
            "y_val": data["y_val"],
            "X_seq_test": data["X_seq_test"],
            "X_cont_test": data["X_cont_test"],
            "test_ids": data["test_ids"],
            "vocab_size": int(data["vocab_size"]),
        }

    print("Cache not found or ignored. Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test_metadata.csv"))

    # 3. Load Raw Data
    # We load full train and test files once
    df_train_full = pd.read_csv(os.path.join(config.INPUT_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(config.INPUT_DIR, "test.csv"))

    # 4. Feature Engineering
    # 4.1 Decompose f_27
    def process_f27(series):
        # Split string into list of chars
        return series.apply(lambda x: list(x))

    # Combine for vocabulary building
    all_f27 = pd.concat([df_train_full["f_27"], df_test["f_27"]], axis=0)

    # Build Vocab
    # We use OrdinalEncoder to map chars to ints
    # Flatten to find unique chars
    unique_chars = sorted(list(set("".join(all_f27.unique()))))
    char_to_int = {
        c: i + 1 for i, c in enumerate(unique_chars)
    }  # 1-based indexing, 0 for padding if needed
    vocab_size = len(unique_chars) + 1

    def encode_f27(series):
        # Map chars to ints
        # Result is (N, 10) array
        # This is slow in pure python, let's use a vectorized approach if possible or list comp
        # List comp is fast enough for 1M rows
        return np.array(
            [[char_to_int[c] for c in s] for s in series.values], dtype=np.int32
        )

    X_seq_full = encode_f27(df_train_full["f_27"])
    X_seq_test = encode_f27(df_test["f_27"])

    # 4.2 Continuous Features
    cont_cols = [f"f_{i:02d}" for i in range(31) if f"f_{i:02d}" != "f_27"]

    scaler = StandardScaler()
    # Fit on training set defined by metadata (to avoid leakage from val)
    train_ids = set(train_meta["id"])
    train_mask = df_train_full["id"].isin(train_ids)

    X_cont_full = df_train_full[cont_cols].values.astype(np.float32)
    X_cont_test = df_test[cont_cols].values.astype(np.float32)

    scaler.fit(X_cont_full[train_mask])

    X_cont_full = scaler.transform(X_cont_full)
    X_cont_test = scaler.transform(X_cont_test)

    # 5. Split Train/Val based on Metadata
    # Create lookups
    id_to_idx = {id_: i for i, id_ in enumerate(df_train_full["id"])}

    train_indices = [id_to_idx[id_] for id_ in train_meta["id"]]
    val_indices = [id_to_idx[id_] for id_ in val_meta["id"]]

    X_seq_train = X_seq_full[train_indices]
    X_cont_train = X_cont_full[train_indices]
    y_train = df_train_full.loc[train_indices, "target"].values.astype(np.float32)

    X_seq_val = X_seq_full[val_indices]
    X_cont_val = X_cont_full[val_indices]
    y_val = df_train_full.loc[val_indices, "target"].values.astype(np.float32)

    test_ids = df_test["id"].values

    # 6. Cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        X_seq_train=X_seq_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_seq_val=X_seq_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_seq_test=X_seq_test,
        X_cont_test=X_cont_test,
        test_ids=test_ids,
        vocab_size=vocab_size,
    )

    return {
        "X_seq_train": X_seq_train,
        "X_cont_train": X_cont_train,
        "y_train": y_train,
        "X_seq_val": X_seq_val,
        "X_cont_val": X_cont_val,
        "y_val": y_val,
        "X_seq_test": X_seq_test,
        "X_cont_test": X_cont_test,
        "test_ids": test_ids,
        "vocab_size": vocab_size,
    }


class ManufacturingDataset(Dataset):
    def __init__(self, X_seq, X_cont, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_seq[idx], self.X_cont[idx], self.y[idx]
        return self.X_seq[idx], self.X_cont[idx]


def get_dataloaders(config, load_cached_data=True):
    data = load_and_process_data(config, load_cached_data)

    train_ds = ManufacturingDataset(
        data["X_seq_train"], data["X_cont_train"], data["y_train"]
    )
    val_ds = ManufacturingDataset(data["X_seq_val"], data["X_cont_val"], data["y_val"])
    test_ds = ManufacturingDataset(data["X_seq_test"], data["X_cont_test"], None)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_dl, val_dl, test_dl, data


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class GLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block with Stochastic Depth.
    Structure: x + DropPath(Dropout(GLU(Linear(BN(x)))))
    """

    def __init__(self, in_dim, drop_path=0.0, dropout=0.0):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        # GLU reduces dim by half, so we project to 2 * in_dim
        self.linear = nn.Linear(in_dim, in_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Init Linear for GLU
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.linear(out)
        out = F.glu(out, dim=-1)  # Halves dimension back to in_dim
        out = self.dropout(out)
        out = self.drop_path(out)
        return identity + out


class StochasticResFunnel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        # --- Stream 1: Categorical Sequence ---
        self.embedding = nn.Embedding(vocab_size, config.EMBED_DIM)
        # Learnable Positional Encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, config.SEQ_LEN, config.EMBED_DIM)
        )
        # Init Pos Enc with Low Variance Noise
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.EMBED_DIM * 4,
            dropout=config.TRANSFORMER_DROPOUT,
            activation=config.TRANSFORMER_ACTIVATION,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.TRANSFORMER_LAYERS
        )

        # Stream 1 Output Dim
        self.stream1_dim = config.SEQ_LEN * config.EMBED_DIM

        # --- Stream 2: Continuous ---
        # 30 continuous features
        self.stream2_dim = 30

        # --- Fusion ---
        fusion_in_dim = self.stream1_dim + self.stream2_dim
        self.stem = nn.Linear(fusion_in_dim, config.STEM_DIM)

        # --- Backbone ---
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_dim = config.STEM_DIM

        # Calculate total blocks for stochastic depth schedule
        total_blocks = len(config.BACKBONE_STAGES) * config.BLOCKS_PER_STAGE
        global_block_idx = 0

        for i, stage_dim in enumerate(config.BACKBONE_STAGES):
            # Downsample (except first stage if it matches stem, but here stem=512, stage1=512)
            if i > 0:
                # Projected Residual Connection for downsampling
                # We need to reduce width.
                # Standard ResNet uses stride=2, here we just project width.
                downsample = nn.Sequential(
                    nn.BatchNorm1d(current_dim), nn.Linear(current_dim, stage_dim)
                )
                self.downsamples.append(downsample)
            else:
                # If dimensions match (Stem 512 -> Stage 1 512), identity or simple mapping
                if current_dim != stage_dim:
                    self.downsamples.append(nn.Linear(current_dim, stage_dim))
                else:
                    self.downsamples.append(nn.Identity())

            # Blocks
            blocks = nn.ModuleList()
            for _ in range(config.BLOCKS_PER_STAGE):
                # Linear DropPath schedule
                dp_rate = config.DROP_PATH_MAX * (global_block_idx / total_blocks)
                blocks.append(
                    GLUBlock(stage_dim, drop_path=dp_rate, dropout=config.BLOCK_DROPOUT)
                )
                global_block_idx += 1

            self.stages.append(blocks)
            current_dim = stage_dim

        # --- Head ---
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x_seq, x_cont):
        # Stream 1
        x_s = self.embedding(x_seq)  # (B, L, D)
        x_s = x_s + self.pos_encoder
        x_s = self.transformer(x_s)
        x_s = x_s.flatten(1)  # (B, L*D)

        # Stream 2
        x_c = x_cont  # (B, 30)

        # Fusion
        x = torch.cat([x_s, x_c], dim=1)
        x = self.stem(x)  # No BN here

        # Backbone
        for downsample, stage_blocks in zip(self.downsamples, self.stages):
            x = downsample(x)
            for block in stage_blocks:
                x = block(x)

        # Head
        logits = self.head(x)
        return logits


def get_model(config, vocab_size):
    model = StochasticResFunnel(config, vocab_size)
    return model.to(config.DEVICE)
