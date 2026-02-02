import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# ==========================================
# CONFIGURATION
# ==========================================


class Config:
    # Data Paths
    TRAIN_PARQUET = "./metadata/train.parquet"
    VAL_PARQUET = "./metadata/val.parquet"
    TEST_PARQUET = "./metadata/test.parquet"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = "submission.csv"
    CACHE_DIR = "./working/idea_63"

    # Model Hyperparameters
    SEQ_VOCAB_SIZE = 4  # A, G, C, U
    LOOP_VOCAB_SIZE = 7  # S, M, I, B, H, E, X
    SEQ_EMBED_DIM = 128
    LOOP_EMBED_DIM = 64
    DIST_EMBED_DIM = 64
    HIDDEN_DIM = 512
    NUM_LAYERS = 6
    DROPOUT = 0.2

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    EPOCHS = 20
    NUM_TARGETS = 3  # reactivity, deg_Mg_pH10, deg_Mg_50C
    SEQ_LEN = 107
    PRED_LEN = 68

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 42


# ==========================================
# DATA PROCESSING
# ==========================================


def get_structure_distance_map(structure):
    # Parse dot-bracket to find pairs
    stack = []
    indices = np.zeros(len(structure), dtype=np.int32)  # Default 0 for unpaired

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                # Signed distance
                indices[start_idx] = i - start_idx
                indices[i] = start_idx - i
    return indices


def process_data(df, is_test=False):
    # Tokenizers
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = df["id"].values
    sequences = []
    loops = []
    distances = []

    for _, row in df.iterrows():
        # Sequence
        seq_ints = [seq_map.get(c, 0) for c in row["sequence"]]
        sequences.append(seq_ints)

        # Loop
        loop_ints = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loops.append(loop_ints)

        # Structure Distance
        dist = get_structure_distance_map(row["structure"])
        distances.append(dist)

    sequences = np.array(sequences, dtype=np.int32)
    loops = np.array(loops, dtype=np.int32)
    distances = np.array(distances, dtype=np.int32)

    if is_test:
        return ids, sequences, loops, distances
    else:
        # Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # These are lists in the parquet dataframe
        t1 = np.vstack(df["reactivity"].values)
        t2 = np.vstack(df["deg_Mg_pH10"].values)
        t3 = np.vstack(df["deg_Mg_50C"].values)

        # Stack depth-wise: (N, 68, 3)
        targets = np.dstack([t1, t2, t3]).astype(np.float32)
        return ids, sequences, loops, distances, targets


def load_and_cache_data(parquet_path, cache_name, is_test=False, load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if is_test:
            return data["ids"], data["sequences"], data["loops"], data["distances"]
        else:
            return (
                data["ids"],
                data["sequences"],
                data["loops"],
                data["distances"],
                data["targets"],
            )

    # print(f"Processing data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    processed = process_data(df, is_test=is_test)

    if is_test:
        np.savez(
            cache_path,
            ids=processed[0],
            sequences=processed[1],
            loops=processed[2],
            distances=processed[3],
        )
    else:
        np.savez(
            cache_path,
            ids=processed[0],
            sequences=processed[1],
            loops=processed[2],
            distances=processed[3],
            targets=processed[4],
        )

    return processed


class RNADataset(Dataset):
    def __init__(self, sequences, loops, distances, targets=None):
        self.sequences = sequences
        self.loops = loops
        self.distances = distances
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        dist = torch.tensor(
            self.distances[idx], dtype=torch.float
        )  # Float for sinusoidal

        if self.targets is not None:
            # Targets are (68, 3). Pad to 107 for batching convenience, though loss only uses 68
            tgt = torch.tensor(self.targets[idx], dtype=torch.float)
            # Pad target to seq_len with zeros (we won't use them in loss)
            pad_len = Config.SEQ_LEN - Config.PRED_LEN
            tgt = F.pad(tgt, (0, 0, 0, pad_len))
            return seq, loop, dist, tgt
        else:
            return seq, loop, dist


# ==========================================
# MODEL
# ==========================================


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (Batch, Seq_Len) containing integer distances
        # Output: (Batch, Seq_Len, Dim)

        # Create frequency bands
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)

        # x is signed.
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)  # (B, L, half_dim)

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, L, dim)
        return emb


class OrthogonalBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)  # Pre-LN

        # Orthogonal Initialization
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        residual = x
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x + residual


class RNA_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Embeddings
        self.seq_emb = nn.Embedding(Config.SEQ_VOCAB_SIZE, Config.SEQ_EMBED_DIM)
        self.loop_emb = nn.Embedding(Config.LOOP_VOCAB_SIZE, Config.LOOP_EMBED_DIM)
        self.dist_emb = SinusoidalPositionalEmbedding(Config.DIST_EMBED_DIM)

        input_dim = Config.SEQ_EMBED_DIM + Config.LOOP_EMBED_DIM + Config.DIST_EMBED_DIM

        # Stem
        self.stem_lstm = nn.LSTM(
            input_dim, Config.HIDDEN_DIM // 2, batch_first=True, bidirectional=True
        )
        # No dropout after stem

        # Backbone
        self.blocks = nn.ModuleList(
            [
                OrthogonalBiLSTM(
                    Config.HIDDEN_DIM, Config.HIDDEN_DIM, dropout=Config.DROPOUT
                )
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # Aggregation
        self.mix_weights = nn.Parameter(torch.ones(Config.NUM_LAYERS + 1))

        # Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

        # Init stem
        for name, param in self.stem_lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, seq, loop, dist):
        # Embeddings
        s = self.seq_emb(seq)
        l = self.loop_emb(loop)
        d = self.dist_emb(dist)

        x = torch.cat([s, l, d], dim=-1)

        # Stem
        x, _ = self.stem_lstm(x)

        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Weighted Aggregation
        # Stack outputs: (Layers+1, B, L, H)
        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(self.mix_weights, dim=0).view(-1, 1, 1, 1)
        aggregated = (stacked * weights).sum(dim=0)

        # Head
        out = self.head(aggregated)
        return out


# ==========================================
# TRAINING & INFERENCE
# ==========================================


def mcrmse_loss(pred, target, mask=None):
    # Standard MSE, but we only care about first 68
    # pred: (B, 107, 3), target: (B, 107, 3)

    pred = pred[:, : Config.PRED_LEN, :]
    target = target[:, : Config.PRED_LEN, :]

    mse = F.mse_loss(pred, target)
    return mse


def validate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq, loop, dist, tgt in loader:
            seq, loop, dist, tgt = (
                seq.to(Config.DEVICE),
                loop.to(Config.DEVICE),
                dist.to(Config.DEVICE),
                tgt.to(Config.DEVICE),
            )
            pred = model(seq, loop, dist)

            # Slice to scored region
            pred = pred[:, : Config.PRED_LEN, :]
            tgt = tgt[:, : Config.PRED_LEN, :]

            all_preds.append(pred.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 68, 3)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE: Mean of RMSEs of columns
    rmses = []
    for i in range(3):
        col_mse = np.mean((all_preds[:, :, i] - all_targets[:, :, i]) ** 2)
        rmses.append(np.sqrt(col_mse))

    return np.mean(rmses)


def train_model():
    # Load Data
    train_ids, train_seq, train_loop, train_dist, train_tgt = load_and_cache_data(
        Config.TRAIN_PARQUET, "train_data"
    )
    val_ids, val_seq, val_loop, val_dist, val_tgt = load_and_cache_data(
        Config.VAL_PARQUET, "val_data"
    )

    train_ds = RNADataset(train_seq, train_loop, train_dist, train_tgt)
    val_ds = RNADataset(val_seq, val_loop, val_dist, val_tgt)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model Setup
    model = RNA_Model().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for seq, loop, dist, tgt in train_loader:
            seq, loop, dist, tgt = (
                seq.to(Config.DEVICE),
                loop.to(Config.DEVICE),
                dist.to(Config.DEVICE),
                tgt.to(Config.DEVICE),
            )

            optimizer.zero_grad()
            pred = model(seq, loop, dist)
            loss = mcrmse_loss(pred, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            train_loss_accum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_accum / len(train_loader)

        val_mcrmse = validate(model, val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MSE: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")
    return best_model_path


def generate_submission(model_path):
    print("Generating submission...")
    test_ids, test_seq, test_loop, test_dist = load_and_cache_data(
        Config.TEST_PARQUET, "test_data", is_test=True
    )
    test_ds = RNADataset(test_seq, test_loop, test_dist)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model = RNA_Model().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for seq, loop, dist in test_loader:
            seq, loop, dist = (
                seq.to(Config.DEVICE),
                loop.to(Config.DEVICE),
                dist.to(Config.DEVICE),
            )
            pred = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    submission_rows = []

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 3)

        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"
            reactivity = sample_preds[pos, 0]
            deg_Mg_pH10 = sample_preds[pos, 1]
            deg_Mg_50C = sample_preds[pos, 2]

            # Fill unscored/unpredicted columns with 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run_pipeline():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    best_model = train_model()
    generate_submission(best_model)
