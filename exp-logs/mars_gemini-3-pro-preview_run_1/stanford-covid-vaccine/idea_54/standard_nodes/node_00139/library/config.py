import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# =========================================================================================
# CONFIGURATION
# =========================================================================================


class Config:
    # Paths
    TRAIN_METADATA = "./metadata/train.parquet"
    VAL_METADATA = "./metadata/val.parquet"
    TEST_METADATA = "./metadata/test.parquet"
    CACHE_DIR = "./working/idea_54/"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Dimensions
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Model Architecture
    SEQ_EMBED_DIM = 100
    LOOP_EMBED_DIM = 64
    DIST_EMBED_DIM = 64
    HIDDEN_DIM = 384
    NUM_BLOCKS = 6
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    CLIP_NORM = 1.0
    EPOCHS = 20

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    ALL_PRED_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Seed
    SEED = 42


# Ensure reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# =========================================================================================
# DATA PROCESSING
# =========================================================================================


def get_structure_distance(structure):
    """
    Parses dot-bracket structure and returns a signed distance array.
    If base i is paired with j, dist = j - i.
    If unpaired, dist = 0.
    """
    stack = []
    indices = np.zeros(len(structure), dtype=np.int32)

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is upstream (smaller), i is downstream (larger)
                # For j: pair is i, dist = i - j (positive)
                # For i: pair is j, dist = j - i (negative)
                indices[j] = i - j
                indices[i] = j - i
    return indices


def process_data(df, is_test=False):
    # Dictionaries
    token2int = {x: i for i, x in enumerate("AGUC")}
    loop2int = {x: i for i, x in enumerate("BEHIMSX")}

    # Arrays
    ids = df["id"].values
    sequences = []
    loops = []
    distances = []

    for _, row in df.iterrows():
        # Sequence
        seq = [token2int.get(x, 0) for x in row["sequence"]]
        sequences.append(seq)

        # Loop
        lp = [loop2int.get(x, 0) for x in row["predicted_loop_type"]]
        loops.append(lp)

        # Structure Distance
        dist = get_structure_distance(row["structure"])
        distances.append(dist)

    sequences = np.array(sequences, dtype=np.int32)
    loops = np.array(loops, dtype=np.int32)
    distances = np.array(distances, dtype=np.int32)

    if is_test:
        return ids, sequences, loops, distances

    # Targets
    targets = []
    for col in Config.TARGET_COLS:
        # Each row in df[col] is a list/array of floats
        # Stack them
        val = np.vstack(df[col].values)
        targets.append(val)

    # Shape: (N, 3, 68) -> Transpose to (N, 68, 3)
    targets = np.stack(targets, axis=2)

    return ids, sequences, loops, distances, targets


def load_data(mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches it using .npy files.
    mode: 'train', 'val', or 'test'
    """
    cache_prefix = os.path.join(Config.CACHE_DIR, f"{mode}_data")
    files = {
        "ids": f"{cache_prefix}_ids.npy",
        "seq": f"{cache_prefix}_seq.npy",
        "loop": f"{cache_prefix}_loop.npy",
        "dist": f"{cache_prefix}_dist.npy",
        "tgt": f"{cache_prefix}_tgt.npy",
    }

    # Check cache
    if load_cached_data:
        all_exist = True
        required_keys = ["ids", "seq", "loop", "dist"]
        if mode != "test":
            required_keys.append("tgt")

        for k in required_keys:
            if not os.path.exists(files[k]):
                all_exist = False
                break

        if all_exist:
            print(f"Loading cached {mode} data...")
            ids = np.load(files["ids"], allow_pickle=True)
            seq = np.load(files["seq"])
            loop = np.load(files["loop"])
            dist = np.load(files["dist"])
            if mode == "test":
                return ids, seq, loop, dist
            else:
                tgt = np.load(files["tgt"])
                return ids, seq, loop, dist, tgt

    # Process from scratch
    print(f"Processing {mode} data from scratch...")
    if mode == "train":
        df = pd.read_parquet(Config.TRAIN_METADATA)
    elif mode == "val":
        df = pd.read_parquet(Config.VAL_METADATA)
    else:
        df = pd.read_parquet(Config.TEST_METADATA)

    if mode == "test":
        ids, seq, loop, dist = process_data(df, is_test=True)
        np.save(files["ids"], ids)
        np.save(files["seq"], seq)
        np.save(files["loop"], loop)
        np.save(files["dist"], dist)
        return ids, seq, loop, dist
    else:
        ids, seq, loop, dist, tgt = process_data(df, is_test=False)
        np.save(files["ids"], ids)
        np.save(files["seq"], seq)
        np.save(files["loop"], loop)
        np.save(files["dist"], dist)
        np.save(files["tgt"], tgt)
        return ids, seq, loop, dist, tgt


class RNADataset(Dataset):
    def __init__(self, sequences, loops, distances, targets=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.loops = torch.tensor(loops, dtype=torch.long)
        self.distances = torch.tensor(
            distances, dtype=torch.float
        )  # Float for sinusoidal
        self.targets = (
            torch.tensor(targets, dtype=torch.float) if targets is not None else None
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequence": self.sequences[idx],
            "loop": self.loops[idx],
            "distance": self.distances[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


# =========================================================================================
# MODEL
# =========================================================================================


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # We use a fixed scale for frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: (B, L) containing signed distances
        # We want to encode the value of distance
        # Create sinusoidal encoding for the scalar values in x

        # Handle sign: we process absolute value, but maybe sign matters?
        # The idea says "Signed Sinusoidal".
        # A simple way: encode abs(x) and concatenate with sign embedding or just use raw x in sin/cos?
        # sin(-x) = -sin(x), cos(-x) = cos(x). This preserves sign info in the sine component.

        sin_inp = x.unsqueeze(-1) * self.inv_freq  # (B, L, D/2)
        pos_emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)  # (B, L, D)
        return pos_emb


class ParallelBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        # Global Branch: BiGRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Local Branch: Depthwise Conv1d
        # Input (B, C, L), Output (B, C, L)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,  # Depthwise
            ),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, L, C)

        # Global Branch
        gru_out, _ = self.gru(x)  # (B, L, C)

        # Local Branch
        # Conv1d expects (B, C, L)
        conv_in = x.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        conv_out = conv_out.permute(0, 2, 1)  # (B, L, C)

        # Fusion
        out = gru_out + conv_out
        out = self.dropout(out)

        # Residual
        return self.norm(x + out)


class RNA_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_embed = nn.Embedding(4, config.SEQ_EMBED_DIM)
        self.loop_embed = nn.Embedding(7, config.LOOP_EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.DIST_EMBED_DIM)

        input_dim = config.SEQ_EMBED_DIM + config.LOOP_EMBED_DIM + config.DIST_EMBED_DIM

        # Stem
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Backbone
        self.blocks = nn.ModuleList(
            [
                ParallelBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_BLOCKS)
            ]
        )

        # Aggregation Weights
        # 1 (Stem) + N (Blocks)
        self.mix_weights = nn.Parameter(torch.zeros(config.NUM_BLOCKS + 1))

        # Head
        self.head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, sequence, loop, distance):
        # Embeddings
        emb_seq = self.seq_embed(sequence)
        emb_loop = self.loop_embed(loop)
        emb_dist = self.dist_embed(distance)

        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)

        # Stem
        x, _ = self.stem(x)  # (B, L, Hidden)

        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregation (Scalar Mixture)
        # Stack outputs: (K, B, L, C)
        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(self.mix_weights, dim=0).view(-1, 1, 1, 1)
        weighted_sum = (stacked * weights).sum(dim=0)

        # Head
        logits = self.head(weighted_sum)  # (B, L, 3)

        return logits


# =========================================================================================
# TRAINING UTILS
# =========================================================================================


def mcrmse_loss(y_true, y_pred):
    # y_true, y_pred: (B, 68, 3)
    # Calculate RMSE per column
    mse = torch.mean((y_true - y_pred) ** 2, dim=(0, 1))  # Mean over batch and seq
    rmse = torch.sqrt(mse)
    return torch.mean(rmse)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0

    for batch in loader:
        seq = batch["sequence"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["distance"].to(device)
        target = batch["target"].to(device)  # (B, 68, 3)

        optimizer.zero_grad()

        # Forward
        pred = model(seq, loop, dist)  # (B, 107, 3)

        # Slice prediction to scored region
        pred_scored = pred[:, : Config.SEQ_SCORED, :]

        # Loss
        loss = criterion(pred_scored, target)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_NORM)

        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * seq.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)
            target = batch["target"].to(device)

            pred = model(seq, loop, dist)
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE column-wise
    # (N, 68, 3)
    mse = np.mean((all_targets - all_preds) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    mcrmse = np.mean(rmse)

    return mcrmse


# =========================================================================================
# MAIN
# =========================================================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_ids, train_seq, train_loop, train_dist, train_tgt = load_data("train")
    val_ids, val_seq, val_loop, val_dist, val_tgt = load_data("val")

    train_dataset = RNADataset(train_seq, train_loop, train_dist, train_tgt)
    val_dataset = RNADataset(val_seq, val_loop, val_dist, val_tgt)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Model
    model = RNA_Model(Config).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing over total steps
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss
    # We use MSE for optimization as per Idea
    criterion = nn.MSELoss()

    # Training Loop
    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_mcrmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")

    # =========================================================================================
    # INFERENCE
    # =========================================================================================
    print("Generating submission...")

    # Load Test Data
    test_ids, test_seq, test_loop, test_dist = load_data("test")
    test_dataset = RNADataset(test_seq, test_loop, test_dist, None)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Best Model
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CACHE_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    preds_dict = {}  # Key: id, Value: (107, 5) array

    # We need to output for all positions (seq_length=107) and all 5 columns.
    # The model predicts 3 columns for 107 positions.
    # The other 2 columns (deg_pH10, deg_50C) should be 0.

    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)

            pred = model(seq, loop, dist)  # (B, 107, 3)
            pred = pred.cpu().numpy()

            # Map back to IDs
            current_batch_size = seq.size(0)
            batch_ids = test_ids[
                batch_idx * Config.BATCH_SIZE : (batch_idx + 1) * Config.BATCH_SIZE
            ]

            for i, sample_id in enumerate(batch_ids):
                # Prepare 5-column output
                # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                # Model output: reactivity, deg_Mg_pH10, deg_Mg_50C

                full_pred = np.zeros((Config.SEQ_LENGTH, 5), dtype=np.float32)

                # Fill predicted columns
                # 0 -> reactivity (0)
                # 1 -> deg_Mg_pH10 (1)
                # 2 -> deg_Mg_50C (3)

                p = pred[i]  # (107, 3)
                full_pred[:, 0] = p[:, 0]
                full_pred[:, 1] = p[:, 1]
                full_pred[:, 3] = p[:, 2]

                preds_dict[sample_id] = full_pred

            batch_idx += 1

    # Create Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []

    for sample_id in test_ids:
        preds = preds_dict[sample_id]
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = preds[seqpos]
            submission_data.append([row_id] + list(row_vals))

    sub_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + Config.ALL_PRED_COLS)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
