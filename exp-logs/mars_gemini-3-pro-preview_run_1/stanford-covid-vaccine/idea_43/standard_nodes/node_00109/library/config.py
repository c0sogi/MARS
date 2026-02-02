import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random

# ==========================================
# Configuration
# ==========================================


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_44"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data
    SEQ_LEN = 107
    SCORED_LEN = 68
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    ALL_SUBMISSION_COLS = [
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]

    # Model Architecture
    # Cite solution_lesson_node_00099: Proportional Feature Embedding (Seq 100, others 64)
    EMBED_DIM_SEQ = 100
    EMBED_DIM_LOOP = 64
    EMBED_DIM_DIST = 64
    TOTAL_INPUT_DIM = 228  # 100 + 64 + 64
    # Cite solution_lesson_node_00108, solution_lesson_node_00081: Hidden Dim 384 for BiGRU
    HIDDEN_DIM = 384  # Residual stream width
    NUM_LAYERS = 6
    # Cite solution_lesson_node_00102: Dropout 0.1
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD_NORM = 1.0
    SEED = 42


# ==========================================
# Utils & Data Processing
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_sinusoidal_encoding(num_positions, d_model):
    """
    Generates fixed sinusoidal encodings.
    """
    pe = torch.zeros(num_positions, d_model)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def parse_structure_distances(structure):
    """
    Parses dot-bracket structure to get signed pairing distances.
    Unpaired bases get distance 0.
    Paired bases (i, j) get distance j - i for index i.
    """
    n = len(structure)
    distances = np.zeros(n, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                dist = j - i
                distances[i] = dist
                distances[j] = -dist
    return distances


def process_data(df, mode="train"):
    """
    Processes DataFrame into numpy arrays.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = df["id"].values

    # Sequences
    sequences = df["sequence"].values
    seq_encoded = np.array([[seq_map[c] for c in s] for s in sequences], dtype=np.int32)

    # Loop Types
    loops = df["predicted_loop_type"].values
    loop_encoded = np.array([[loop_map[c] for c in l] for l in loops], dtype=np.int32)

    # Structure Distances
    structures = df["structure"].values
    dist_encoded = np.array(
        [parse_structure_distances(s) for s in structures], dtype=np.int32
    )

    if mode in ["train", "val"]:
        # Targets
        targets = []
        for col in Config.TARGET_COLS:
            # Each row in df[col] is a list/array of length 68
            # Stack them into (N, 68)
            col_data = np.vstack(df[col].values)
            targets.append(col_data)
        targets = np.stack(targets, axis=2)  # (N, 68, 3)
        return ids, seq_encoded, loop_encoded, dist_encoded, targets
    else:
        return ids, seq_encoded, loop_encoded, dist_encoded


def get_dataset(mode="train", load_cached_data=True):
    """
    Loads data from metadata parquet files, with caching mechanism.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_{mode}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache...")
        data = np.load(cache_path, allow_pickle=True)
        if mode in ["train", "val"]:
            return data["ids"], data["seq"], data["loop"], data["dist"], data["targets"]
        else:
            return data["ids"], data["seq"], data["loop"], data["dist"]

    print(f"Processing {mode} data from scratch...")
    parquet_path = os.path.join(Config.METADATA_DIR, f"{mode}.parquet")
    df = pd.read_parquet(parquet_path)

    processed = process_data(df, mode)

    if mode in ["train", "val"]:
        ids, seq, loop, dist, targets = processed
        np.savez(cache_path, ids=ids, seq=seq, loop=loop, dist=dist, targets=targets)
        return ids, seq, loop, dist, targets
    else:
        ids, seq, loop, dist = processed
        np.savez(cache_path, ids=ids, seq=seq, loop=loop, dist=dist)
        return ids, seq, loop, dist


class RNADataset(Dataset):
    def __init__(self, ids, seq, loop, dist, targets=None):
        self.ids = ids
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.loop = torch.tensor(loop, dtype=torch.long)
        self.dist = torch.tensor(dist, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {"seq": self.seq[idx], "loop": self.loop[idx], "dist": self.dist[idx]}
        if self.targets is not None:
            sample["targets"] = self.targets[idx]
        return sample


# ==========================================
# Model Architecture
# ==========================================


class StabilizedWideResBiLSTM(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.seq_embed = nn.Embedding(4, config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(7, config.EMBED_DIM_LOOP)

        # Distance Embedding (Fixed Sinusoidal)
        # Distances range approx -107 to 107. We map them to indices.
        # Offset by SEQ_LEN to make indices positive. 0 -> SEQ_LEN.
        self.dist_offset = config.SEQ_LEN
        num_dist_tokens = 2 * config.SEQ_LEN + 1
        # Create fixed sinusoidal matrix
        pe = get_sinusoidal_encoding(num_dist_tokens, config.EMBED_DIM_DIST)
        self.dist_embed = nn.Embedding.from_pretrained(pe, freeze=True)

        # Stem
        self.stem = nn.LSTM(
            input_size=config.TOTAL_INPUT_DIM,
            hidden_size=config.HIDDEN_DIM // 2,  # Bidirectional -> HIDDEN_DIM
            batch_first=True,
            bidirectional=True,
        )

        # Residual Blocks
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(config.HIDDEN_DIM),
                        "lstm": nn.LSTM(
                            input_size=config.HIDDEN_DIM,
                            hidden_size=config.HIDDEN_DIM // 2,
                            batch_first=True,
                            bidirectional=True,
                        ),
                        "dropout": nn.Dropout(config.DROPOUT),
                    }
                )
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # Scalar Mixture Aggregation
        # Weights for Stem + 6 Layers
        self.mix_weights = nn.Parameter(torch.zeros(config.NUM_LAYERS + 1))

        # Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, len(config.TARGET_COLS))

    def forward(self, seq, loop, dist):
        # Embeddings
        x_seq = self.seq_embed(seq)
        x_loop = self.loop_embed(loop)

        # Distance embedding: offset indices
        dist_idx = torch.clamp(dist + self.dist_offset, 0, 2 * self.config.SEQ_LEN)
        x_dist = self.dist_embed(dist_idx)

        # Concatenate
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)  # (B, L, 256)

        # Stem
        x, _ = self.stem(x)  # (B, L, 512)

        layer_outputs = [x]

        # Residual Blocks
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x, _ = layer["lstm"](x)
            x = layer["dropout"](x)
            x = residual + x
            layer_outputs.append(x)

        # Aggregation
        # Stack outputs: (B, L, 512, Num_Layers+1)
        stacked = torch.stack(layer_outputs, dim=-1)
        weights = F.softmax(self.mix_weights, dim=0)
        aggregated = torch.sum(stacked * weights, dim=-1)  # (B, L, 512)

        # Head
        out = self.head(aggregated)  # (B, L, 3)

        return out


# ==========================================
# Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["targets"].to(device)  # (B, 68, 3)

        optimizer.zero_grad()

        preds = model(seq, loop, dist)  # (B, 107, 3)

        # Slice predictions to scored length
        preds_scored = preds[:, : Config.SCORED_LEN, :]

        loss = criterion(preds_scored, targets)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, dist)
            preds_scored = preds[:, : Config.SCORED_LEN, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # MCRMSE: Average of column-wise RMSEs
    mse = (all_preds - all_targets) ** 2
    rmse_per_col = torch.sqrt(mse.mean(dim=0))  # Mean over samples -> (68, 3)
    # Average over columns and positions?
    # The metric definition: Mean columnwise root mean squared error.
    # Usually: Average(RMSE(col1), RMSE(col2), RMSE(col3))
    # Where RMSE(col) is scalar over all samples and positions for that col.

    # Flatten positions and samples for each target column
    # (N, 68, 3) -> (N*68, 3)
    flat_preds = all_preds.reshape(-1, 3)
    flat_targets = all_targets.reshape(-1, 3)

    col_mse = ((flat_preds - flat_targets) ** 2).mean(dim=0)
    col_rmse = torch.sqrt(col_mse)
    mcrmse = col_rmse.mean().item()

    return mcrmse


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)

            preds = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def run_pipeline():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_ids, train_seq, train_loop, train_dist, train_targets = get_dataset("train")
    val_ids, val_seq, val_loop, val_dist, val_targets = get_dataset("val")
    test_ids, test_seq, test_loop, test_dist = get_dataset("test")

    # Datasets & Loaders
    train_ds = RNADataset(train_ids, train_seq, train_loop, train_dist, train_targets)
    val_ds = RNADataset(val_ids, val_seq, val_loop, val_dist, val_targets)
    test_ds = RNADataset(test_ids, test_seq, test_loop, test_dist)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Model Setup
    model = StabilizedWideResBiLSTM(Config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Training Loop
    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mcrmse = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            # Save best model state if needed, but we just need final submission for this task logic
            # torch.save(model.state_dict(), os.path.join(Config.WORKING_DIR, 'best_model.pth'))

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")

    # Inference
    print("Generating predictions...")
    test_preds = predict(model, test_loader, device)  # (N_test, 107, 3)

    # Prepare Submission
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # We predicted: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)
    # Missing: deg_pH10, deg_50C (fill with 0)

    submission_data = []

    # Map prediction indices to submission columns
    # Preds: [reactivity, deg_Mg_pH10, deg_Mg_50C]
    # Sub Cols: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]

    for i, sample_id in enumerate(test_ids):
        sample_preds = test_preds[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            p_reactivity = sample_preds[seqpos, 0]
            p_deg_Mg_pH10 = sample_preds[seqpos, 1]
            p_deg_Mg_50C = sample_preds[seqpos, 2]

            # Unscored/Unpredicted columns
            p_deg_pH10 = 0.0
            p_deg_50C = 0.0

            submission_data.append(
                [
                    row_id,
                    p_reactivity,
                    p_deg_Mg_pH10,
                    p_deg_pH10,
                    p_deg_Mg_50C,
                    p_deg_50C,
                ]
            )

    submission_df = pd.DataFrame(
        submission_data, columns=["id_seqpos"] + Config.ALL_SUBMISSION_COLS
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


# Run the pipeline
run_pipeline()
