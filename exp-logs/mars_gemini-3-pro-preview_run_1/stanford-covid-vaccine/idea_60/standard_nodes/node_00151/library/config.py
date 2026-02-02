import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ==================================================================================
# CONFIGURATION
# ==================================================================================
class Config:
    # Data
    TRAIN_PATH = "./metadata/train.parquet"
    VAL_PATH = "./metadata/val.parquet"
    TEST_PATH = "./metadata/test.parquet"
    CACHE_DIR = "./working/idea_60/"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Model Architecture
    SEQ_LEN = 107
    PRED_LEN = 68
    EMBED_SEQ_DIM = 128
    EMBED_LOOP_DIM = 64
    EMBED_DIST_DIM = 64
    TOTAL_INPUT_DIM = 256  # 128 + 64 + 64
    HIDDEN_DIM = 384
    NUM_LAYERS = 6
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================================
# UTILS & DATA PROCESSING
# ==================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string and returns a mapping of paired indices.
    Returns a dictionary {index: paired_index}. Unpaired indices are not in the dict.
    """
    pairs = {}
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def process_data(df, mode="train"):
    """
    Processes DataFrame into numpy arrays for the model.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}

    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    # Pre-allocate
    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    X_seq = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_loop = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_dist = np.zeros((n_samples, seq_len), dtype=np.float32)  # Signed distance

    for i in range(n_samples):
        # Sequence
        X_seq[i] = [seq_map.get(c, 0) for c in sequences[i]]

        # Loop
        X_loop[i] = [loop_map.get(c, 0) for c in loops[i]]

        # Distance
        pairs = get_structure_pairs(structures[i])
        for j in range(seq_len):
            if j in pairs:
                # Signed distance: current_index - paired_index
                X_dist[i, j] = j - pairs[j]
            else:
                X_dist[i, j] = 0.0  # Unpaired

    if mode in ["train", "val"]:
        # Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # These are stored as lists/arrays in the parquet file
        # We need to stack them.
        # Note: The parquet lists might be of length 68. We pad to 107 or keep as is.
        # The prompt says "Predict targets for *each* sequence position".
        # But ground truth is only 68. We train on 68.
        # We will store targets as (N, 68, 3).

        y_reactivity = np.vstack(df["reactivity"].values)
        y_deg_Mg_pH10 = np.vstack(df["deg_Mg_pH10"].values)
        y_deg_Mg_50C = np.vstack(df["deg_Mg_50C"].values)

        # Stack channels: (N, 68, 3)
        y = np.stack([y_reactivity, y_deg_Mg_pH10, y_deg_Mg_50C], axis=2)
        return ids, X_seq, X_loop, X_dist, y
    else:
        return ids, X_seq, X_loop, X_dist


def get_dataset(path, mode="train", load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        if mode in ["train", "val"]:
            return data["ids"], data["X_seq"], data["X_loop"], data["X_dist"], data["y"]
        else:
            return data["ids"], data["X_seq"], data["X_loop"], data["X_dist"]

    print(f"Processing {mode} data from {path}...")
    df = pd.read_parquet(path)
    processed = process_data(df, mode)

    if mode in ["train", "val"]:
        ids, X_seq, X_loop, X_dist, y = processed
        np.savez(cache_file, ids=ids, X_seq=X_seq, X_loop=X_loop, X_dist=X_dist, y=y)
        return ids, X_seq, X_loop, X_dist, y
    else:
        ids, X_seq, X_loop, X_dist = processed
        np.savez(cache_file, ids=ids, X_seq=X_seq, X_loop=X_loop, X_dist=X_dist)
        return ids, X_seq, X_loop, X_dist


class RNADataset(Dataset):
    def __init__(self, X_seq, X_loop, X_dist, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.long)
        self.X_loop = torch.tensor(X_loop, dtype=torch.long)
        self.X_dist = torch.tensor(X_dist, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_seq[idx], self.X_loop[idx], self.X_dist[idx], self.y[idx]
        return self.X_seq[idx], self.X_loop[idx], self.X_dist[idx]


# ==================================================================================
# MODEL
# ==================================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create a sufficiently large buffer for relative distances
        # Range is approx -107 to 107. Let's cover -200 to 200.
        self.register_buffer(
            "div_term",
            1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)),
        )

    def forward(self, x):
        # x is (Batch, Seq_Len) containing signed float distances
        # We compute sin/cos encoding on the fly

        # x: (B, L) -> (B, L, 1)
        x_expanded = x.unsqueeze(-1)

        # div_term: (D/2)
        # phase: (B, L, D/2)
        phase = x_expanded * self.div_term

        # sin/cos
        sin_enc = torch.sin(phase)
        cos_enc = torch.cos(phase)

        # Concatenate: (B, L, D)
        # Interleave sin and cos
        enc = torch.cat([sin_enc, cos_enc], dim=-1)
        return enc


class VectorScaledResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Vector scaling parameter initialized to 1.0
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        # Pre-LayerNorm
        residual = x
        out = self.ln(x)
        out, _ = self.gru(out)
        out = self.dropout(out)

        # Vector Scaling: Element-wise multiplication
        # scale: (D,) -> (1, 1, D)
        out = out * self.scale.view(1, 1, -1)

        return residual + out


class RNA_Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Embeddings
        self.seq_emb = nn.Embedding(4, config.EMBED_SEQ_DIM)
        self.loop_emb = nn.Embedding(7, config.EMBED_LOOP_DIM)
        self.dist_enc = SinusoidalPositionalEncoding(config.EMBED_DIST_DIM)

        # Stem
        self.stem_gru = nn.GRU(
            config.TOTAL_INPUT_DIM,
            config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Backbone
        self.blocks = nn.ModuleList(
            [
                VectorScaledResidualBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.NUM_LAYERS)
            ]
        )

        # Scalar Mixture Aggregation
        # Weights for Stem + 6 Blocks = 7 outputs
        self.mix_weights = nn.Parameter(torch.zeros(config.NUM_LAYERS + 1))

        # Head
        self.head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, x_seq, x_loop, x_dist):
        # Embeddings
        e_seq = self.seq_emb(x_seq)
        e_loop = self.loop_emb(x_loop)
        e_dist = self.dist_enc(x_dist)

        # Fusion
        x = torch.cat([e_seq, e_loop, e_dist], dim=-1)

        # Stem
        x, _ = self.stem_gru(x)
        # No dropout after stem as per instructions

        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Scalar Mixture Aggregation
        # Stack outputs: (B, L, D, Num_Layers+1)
        stacked = torch.stack(outputs, dim=-1)

        # Softmax weights for stability, or raw?
        # "Scalar Mixture" usually implies weighted sum.
        # Instructions say "learnable weighted sum".
        weights = F.softmax(self.mix_weights, dim=0)

        # Weighted sum
        # weights: (Num_Layers+1) -> (1, 1, 1, Num_Layers+1)
        aggregated = torch.sum(stacked * weights.view(1, 1, 1, -1), dim=-1)

        # Head
        logits = self.head(aggregated)

        return logits


# ==================================================================================
# TRAINING ENGINE
# ==================================================================================
def mcrmse_loss(y_true, y_pred):
    """
    Calculate MCRMSE.
    y_true, y_pred: (N, 3) flattened or (N, L, 3)
    """
    col_losses = torch.mean((y_true - y_pred) ** 2, dim=0)
    return torch.mean(torch.sqrt(col_losses))


def train_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0

    criterion = nn.MSELoss()

    for x_seq, x_loop, x_dist, y in loader:
        x_seq, x_loop, x_dist = x_seq.to(device), x_loop.to(device), x_dist.to(device)
        y = y.to(device)  # (B, 68, 3)

        optimizer.zero_grad()

        # Forward
        preds = model(x_seq, x_loop, x_dist)  # (B, 107, 3)

        # Masked Loss: Only first 68 positions
        preds_scored = preds[:, : config.PRED_LEN, :]

        loss = criterion(preds_scored, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_seq, x_loop, x_dist, y in loader:
            x_seq, x_loop, x_dist = (
                x_seq.to(device),
                x_loop.to(device),
                x_dist.to(device),
            )

            preds = model(x_seq, x_loop, x_dist)
            preds_scored = preds[:, : config.PRED_LEN, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)  # (N, 68, 3)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 68, 3)

    # Calculate MCRMSE
    # Average RMSE of each column
    mse_per_col = torch.mean(
        (all_targets - all_preds) ** 2, dim=(0, 1)
    )  # This averages over samples and length
    # Wait, MCRMSE definition: mean of sqrt of MSE per column.
    # The metric formula: 1/Nt * sum_j( sqrt( 1/n * sum_i (y_ij - y_hat_ij)^2 ) )
    # Here i is sample index (flattening batch and seq_len for that column).

    # Flatten batch and seq dimensions
    flat_preds = all_preds.reshape(-1, 3)
    flat_targets = all_targets.reshape(-1, 3)

    mse_cols = torch.mean((flat_targets - flat_preds) ** 2, dim=0)
    rmse_cols = torch.sqrt(mse_cols)
    mcrmse = torch.mean(rmse_cols)

    return mcrmse.item()


def run_training():
    set_seed(Config.SEED)

    # Load Data
    train_ids, train_seq, train_loop, train_dist, train_y = get_dataset(
        Config.TRAIN_PATH, "train"
    )
    val_ids, val_seq, val_loop, val_dist, val_y = get_dataset(Config.VAL_PATH, "val")

    train_ds = RNADataset(train_seq, train_loop, train_dist, train_y)
    val_ds = RNADataset(val_seq, val_loop, val_dist, val_y)

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
    model = RNA_Model(Config).to(Config.DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE, Config)
        val_mcrmse = validate(model, val_loader, Config.DEVICE, Config)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.6f}")
    return best_model_path


def run_inference(model_path):
    # Load Test Data
    test_ids, test_seq, test_loop, test_dist = get_dataset(Config.TEST_PATH, "test")
    test_ds = RNADataset(test_seq, test_loop, test_dist)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = RNA_Model(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x_seq, x_loop, x_dist in test_loader:
            x_seq, x_loop, x_dist = (
                x_seq.to(Config.DEVICE),
                x_loop.to(Config.DEVICE),
                x_dist.to(Config.DEVICE),
            )

            # Predict full length (107)
            preds = model(x_seq, x_loop, x_dist)  # (B, 107, 3)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 3)

    # Prepare Submission
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # We predicted: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)
    # We need to insert 0s for deg_pH10 and deg_50C

    submission_data = []

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 3)

        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            reactivity = sample_preds[j, 0]
            deg_Mg_pH10 = sample_preds[j, 1]
            deg_pH10 = 0.0
            deg_Mg_50C = sample_preds[j, 2]
            deg_50C = 0.0

            submission_data.append(
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
    sub_df = pd.DataFrame(submission_data, columns=columns)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    best_model = run_training()
    run_inference(best_model)


# Execute if run directly
if __name__ == "__main__":
    main()
