import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Model Architecture
    HIDDEN_DIM = 512
    NUM_LAYERS = 6
    EMBED_DIM_SEQ = 128
    EMBED_DIM_LOOP = 64
    EMBED_DIM_DIST = 64
    DROPOUT = 0.2

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    EPOCHS = 20

    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    SUBMISSION_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Directories
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"
    WORKING_DIR = "./working/idea_59"
    SUBMISSION_PATH = "./submission/submission.csv"


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_distance(structure):
    """
    Computes signed distance for paired bases.
    Unpaired bases have distance 0.
    Paired (i, j): dist[i] = j - i, dist[j] = i - j
    """
    n = len(structure)
    dists = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                dists[j] = float(i - j)
                dists[i] = float(j - i)
    return dists


class RNADataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df
        self.mode = mode
        self.seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
        self.loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Sequence
        seq = [self.seq_map.get(c, 0) for c in row["sequence"]]
        seq = np.array(seq, dtype=np.int64)

        # Loop Type
        loop = [self.loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loop = np.array(loop, dtype=np.int64)

        # Distance
        dist = get_structure_distance(row["structure"])

        if self.mode in ["train", "val"]:
            targets = []
            for col in Config.TARGET_COLS:
                val = np.array(row[col], dtype=np.float32)
                # Ensure length consistency
                if len(val) < Config.SEQ_LEN:
                    pad = np.zeros(Config.SEQ_LEN - len(val), dtype=np.float32)
                    val = np.concatenate([val, pad])
                targets.append(val)
            targets = np.stack(targets, axis=1)  # (SEQ_LEN, 3)

            # Mask for scored positions (first 68)
            mask = np.zeros(Config.SEQ_LEN, dtype=np.float32)
            mask[: Config.PRED_LEN] = 1.0

            return {
                "seq": torch.from_numpy(seq),
                "loop": torch.from_numpy(loop),
                "dist": torch.from_numpy(dist),
                "targets": torch.from_numpy(targets),
                "mask": torch.from_numpy(mask),
            }
        else:
            return {
                "seq": torch.from_numpy(seq),
                "loop": torch.from_numpy(loop),
                "dist": torch.from_numpy(dist),
                "id": row["id"],
            }


def load_data(load_cached_data=True, debug=False):
    """
    Loads data from metadata parquet files.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # In this pipeline, parquet IS the cache mechanism.
    # We load directly from metadata.

    train_df = pd.read_parquet(os.path.join(Config.METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(Config.METADATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(Config.METADATA_DIR, "test.parquet"))

    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:20]
        test_df = test_df.iloc[:20]

    return train_df, val_df, test_df


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, L) signed distances
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)

        x_ex = x.unsqueeze(-1)  # (B, L, 1)
        emb_ex = emb.view(1, 1, -1)  # (1, 1, half_dim)

        args = x_ex * emb_ex
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class WideStreamBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Vector Scaling: Initialize to Identity (ones)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        residual = x
        out = self.ln(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        # Channel-wise scaling
        out = out * self.scale
        return residual + out


class VectorScaledBiGRU(nn.Module):
    def __init__(self):
        super().__init__()

        # Embeddings
        self.seq_emb = nn.Embedding(4, Config.EMBED_DIM_SEQ)
        self.loop_emb = nn.Embedding(7, Config.EMBED_DIM_LOOP)
        self.dist_emb = SinusoidalPositionalEmbedding(Config.EMBED_DIM_DIST)

        input_dim = Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST

        # Stem: Project to 512
        self.stem_gru = nn.GRU(
            input_dim, Config.HIDDEN_DIM // 2, batch_first=True, bidirectional=True
        )

        # Backbone: 6 Wide Blocks
        self.blocks = nn.ModuleList(
            [
                WideStreamBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # Scalar Mixture Aggregation
        self.mix_weights = nn.Parameter(torch.zeros(Config.NUM_LAYERS + 1))

        # Head
        self.head = nn.Linear(Config.HIDDEN_DIM, len(Config.TARGET_COLS))

    def forward(self, seq, loop, dist):
        # Embed
        x_seq = self.seq_emb(seq)
        x_loop = self.loop_emb(loop)
        x_dist = self.dist_emb(dist)

        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)

        # Stem
        x, _ = self.stem_gru(x)
        outputs = [x]

        # Blocks
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregation
        stacked = torch.stack(outputs, dim=-1)  # (B, L, H, Layers)
        weights = F.softmax(self.mix_weights, dim=0)
        x_agg = torch.sum(stacked * weights, dim=-1)

        # Head
        logits = self.head(x_agg)
        return logits


# ==================================================================================
# TRAINING & INFERENCE
# ==================================================================================


def train_model(epochs=Config.EPOCHS, debug=False, load_cached_data=True):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Data
    train_df, val_df, test_df = load_data(load_cached_data, debug)

    train_loader = DataLoader(
        RNADataset(train_df, "train"),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        RNADataset(val_df, "val"),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    # Model
    model = VectorScaledBiGRU().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(seq, loop, dist)

            # MSE Loss on masked region
            mse = (preds - targets) ** 2
            loss = (mse * mask.unsqueeze(-1)).sum() / mask.unsqueeze(-1).sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seq, loop, dist)

                # Extract scored positions
                preds = preds[:, : Config.PRED_LEN, :]
                targets = targets[:, : Config.PRED_LEN, :]

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # MCRMSE: Mean of column-wise RMSEs
        rmses = []
        for i in range(3):
            col_mse = np.mean((all_preds[:, :, i] - all_targets[:, :, i]) ** 2)
            rmses.append(np.sqrt(col_mse))
        val_mcrmse = np.mean(rmses)

        scheduler.step()

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best MCRMSE: {best_mcrmse}")

    # Generate Submission
    generate_submission(best_model_path, test_df, device)


def generate_submission(model_path, test_df, device):
    print("Generating submission...")
    model = VectorScaledBiGRU().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(
        RNADataset(test_df, "test"),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)
            preds = preds.cpu().numpy()  # (B, 107, 3)

            for i, sample_id in enumerate(ids):
                # We need to output for all 107 positions
                for pos in range(Config.SEQ_LEN):
                    row_id = f"{sample_id}_{pos}"
                    p_react = preds[i, pos, 0]
                    p_mg_ph10 = preds[i, pos, 1]
                    p_mg_50c = preds[i, pos, 2]

                    submission_data.append(
                        {
                            "id_seqpos": row_id,
                            "reactivity": p_react,
                            "deg_Mg_pH10": p_mg_ph10,
                            "deg_pH10": 0.0,
                            "deg_Mg_50C": p_mg_50c,
                            "deg_50C": 0.0,
                        }
                    )

    sub_df = pd.DataFrame(submission_data)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
