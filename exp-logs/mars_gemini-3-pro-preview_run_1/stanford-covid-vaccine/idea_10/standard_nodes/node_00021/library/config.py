import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Data Dimensions
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Targets
    # Full list: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Scored: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_INDICES = [0, 1, 3]

    # Vocabularies
    # Sequence: A:0, G:1, C:2, U:3
    TOKEN2ID = {"A": 0, "G": 1, "C": 2, "U": 3}
    VOCAB_SIZE = 4

    # Loop types: S, M, I, B, H, E, X
    LOOP2ID = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    LOOP_VOCAB_SIZE = 7

    # Model Architecture
    # Scaled up based on Lesson 17
    EMBED_DIM = 192
    HIDDEN_DIM = 384
    NUM_LAYERS = 5
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 20
    SEED = 42

    # Paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_PATH = "./submission/submission.csv"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def parse_structure(structure_str, seq_len):
    """
    Parses dot-bracket structure to find pairs and calculate distances.
    Returns:
        pair_indices: (seq_len,) array where val is index of paired base or -1
        distances: (seq_len,) array where val is j-i (if paired) or 0
    """
    stack = []
    pair_indices = np.full(seq_len, -1, dtype=np.int32)
    distances = np.zeros(seq_len, dtype=np.float32)

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                # Register pair for both positions
                pair_indices[start] = i
                pair_indices[i] = start

                # Calculate distance
                dist = i - start
                distances[start] = dist
                distances[i] = -dist

    return pair_indices, distances


class RNADataset(Dataset):
    def __init__(self, split="train", config=Config(), load_cached_data=True):
        self.config = config
        self.split = split
        self.rng = np.random.default_rng(config.SEED)

        os.makedirs(config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(config.WORKING_DIR, f"{split}_data.npz")

        # 1. Load from Cache if available
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            self.ids = data["ids"]
            self.sequences = data["sequences"]
            self.loop_types = data["loop_types"]
            self.pair_indices = data["pair_indices"]
            self.pair_distances = data["pair_distances"]
            if split in ["train", "val"]:
                self.targets = data["targets"]
            else:
                self.targets = None

        # 2. Process from Scratch
        else:
            print(f"Processing {split} data from metadata...")
            parquet_path = os.path.join(config.METADATA_DIR, f"{split}.parquet")
            df = pd.read_parquet(parquet_path)

            self.ids = df["id"].values

            # Encode Sequences
            self.sequences = []
            for seq in df["sequence"]:
                encoded = [config.TOKEN2ID.get(c, 0) for c in seq]
                self.sequences.append(encoded)
            self.sequences = np.array(self.sequences, dtype=np.int32)

            # Encode Loop Types
            self.loop_types = []
            for lt in df["predicted_loop_type"]:
                encoded = [config.LOOP2ID.get(c, 0) for c in lt]
                self.loop_types.append(encoded)
            self.loop_types = np.array(self.loop_types, dtype=np.int32)

            # Parse Structures
            self.pair_indices = []
            self.pair_distances = []
            for struct in df["structure"]:
                pidx, pdist = parse_structure(struct, config.SEQ_LEN)
                self.pair_indices.append(pidx)
                self.pair_distances.append(pdist)
            self.pair_indices = np.array(self.pair_indices, dtype=np.int32)
            self.pair_distances = np.array(self.pair_distances, dtype=np.float32)

            # Process Targets (Train/Val only)
            if split in ["train", "val"]:
                t_list = []
                for col in config.TARGET_COLS:
                    # df[col] contains lists/arrays. Stack them.
                    col_data = np.vstack(df[col].values)
                    t_list.append(col_data)

                # Stack to shape (N, 68, 5)
                self.targets = np.stack(t_list, axis=2)
            else:
                self.targets = None

            # Save to Cache
            save_dict = {
                "ids": self.ids,
                "sequences": self.sequences,
                "loop_types": self.loop_types,
                "pair_indices": self.pair_indices,
                "pair_distances": self.pair_distances,
            }
            if self.targets is not None:
                save_dict["targets"] = self.targets

            np.savez(cache_path, **save_dict)
            print(f"Data cached to {cache_path}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        loop = self.loop_types[idx]
        pair_idx = self.pair_indices[idx]
        pair_dist = self.pair_distances[idx]

        # Dynamic Masking for Reconstruction Task
        mask_labels = np.full_like(seq, -100)  # -100 is ignored by CrossEntropyLoss

        if self.split == "train":
            prob = self.rng.random(len(seq))
            mask_bool = prob < self.config.MASK_PROB

            # Store ground truth for masked tokens
            mask_labels[mask_bool] = seq[mask_bool]

            # Apply mask to input
            seq[mask_bool] = self.config.MASK_TOKEN_ID

        item = {
            "seq": torch.tensor(seq, dtype=torch.long),
            "loop": torch.tensor(loop, dtype=torch.long),
            "pair_idx": torch.tensor(pair_idx, dtype=torch.long),
            "pair_dist": torch.tensor(pair_dist, dtype=torch.float),
            "mask_labels": torch.tensor(mask_labels, dtype=torch.long),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class SinusoidalPositionalEmbedding(nn.Module):
    """Encodes scalar distances using sinusoidal functions."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(-1)
        pe = torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device)
        sin_inp = x * self.div_term
        pe[..., 0::2] = torch.sin(sin_inp)
        pe[..., 1::2] = torch.cos(sin_inp)
        return pe


class RNAModel(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config

        # Feature Embeddings
        self.seq_embed = nn.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(config.LOOP_VOCAB_SIZE, config.EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.EMBED_DIM)

        # Projection for concatenated features
        # Input: Seq + Loop + Dist + PairedSeq
        input_dim = config.EMBED_DIM * 4
        self.input_proj = nn.Linear(input_dim, config.HIDDEN_DIM)
        self.input_norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

        # Backbone: Residual BiGRU
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(config.NUM_LAYERS):
            self.layers.append(
                nn.GRU(
                    config.HIDDEN_DIM,
                    config.HIDDEN_DIM // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(config.HIDDEN_DIM))

        # Heads
        # 1. Regression Head (3 scored targets)
        self.reg_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 3),
        )

        # 2. Reconstruction Head (4 bases)
        self.recon_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM // 2, 4),
        )

    def forward(self, seq, loop, pair_idx, pair_dist):
        # 1. Embeddings
        x_seq = self.seq_embed(seq)  # (B, L, D)
        x_loop = self.loop_embed(loop)  # (B, L, D)
        x_dist = self.dist_embed(pair_dist)  # (B, L, D)

        # 2. Paired-Base Identity Feature
        # Gather embedding of the paired base.
        # Handle unpaired (-1) by mapping to 0 temporarily and masking later.
        safe_pair_idx = pair_idx.clone()
        unpaired_mask = safe_pair_idx == -1
        safe_pair_idx[unpaired_mask] = 0

        # Expand indices for gather: (B, L, D)
        idx_expanded = safe_pair_idx.unsqueeze(-1).expand(-1, -1, self.config.EMBED_DIM)
        x_paired = torch.gather(x_seq, 1, idx_expanded)

        # Zero out features for unpaired positions
        x_paired[unpaired_mask] = 0.0

        # 3. Combine & Project
        x = torch.cat([x_seq, x_loop, x_dist, x_paired], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.dropout(x)

        # 4. Backbone
        for i in range(self.config.NUM_LAYERS):
            residual = x
            x = self.layer_norms[i](x)
            x, _ = self.layers[i](x)
            x = x + residual

        # 5. Heads
        reg_out = self.reg_head(x)
        recon_out = self.recon_head(x)

        return reg_out, recon_out


# ==================================================================================
# TRAINING & UTILS
# ==================================================================================


def mcrmse_loss(pred, target):
    """
    Calculates Mean Columnwise Root Mean Squared Error.
    pred, target: (B, 68, 3)
    """
    # Flatten batch and sequence dims: (B*68, 3)
    pred_flat = pred.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)

    # MSE per column
    mse = torch.mean((pred_flat - target_flat) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    # Mean across columns
    return torch.mean(rmse)


def train_model(config=Config()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")

    # Data Loaders
    train_dataset = RNADataset("train", config)
    val_dataset = RNADataset("val", config)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Setup
    model = RNAModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    best_mcrmse = float("inf")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_idx = batch["pair_idx"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            mask_labels = batch["mask_labels"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            reg_out, recon_out = model(seq, loop, pair_idx, pair_dist)

            # 1. Regression Loss (Scored columns only)
            scored_targets = targets[:, : config.SCORED_LEN, config.SCORED_INDICES]
            scored_preds = reg_out[:, : config.SCORED_LEN, :]
            loss_reg = F.mse_loss(scored_preds, scored_targets)

            # 2. Reconstruction Loss (Masked tokens only)
            loss_recon = F.cross_entropy(recon_out.transpose(1, 2), mask_labels)

            # Joint Loss
            loss = loss_reg + config.LAMBDA_RECON * loss_recon

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_mcrmse_accum = 0
        steps = 0

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                pair_idx = batch["pair_idx"].to(device)
                pair_dist = batch["pair_dist"].to(device)
                targets = batch["targets"].to(device)

                reg_out, _ = model(seq, loop, pair_idx, pair_dist)

                scored_targets = targets[:, : config.SCORED_LEN, config.SCORED_INDICES]
                scored_preds = reg_out[:, : config.SCORED_LEN, :]

                val_mcrmse_accum += mcrmse_loss(scored_preds, scored_targets).item()
                steps += 1

        val_mcrmse = val_mcrmse_accum / steps
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(
                model.state_dict(), os.path.join(config.WORKING_DIR, "best_model.pth")
            )

    print(f"Training Complete. Best Val MCRMSE: {best_mcrmse:.6f}")


def predict_and_submit(config=Config()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    test_dataset = RNADataset("test", config)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Best Model
    model = RNAModel(config).to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: Best model not found, using initialized weights.")

    model.eval()
    all_preds = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_idx = batch["pair_idx"].to(device)
            pair_dist = batch["pair_dist"].to(device)

            reg_out, _ = model(seq, loop, pair_idx, pair_dist)
            all_preds.append(reg_out.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 3)

    # Generate Submission
    print("Generating submission file...")
    submission_rows = []
    ids = test_dataset.ids

    for i, sample_id in enumerate(ids):
        pred_matrix = all_preds[i]

        for pos in range(config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"

            # Map predictions to columns
            # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": pred_matrix[pos, 0],
                    "deg_Mg_pH10": pred_matrix[pos, 1],
                    "deg_pH10": 0.0,  # Unscored
                    "deg_Mg_50C": pred_matrix[pos, 2],
                    "deg_50C": 0.0,  # Unscored
                }
            )

    df_sub = pd.DataFrame(submission_rows)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
