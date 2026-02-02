import os
import random
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
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_34"
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Data Files
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")

    # Data Dimensions
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Model Architecture
    EMBED_DIM = 128
    HIDDEN_DIM = 384  # Wide stream width
    N_LAYERS = 6
    N_TARGETS = 3  # reactivity, deg_Mg_pH10, deg_Mg_50C
    DROPOUT = 0.1

    # Training
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @staticmethod
    def set_seed():
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed(Config.SEED)
        torch.backends.cudnn.deterministic = True


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def parse_structure(structure_str):
    """
    Parses dot-bracket structure to find pair indices.
    Returns an array of length L where arr[i] is the index of the base paired with i,
    or -1 if unpaired.
    """
    L = len(structure_str)
    pairs = np.full(L, -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure_str):
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
    Processes a dataframe into tensors.
    """
    token2id = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop2id = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    sequences = []
    loop_types = []
    pair_indices = []
    distances = []
    targets = []

    for _, row in df.iterrows():
        # Sequence
        seq = [token2id.get(c, 0) for c in row["sequence"]]
        sequences.append(seq)

        # Loop Type
        loop = [loop2id.get(c, 0) for c in row["predicted_loop_type"]]
        loop_types.append(loop)

        # Structure Pairs & Distances
        pairs = parse_structure(row["structure"])
        pair_indices.append(pairs)

        # Calculate signed distance: j - i
        dists = np.zeros(len(seq), dtype=np.float32)
        for i, p in enumerate(pairs):
            if p != -1:
                dists[i] = p - i
        distances.append(dists)

        if mode in ["train", "val"]:
            # Extract targets: reactivity, deg_Mg_pH10, deg_Mg_50C
            t_react = row["reactivity"]
            t_mg_ph10 = row["deg_Mg_pH10"]
            t_mg_50c = row["deg_Mg_50C"]

            sample_targets = np.zeros((Config.SEQ_LENGTH, 3), dtype=np.float32)
            L_scored = len(t_react)  # Should be 68

            sample_targets[:L_scored, 0] = t_react
            sample_targets[:L_scored, 1] = t_mg_ph10
            sample_targets[:L_scored, 2] = t_mg_50c

            targets.append(sample_targets)

    # Convert to tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    loop_types = torch.tensor(loop_types, dtype=torch.long)
    pair_indices = torch.tensor(np.array(pair_indices), dtype=torch.long)
    distances = torch.tensor(np.array(distances), dtype=torch.float32)

    data_dict = {
        "sequence": sequences,
        "loop_type": loop_types,
        "pair_index": pair_indices,
        "distance": distances,
    }

    if mode in ["train", "val"]:
        data_dict["targets"] = torch.tensor(np.array(targets), dtype=torch.float32)

    data_dict["ids"] = df["id"].values.tolist()

    return data_dict


def get_dataset(config, mode="train", load_cached_data=True):
    """
    Loads data, checks cache, processes if needed.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(config.WORKING_DIR, f"{mode}_data.pt")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        data_dict = torch.load(cache_path)
    else:
        print(f"Processing {mode} data...")
        if mode == "train":
            df = pd.read_parquet(config.TRAIN_PARQUET)
        elif mode == "val":
            df = pd.read_parquet(config.VAL_PARQUET)
        elif mode == "test":
            df = pd.read_parquet(config.TEST_PARQUET)
        else:
            raise ValueError("Invalid mode")

        data_dict = process_data(df, mode)
        torch.save(data_dict, cache_path)
        print(f"Saved {mode} data to {cache_path}")

    return RNADataset(data_dict, mode)


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.sequence = data_dict["sequence"]
        self.loop_type = data_dict["loop_type"]
        self.pair_index = data_dict["pair_index"]
        self.distance = data_dict["distance"]
        self.ids = data_dict["ids"]
        self.mode = mode
        if mode in ["train", "val"]:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        item = {
            "sequence": self.sequence[idx],
            "loop_type": self.loop_type[idx],
            "pair_index": self.pair_index[idx],
            "distance": self.distance[idx],
        }
        if self.mode in ["train", "val"]:
            item["targets"] = self.targets[idx]
        else:
            item["ids"] = self.ids[idx]  # Pass ID for test set
        return item


# ==================================================================================
# MODEL
# ==================================================================================


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x):
        # x: [batch, seq_len] containing signed distances
        x_expanded = x.unsqueeze(-1)  # [B, L, 1]
        freqs = self.inv_freq.to(x.device)  # [dim/2]
        args = x_expanded * freqs  # [B, L, dim/2]
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, L, dim]
        return pe


class StructuralShortcutBiGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.seq_embed = nn.Embedding(4, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(7, config.EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.EMBED_DIM)

        # Input projection
        input_dim = config.EMBED_DIM * 3
        self.input_proj = nn.Linear(input_dim, config.HIDDEN_DIM)

        # Stem
        self.stem = nn.GRU(
            config.HIDDEN_DIM,
            config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Residual Blocks
        self.blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(config.N_LAYERS):
            self.blocks.append(
                nn.GRU(
                    config.HIDDEN_DIM,
                    config.HIDDEN_DIM // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(config.HIDDEN_DIM))

        self.dropout = nn.Dropout(config.DROPOUT)

        # Scalar Aggregation
        self.aggregation_weights = nn.Parameter(torch.zeros(config.N_LAYERS + 1))

        # Structural Shortcut Readout
        self.readout = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.N_TARGETS),
        )

    def forward(self, batch):
        seq = batch["sequence"]
        loop = batch["loop_type"]
        dist = batch["distance"]
        pair_idx = batch["pair_index"]

        # 1. Embeddings
        emb_seq = self.seq_embed(seq)
        emb_loop = self.loop_embed(loop)
        emb_dist = self.dist_embed(dist)

        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)
        x = self.input_proj(x)

        # 2. Recurrent Stem
        x, _ = self.stem(x)
        layer_outputs = [x]

        # 3. Residual Blocks
        current_x = x
        for i, block in enumerate(self.blocks):
            norm_x = self.layer_norms[i](current_x)
            out, _ = block(norm_x)
            out = self.dropout(out)
            current_x = current_x + out
            layer_outputs.append(current_x)

        # 4. Aggregation
        weights = F.softmax(self.aggregation_weights, dim=0)
        stacked_outputs = torch.stack(layer_outputs, dim=0)
        agg_state = (stacked_outputs * weights.view(-1, 1, 1, 1)).sum(
            dim=0
        )  # [B, L, H]

        # 5. Structural Shortcut Readout
        B, L, H = agg_state.shape

        # Mask unpaired indices (-1)
        valid_mask = pair_idx != -1  # [B, L]
        safe_pair_idx = pair_idx.clone()
        safe_pair_idx[~valid_mask] = 0

        # Gather partner states: [B, L, H]
        gather_idx = safe_pair_idx.unsqueeze(-1).expand(-1, -1, H)
        partner_states = torch.gather(agg_state, 1, gather_idx)

        # Zero out unpaired partners
        partner_states = partner_states * valid_mask.unsqueeze(-1).float()

        # Concatenate [h_i, h_j]
        fused = torch.cat([agg_state, partner_states], dim=-1)  # [B, L, 2H]
        logits = self.readout(fused)  # [B, L, 3]

        return logits


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================


def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        preds = model(batch)
        targets = batch["targets"]

        # Slice to scored region
        preds_scored = preds[:, : config.SEQ_SCORED, :]
        targets_scored = targets[:, : config.SEQ_SCORED, :]

        loss = F.mse_loss(preds_scored, targets_scored)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            targets = batch["targets"]

            # Slice
            preds_scored = preds[:, : config.SEQ_SCORED, :]
            targets_scored = targets[:, : config.SEQ_SCORED, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    all_preds = torch.cat(all_preds, dim=0)  # [Total_Samples, 68, 3]
    all_targets = torch.cat(all_targets, dim=0)

    # MCRMSE: Mean Columnwise RMSE
    # Flatten samples and positions: [Total_Pixels, 3]
    flat_preds = all_preds.view(-1, 3)
    flat_targets = all_targets.view(-1, 3)

    mse = torch.mean((flat_preds - flat_targets) ** 2, dim=0)  # [3]
    rmse = torch.sqrt(mse)
    mcrmse = torch.mean(rmse).item()

    return mcrmse


def run_training(config=Config):
    config.set_seed()

    train_ds = get_dataset(config, "train")
    val_ds = get_dataset(config, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = StructuralShortcutBiGRU(config).to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, config.DEVICE, config
        )
        val_mcrmse = validate(model, val_loader, config.DEVICE, config)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.10f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved!")

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.10f}")
    return best_model_path


def run_inference(config=Config):
    print("Generating submission...")
    config.set_seed()

    test_ds = get_dataset(config, "test")
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = StructuralShortcutBiGRU(config).to(config.DEVICE)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Model not found, skipping inference.")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(config.DEVICE)

            ids = batch["ids"]  # List of IDs
            preds = model(batch)  # [B, 107, 3]
            preds = preds.cpu().numpy()

            for i, sample_id in enumerate(ids):
                sample_preds = preds[i]

                for seqpos in range(config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seqpos}"

                    # Map predictions to columns
                    reactivity = sample_preds[seqpos, 0]
                    deg_Mg_pH10 = sample_preds[seqpos, 1]
                    deg_pH10 = 0.0  # Not predicted
                    deg_Mg_50C = sample_preds[seqpos, 2]
                    deg_50C = 0.0  # Not predicted

                    results.append(
                        {
                            "id_seqpos": row_id,
                            "reactivity": reactivity,
                            "deg_Mg_pH10": deg_Mg_pH10,
                            "deg_pH10": deg_pH10,
                            "deg_Mg_50C": deg_Mg_50C,
                            "deg_50C": deg_50C,
                        }
                    )

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
