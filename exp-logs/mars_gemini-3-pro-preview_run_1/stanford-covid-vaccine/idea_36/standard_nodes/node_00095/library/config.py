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


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class Config:
    # Model Hyperparameters
    embed_dim = 128
    hidden_dim = 384
    n_layers = 6
    dropout = 0.1

    # Data Dimensions
    seq_len = 107
    pred_len = 68

    # Training Hyperparameters
    batch_size = 32
    epochs = 20
    lr = 1e-3
    weight_decay = 1e-4

    # Paths
    train_parquet = "./metadata/train.parquet"
    val_parquet = "./metadata/val.parquet"
    test_parquet = "./metadata/test.parquet"
    cache_dir = "./working/idea_36/"
    submission_path = "./submission/submission.csv"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2


# ------------------------------------------------------------------------------
# Data Processing & Caching
# ------------------------------------------------------------------------------
def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns a mapping {index: paired_index}. Unpaired indices are not in the dict.
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
    Processes DataFrame into features.
    """
    # Dictionaries
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    sequences = []
    loops = []
    pair_dists = []

    # Targets
    targets = []
    errors = []
    ids = []

    for idx, row in df.iterrows():
        # 1. Sequence
        seq_int = [seq_map.get(c, 0) for c in row["sequence"]]
        sequences.append(seq_int)

        # 2. Loop Type
        loop_int = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loops.append(loop_int)

        # 3. Pairing Distance
        structure = row["structure"]
        pairs = get_couples(structure)
        dists = []
        for i in range(len(structure)):
            if i in pairs:
                dists.append(pairs[i] - i)  # Signed distance
            else:
                dists.append(0)  # 0 for unpaired
        pair_dists.append(dists)

        ids.append(row["id"])

        if mode in ["train", "val"]:
            # Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
            t_react = row["reactivity"]
            t_mg_ph10 = row["deg_Mg_pH10"]
            t_mg_50c = row["deg_Mg_50C"]

            # Errors
            e_react = row["reactivity_error"]
            e_mg_ph10 = row["deg_error_Mg_pH10"]
            e_mg_50c = row["deg_error_Mg_50C"]

            # Stack and pad
            # Targets shape: [107, 3]
            sample_targets = np.zeros((Config.seq_len, 3), dtype=np.float32)
            sample_errors = np.zeros((Config.seq_len, 3), dtype=np.float32)

            length = min(len(t_react), Config.seq_len)

            sample_targets[:length, 0] = t_react[:length]
            sample_targets[:length, 1] = t_mg_ph10[:length]
            sample_targets[:length, 2] = t_mg_50c[:length]

            sample_errors[:length, 0] = e_react[:length]
            sample_errors[:length, 1] = e_mg_ph10[:length]
            sample_errors[:length, 2] = e_mg_50c[:length]

            targets.append(sample_targets)
            errors.append(sample_errors)

    sequences = np.array(sequences, dtype=np.int64)
    loops = np.array(loops, dtype=np.int64)
    pair_dists = np.array(pair_dists, dtype=np.float32)

    data_dict = {
        "sequences": sequences,
        "loops": loops,
        "pair_dists": pair_dists,
        "ids": ids,
    }

    if mode in ["train", "val"]:
        data_dict["targets"] = np.array(targets, dtype=np.float32)
        data_dict["errors"] = np.array(errors, dtype=np.float32)

    return data_dict


def get_dataset(mode="train", load_cached_data=True):
    """
    Loads and processes dataset with caching.
    """
    os.makedirs(Config.cache_dir, exist_ok=True)
    cache_file = os.path.join(Config.cache_dir, f"{mode}_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        # print(f"Loading cached {mode} data from {cache_file}")
        return np.load(cache_file, allow_pickle=True)

    # print(f"Processing {mode} data...")
    if mode == "train":
        df = pd.read_parquet(Config.train_parquet)
    elif mode == "val":
        df = pd.read_parquet(Config.val_parquet)
    elif mode == "test":
        df = pd.read_parquet(Config.test_parquet)
    else:
        raise ValueError("Invalid mode")

    data = process_data(df, mode=mode)

    # Save to cache
    # print(f"Saving {mode} data to {cache_file}")
    np.savez(cache_file, **data)

    return data


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.sequences = torch.from_numpy(data["sequences"])
        self.loops = torch.from_numpy(data["loops"])
        self.pair_dists = torch.from_numpy(data["pair_dists"])
        self.ids = data["ids"]
        self.mode = mode

        if mode in ["train", "val"]:
            self.targets = torch.from_numpy(data["targets"])
            self.errors = torch.from_numpy(data["errors"])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequence": self.sequences[idx],
            "loop": self.loops[idx],
            "pair_dist": self.pair_dists[idx],
        }
        if self.mode in ["train", "val"]:
            item["target"] = self.targets[idx]
            item["error"] = self.errors[idx]
        return item


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Create constant 'pe' matrix with values dependent on pos and i
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    def forward(self, x):
        # x: [Batch, SeqLen] containing signed distances
        # Output: [Batch, SeqLen, d_model]

        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)

        sin_inp = torch.einsum("bi,j->bij", x, self.inv_freq)

        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return emb


class WideResBiGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.seq_embed = nn.Embedding(4, config.embed_dim)
        self.loop_embed = nn.Embedding(7, config.embed_dim)
        self.dist_embed = SinusoidalPositionalEmbedding(config.embed_dim)

        # Input Projection: 3 * embed_dim -> hidden_dim
        self.input_proj = nn.Linear(config.embed_dim * 3, config.hidden_dim)

        # Stem
        self.stem_gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Backbone: Residual Blocks
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(config.hidden_dim),
                        "gru": nn.GRU(
                            config.hidden_dim,
                            config.hidden_dim // 2,
                            batch_first=True,
                            bidirectional=True,
                        ),
                        "dropout": nn.Dropout(config.dropout),
                    }
                )
            )

        # Scalar Mixture Weights (Stem + 6 Layers = 7 weights)
        self.mix_weights = nn.Parameter(torch.ones(config.n_layers + 1))

        # Heads
        self.value_head = nn.Linear(config.hidden_dim, 3)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 3)

    def forward(self, seq, loop, dist):
        # Embeddings
        e_seq = self.seq_embed(seq)
        e_loop = self.loop_embed(loop)
        e_dist = self.dist_embed(dist)

        # Concatenate
        x = torch.cat([e_seq, e_loop, e_dist], dim=-1)
        x = self.input_proj(x)

        # Stem
        x, _ = self.stem_gru(x)

        # Collect outputs for mixture
        layer_outputs = [x]

        # Backbone
        curr = x
        for block in self.blocks:
            res = curr
            # Pre-LN
            out = block["norm"](curr)
            out, _ = block["gru"](out)
            out = block["dropout"](out)
            # Residual
            curr = res + out
            layer_outputs.append(curr)

        # Scalar Mixture
        # Stack: [Batch, Len, Hidden, Layers]
        stacked = torch.stack(layer_outputs, dim=-1)
        # Softmax weights
        norm_weights = F.softmax(self.mix_weights, dim=0)
        # Weighted Sum
        aggregated = torch.sum(stacked * norm_weights, dim=-1)

        # Heads
        values = self.value_head(aggregated)
        uncertainties = self.uncertainty_head(aggregated)

        return values, uncertainties


# ------------------------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------------------------
def mcrmse_loss(pred, target, mask):
    # Standard MSE on masked region
    # pred, target: [B, L, 3]
    # mask: [B, L]

    # Expand mask for channels
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    mse = F.mse_loss(pred * mask_expanded, target * mask_expanded, reduction="sum")
    count = mask_expanded.sum()
    return mse / (count + 1e-8)


def compute_metric(pred, target):
    # MCRMSE: Mean Columnwise Root Mean Squared Error
    # pred, target: [N, 3] (flattened valid positions)

    rmse_list = []
    for i in range(3):
        mse = np.mean((pred[:, i] - target[:, i]) ** 2)
        rmse_list.append(np.sqrt(mse))

    return np.mean(rmse_list)


def train_one_epoch(model, loader, optimizer, config):
    model.train()
    total_loss = 0

    # Create mask for first 68 positions
    mask = torch.zeros(config.seq_len, device=config.device)
    mask[: config.pred_len] = 1.0

    for batch in loader:
        seq = batch["sequence"].to(config.device)
        loop = batch["loop"].to(config.device)
        dist = batch["pair_dist"].to(config.device)
        target = batch["target"].to(config.device)
        error_target = batch["error"].to(config.device)

        optimizer.zero_grad()

        pred_val, pred_err = model(seq, loop, dist)

        # Apply mask
        batch_mask = mask.unsqueeze(0).expand(seq.size(0), -1)

        # Multi-task Loss
        loss_val = mcrmse_loss(pred_val, target, batch_mask)
        loss_err = mcrmse_loss(pred_err, error_target, batch_mask)

        loss = loss_val + loss_err  # Lambda = 1.0

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(config.device)
            loop = batch["loop"].to(config.device)
            dist = batch["pair_dist"].to(config.device)
            target = batch["target"].to(config.device)

            pred_val, _ = model(seq, loop, dist)

            # Extract valid positions (0 to 68)
            valid_len = config.pred_len

            # Slice [Batch, 68, 3]
            p = pred_val[:, :valid_len, :].cpu().numpy()
            t = target[:, :valid_len, :].cpu().numpy()

            all_preds.append(p)
            all_targets.append(t)

    all_preds = np.concatenate(all_preds, axis=0)  # [N_samples, 68, 3]
    all_targets = np.concatenate(all_targets, axis=0)

    # Reshape to [N_total, 3] for metric calculation
    flat_preds = all_preds.reshape(-1, 3)
    flat_targets = all_targets.reshape(-1, 3)

    score = compute_metric(flat_preds, flat_targets)
    return score


def generate_submission(model, config):
    # Load test data
    test_data = get_dataset(mode="test", load_cached_data=True)
    dataset = RNADataset(test_data, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model.eval()
    ids = test_data["ids"]
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(config.device)
            loop = batch["loop"].to(config.device)
            dist = batch["pair_dist"].to(config.device)

            pred_val, _ = model(seq, loop, dist)
            preds_list.append(pred_val.cpu().numpy())

    # [N_samples, 107, 3]
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission dataframe
    submission_rows = []
    for i, sample_id in enumerate(ids):
        sample_pred = all_preds[i]  # [107, 3]
        for j in range(config.seq_len):
            row_id = f"{sample_id}_{j}"
            reactivity = sample_pred[j, 0]
            deg_Mg_pH10 = sample_pred[j, 1]
            deg_Mg_50C = sample_pred[j, 2]

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": 0.0,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": 0.0,
                }
            )

    sub_df = pd.DataFrame(submission_rows)
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    sub_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def run_training():
    # Set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    config = Config()

    # Data
    train_data = get_dataset(mode="train", load_cached_data=True)
    val_data = get_dataset(mode="val", load_cached_data=True)

    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Model
    model = WideResBiGRU(config).to(config.device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_score = float("inf")

    print(f"Starting training on {config.device}...")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, config)
        val_score = validate(model, val_loader, config)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(
                model.state_dict(), os.path.join(config.cache_dir, "best_model.pth")
            )

    print(f"Best Val MCRMSE: {best_score}")

    # Load best model and generate submission
    model.load_state_dict(torch.load(os.path.join(config.cache_dir, "best_model.pth")))
    generate_submission(model, config)
