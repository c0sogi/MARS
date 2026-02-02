import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==========================================
# Configuration
# ==========================================
class Config:
    # Model Hyperparameters
    input_dim_seq = 128
    input_dim_loop = 64
    input_dim_dist = 64
    hidden_dim = 512
    n_layers = 6
    dropout = 0.2

    # Training Hyperparameters
    batch_size = 32
    epochs = 20
    lr = 1e-3
    weight_decay = 1e-4
    max_grad_norm = 1.0

    # Data Dimensions
    seq_len = 107
    pred_len = 68

    # Paths
    train_data_path = "./metadata/train.parquet"
    val_data_path = "./metadata/val.parquet"
    test_data_path = "./metadata/test.parquet"
    cache_dir = "./working/idea_41/"
    submission_path = "./submission/submission.csv"
    model_save_path = "./working/idea_41/best_model.pth"


# ==========================================
# Data Processing & Dataset
# ==========================================
def get_structure_distance_matrix(structure, seq_len):
    """
    Parses dot-bracket structure to get signed pairing distances.
    Returns an array of length seq_len where value is (pair_index - current_index).
    Unpaired bases get 0.
    """
    stack = []
    mapping = {}
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                mapping[start] = i
                mapping[i] = start

    distances = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len):
        if i in mapping:
            distances[i] = mapping[i] - i
        else:
            distances[i] = 0
    return distances


class RNADataset(Dataset):
    def __init__(self, df=None, mode="train", data_dict=None):
        self.mode = mode

        if data_dict is not None:
            # Load from pre-processed dictionary
            self.ids = data_dict["ids"]
            self.seqs = data_dict["seqs"]
            self.loops = data_dict["loops"]
            self.dists = data_dict["dists"]
            if mode != "test":
                self.targets = data_dict["targets"]
        else:
            # Process from DataFrame
            self.ids = df["id"].values

            # Sequence Encoding: A:0, G:1, C:2, U:3
            self.seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
            self.seqs = []
            for s in df["sequence"].values:
                self.seqs.append([self.seq_map.get(c, 0) for c in s])
            self.seqs = np.array(self.seqs, dtype=np.int64)

            # Loop Type Encoding: S, M, I, B, H, E, X
            self.loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
            self.loops = []
            for l in df["predicted_loop_type"].values:
                self.loops.append([self.loop_map.get(c, 0) for c in l])
            self.loops = np.array(self.loops, dtype=np.int64)

            # Distance Encoding
            self.dists = []
            for s in df["structure"].values:
                self.dists.append(get_structure_distance_matrix(s, Config.seq_len))
            self.dists = np.array(self.dists, dtype=np.float32)

            # Targets
            if self.mode != "test":
                # Stack targets: reactivity, deg_Mg_pH10, deg_Mg_50C
                def process_target(col_name):
                    raw = df[col_name].values
                    padded = np.zeros((len(raw), Config.seq_len), dtype=np.float32)
                    for i, arr in enumerate(raw):
                        length = len(arr)
                        padded[i, :length] = arr
                    return padded

                t1 = process_target("reactivity")
                t2 = process_target("deg_Mg_pH10")
                t3 = process_target("deg_Mg_50C")

                self.targets = np.stack([t1, t2, t3], axis=2)  # (N, 107, 3)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        out = {
            "seq": torch.tensor(self.seqs[idx], dtype=torch.long),
            "loop": torch.tensor(self.loops[idx], dtype=torch.long),
            "dist": torch.tensor(self.dists[idx], dtype=torch.float),
        }
        if self.mode != "test":
            out["y"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return out


def load_data(load_cached_data=True):
    os.makedirs(Config.cache_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    paths = [Config.train_data_path, Config.val_data_path, Config.test_data_path]
    datasets = {}

    for split, path in zip(splits, paths):
        cache_file = os.path.join(Config.cache_dir, f"cached_{split}.npz")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split} from cache...")
            data = np.load(cache_file, allow_pickle=True)
            # Reconstruct dict for Dataset
            data_dict = {
                "ids": data["ids"],
                "seqs": data["seqs"],
                "loops": data["loops"],
                "dists": data["dists"],
            }
            if split != "test":
                data_dict["targets"] = data["targets"]

            datasets[split] = RNADataset(mode=split, data_dict=data_dict)

        else:
            print(f"Processing {split} data...")
            df = pd.read_parquet(path)
            ds = RNADataset(df, mode=split)

            # Save to cache
            save_dict = {
                "ids": ds.ids,
                "seqs": ds.seqs,
                "loops": ds.loops,
                "dists": ds.dists,
            }
            if split != "test":
                save_dict["targets"] = ds.targets

            np.savez(cache_file, **save_dict)
            datasets[split] = ds

    return datasets["train"], datasets["val"], datasets["test"]


# ==========================================
# Model Architecture
# ==========================================
class SinusoidalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, L) - signed distances
        device = x.device
        B, L = x.shape

        # Create frequencies
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float()
            * -(math.log(10000.0) / self.dim)
        )

        x_expanded = x.unsqueeze(-1)  # (B, L, 1)

        pe = torch.zeros(B, L, self.dim, device=device)
        pe[:, :, 0::2] = torch.sin(x_expanded * div_term)
        pe[:, :, 1::2] = torch.cos(x_expanded * div_term)

        return pe


class BiGRUBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, hidden_dim)
        residual = x
        x = self.norm(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x + residual


class RNAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_seq = nn.Embedding(4, config.input_dim_seq)
        self.emb_loop = nn.Embedding(7, config.input_dim_loop)
        self.emb_dist = SinusoidalEncoding(config.input_dim_dist)

        input_dim = config.input_dim_seq + config.input_dim_loop + config.input_dim_dist

        # Stem
        self.stem = nn.GRU(
            input_dim, config.hidden_dim // 2, batch_first=True, bidirectional=True
        )

        # Backbone (Wide-Stream)
        self.blocks = nn.ModuleList(
            [
                BiGRUBlock(config.hidden_dim, config.dropout)
                for _ in range(config.n_layers)
            ]
        )

        # Scalar Mixture Aggregation
        self.weights = nn.Parameter(torch.zeros(config.n_layers + 1))

        # Shared Head
        self.head = nn.Linear(config.hidden_dim, 3)

    def forward(self, seq, loop, dist):
        # Embeddings
        x_seq = self.emb_seq(seq)
        x_loop = self.emb_loop(loop)
        x_dist = self.emb_dist(dist)

        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)

        # Stem
        x, _ = self.stem(x)  # (B, L, 512)
        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregation
        stacked = torch.stack(outputs, dim=-1)  # (B, L, 512, n_layers+1)
        w = torch.softmax(self.weights, dim=0)
        aggregated = (stacked * w).sum(dim=-1)

        # Head
        out = self.head(aggregated)
        return out


# ==========================================
# Training & Evaluation
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    all_sq_errors = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            y = batch["y"].to(device)

            # Evaluate only on scored positions
            y_target = y[:, : Config.pred_len, :]

            preds = model(seq, loop, dist)
            preds_sliced = preds[:, : Config.pred_len, :]

            # Squared Error per element
            sq_err = (preds_sliced - y_target) ** 2
            all_sq_errors.append(sq_err.cpu().numpy())

    all_sq_errors = np.concatenate(all_sq_errors, axis=0)  # (N, 68, 3)

    # MCRMSE: Mean of RMSE per column
    mse_per_col = np.mean(all_sq_errors, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)
    mcrmse = np.mean(rmse_per_col)

    return mcrmse


def train_model():
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_ds, val_ds, _ = load_data()
    train_loader = DataLoader(
        train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2
    )

    # Model Setup
    model = RNAModel(Config).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")

    for epoch in range(Config.epochs):
        model.train()
        train_loss_accum = 0
        count = 0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            y = batch["y"].to(device)

            # Masked Loss
            y_target = y[:, : Config.pred_len, :]

            optimizer.zero_grad()
            preds = model(seq, loop, dist)
            preds_sliced = preds[:, : Config.pred_len, :]

            loss = criterion(preds_sliced, y_target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()

            train_loss_accum += loss.item() * seq.size(0)
            count += seq.size(0)

        scheduler.step()

        train_loss = train_loss_accum / count
        val_mcrmse = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train MSE: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.model_save_path)

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    _, _, test_ds = load_data()
    test_loader = DataLoader(
        test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=2
    )

    # Load Best Model
    model = RNAModel(Config).to(device)
    if os.path.exists(Config.model_save_path):
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    else:
        print("Warning: Model file not found. Ensure training has run.")
        return

    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)

            preds = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Format Submission
    test_ids = test_ds.ids
    submission_data = []

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]
        for j in range(Config.seq_len):
            row_id = f"{sample_id}_{j}"
            # Predicted columns: reactivity, deg_Mg_pH10, deg_Mg_50C
            reactivity = sample_preds[j, 0]
            deg_Mg_pH10 = sample_preds[j, 1]
            deg_Mg_50C = sample_preds[j, 2]

            submission_data.append(
                [
                    row_id,
                    reactivity,
                    deg_Mg_pH10,
                    0.0,  # deg_pH10 (not scored)
                    deg_Mg_50C,
                    0.0,  # deg_50C (not scored)
                ]
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

    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    sub_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
