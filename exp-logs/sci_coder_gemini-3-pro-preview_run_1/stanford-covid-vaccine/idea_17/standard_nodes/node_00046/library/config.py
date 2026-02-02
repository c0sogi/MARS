import os
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


class ModelConfig:
    hidden_dim = 384
    num_layers = 6
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20

    # Paths
    train_file = "./metadata/train.parquet"
    val_file = "./metadata/val.parquet"
    test_file = "./metadata/test.parquet"
    output_dir = "./working/idea_18"
    submission_file = "./submission/submission.csv"


# ==================================================================================
# UTILS
# ==================================================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def parse_structure_pairs(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a mapping {index: paired_index}.
    """
    pairs = {}
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pairs[start] = i
                pairs[i] = start
    return pairs


def get_distance_vector(structure, seq_len):
    """
    Returns a vector of length seq_len where v[i] = j - i if i is paired with j, else 0.
    Also returns a mask where m[i] = 1 if paired, else 0.
    """
    pairs = parse_structure_pairs(structure)
    dists = np.zeros(seq_len, dtype=np.float32)
    mask = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        if i in pairs:
            dists[i] = pairs[i] - i
            mask[i] = 1.0
    return dists, mask


def preprocess_dataframe(df, mode="train"):
    """
    Converts dataframe columns to numpy arrays suitable for training/inference.
    """
    # Mappings
    token_map = {c: i for i, c in enumerate(["A", "G", "U", "C"])}
    loop_map = {c: i for i, c in enumerate(["S", "M", "I", "B", "H", "E", "X"])}

    seqs = df["sequence"].values
    structs = df["structure"].values
    loops = df["predicted_loop_type"].values
    ids = df["id"].values

    n_samples = len(seqs)
    seq_len = 107

    # Pre-allocate arrays
    X_seq = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_loop = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_dist = np.zeros((n_samples, seq_len), dtype=np.float32)
    X_mask = np.zeros((n_samples, seq_len), dtype=np.float32)

    for i in range(n_samples):
        # Sequence
        X_seq[i] = [token_map.get(c, 0) for c in seqs[i]]
        # Loop
        X_loop[i] = [loop_map.get(c, 0) for c in loops[i]]
        # Distance
        d, m = get_distance_vector(structs[i], seq_len)
        X_dist[i] = d
        X_mask[i] = m

    data_dict = {
        "seq": X_seq,
        "loop": X_loop,
        "dist": X_dist,
        "mask": X_mask,
        "id": ids,
    }

    if mode in ["train", "val"]:
        # Stack targets
        # Note: In the parquet, these are likely arrays/lists.
        # We assume the metadata parquet has them as lists or numpy arrays.
        # We need to ensure they are stacked correctly.
        reactivity = np.vstack(df["reactivity"].values)
        deg_Mg_pH10 = np.vstack(df["deg_Mg_pH10"].values)
        deg_Mg_50C = np.vstack(df["deg_Mg_50C"].values)

        # Shape: (N, 68, 3)
        targets = np.stack([reactivity, deg_Mg_pH10, deg_Mg_50C], axis=2).astype(
            np.float32
        )
        data_dict["target"] = targets

    return data_dict


def load_or_process_data(load_cached_data=True):
    os.makedirs(ModelConfig.output_dir, exist_ok=True)

    files = {
        "train": (
            ModelConfig.train_file,
            os.path.join(ModelConfig.output_dir, "train_data.npz"),
        ),
        "val": (
            ModelConfig.val_file,
            os.path.join(ModelConfig.output_dir, "val_data.npz"),
        ),
        "test": (
            ModelConfig.test_file,
            os.path.join(ModelConfig.output_dir, "test_data.npz"),
        ),
    }

    datasets = {}

    for mode, (input_path, cache_path) in files.items():
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            loaded = np.load(cache_path, allow_pickle=True)
            # Convert back to dict
            data_dict = {k: loaded[k] for k in loaded.files}
            datasets[mode] = data_dict
        else:
            print(f"Processing {mode} data from {input_path}...")
            df = pd.read_parquet(input_path)
            data_dict = preprocess_dataframe(df, mode=mode)
            np.savez(cache_path, **data_dict)
            datasets[mode] = data_dict

    return datasets["train"], datasets["val"], datasets["test"]


# ==================================================================================
# DATASET CLASS
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.dist = data_dict["dist"]
        self.mask = data_dict["mask"]
        self.ids = data_dict["id"]

        if mode in ["train", "val"]:
            self.target = data_dict["target"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        item = {
            "seq": torch.tensor(self.seq[idx], dtype=torch.long),
            "loop": torch.tensor(self.loop[idx], dtype=torch.long),
            "dist": torch.tensor(self.dist[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.float32),
            "id": str(self.ids[idx]),
        }

        if self.mode in ["train", "val"]:
            item["target"] = torch.tensor(self.target[idx], dtype=torch.float32)

        return item


# ==================================================================================
# MODEL
# ==================================================================================


class RBFLayer(nn.Module):
    def __init__(self, num_rbf=128, max_dist=110):
        super().__init__()
        self.num_rbf = num_rbf
        # Initialize centers uniformly across the range of possible distances
        centers = torch.linspace(-max_dist, max_dist, num_rbf)
        self.centers = nn.Parameter(centers)
        # Widths
        self.sigma = nn.Parameter(torch.ones(num_rbf) * (2 * max_dist / num_rbf))

    def forward(self, dists):
        # dists: (B, L)
        d = dists.unsqueeze(-1)  # (B, L, 1)
        c = self.centers.view(1, 1, -1)  # (1, 1, num_rbf)
        s = self.sigma.view(1, 1, -1)
        # Gaussian RBF
        rbf = torch.exp(-((d - c) ** 2) / (s**2))
        return rbf


class RBFEncodedBiGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        # Embeddings
        self.seq_embed = nn.Embedding(4, self.hidden_dim // 2)
        self.loop_embed = nn.Embedding(7, self.hidden_dim // 2)

        # RBF Encoding
        self.rbf = RBFLayer(num_rbf=config.num_rbf)
        self.rbf_proj = nn.Linear(config.num_rbf, self.hidden_dim // 2)

        # Input Projection to Wide Stream (2 * hidden_dim)
        input_feat_dim = (self.hidden_dim // 2) * 3
        self.input_proj = nn.Linear(input_feat_dim, 2 * self.hidden_dim)

        # BiGRU Layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.norms.append(nn.LayerNorm(2 * self.hidden_dim))
            self.layers.append(
                nn.GRU(
                    input_size=2 * self.hidden_dim,
                    hidden_size=self.hidden_dim,  # Bidirectional -> 2*H output
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # Layer Aggregation Weights
        self.agg_weights = nn.Parameter(torch.zeros(self.num_layers + 1))

        # Output Head
        self.head = nn.Linear(2 * self.hidden_dim, 3)

    def forward(self, seq, loop, dist, dist_mask):
        # Embeddings
        x_seq = self.seq_embed(seq)
        x_loop = self.loop_embed(loop)

        # RBF
        x_rbf = self.rbf(dist)
        x_rbf = self.rbf_proj(x_rbf)
        # Mask RBF for unpaired bases
        x_rbf = x_rbf * dist_mask.unsqueeze(-1)

        # Concatenate
        x = torch.cat([x_seq, x_loop, x_rbf], dim=-1)

        # Project to Wide Stream
        x = self.input_proj(x)

        # Store states for aggregation
        states = [x]

        # Residual BiGRU Layers
        curr = x
        for norm, layer in zip(self.norms, self.layers):
            res = curr
            curr = norm(curr)
            out, _ = layer(curr)
            curr = res + out
            states.append(curr)

        # Layer Aggregation
        w = F.softmax(self.agg_weights, dim=0)
        out_agg = 0
        for i, state in enumerate(states):
            out_agg = out_agg + w[i] * state

        # Head
        logits = self.head(out_agg)
        return logits


# ==================================================================================
# TRAINING & EXECUTION
# ==================================================================================


def train_and_predict():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_data, val_data, test_data = load_or_process_data(load_cached_data=True)

    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")
    test_ds = RNADataset(test_data, mode="test")

    train_loader = DataLoader(
        train_ds, batch_size=ModelConfig.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=ModelConfig.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=ModelConfig.batch_size, shuffle=False, num_workers=2
    )

    # Model
    model = RBFEncodedBiGRU(ModelConfig).to(device)
    optimizer = AdamW(model.parameters(), lr=ModelConfig.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=ModelConfig.num_epochs)

    best_mcrmse = float("inf")
    best_model_path = os.path.join(ModelConfig.output_dir, "best_model.pth")

    print("Starting training...")
    for epoch in range(ModelConfig.num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)  # (B, 68, 3)

            optimizer.zero_grad()

            pred = model(seq, loop, dist, mask)  # (B, 107, 3)

            # Mask loss to first 68 positions
            pred_scored = pred[:, :68, :]

            loss = F.mse_loss(pred_scored, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)

                pred = model(seq, loop, dist, mask)
                pred_scored = pred[:, :68, :]

                val_preds.append(pred_scored.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)  # (N, 68, 3)
        val_targets = np.concatenate(val_targets, axis=0)  # (N, 68, 3)

        # Calculate MCRMSE (Column-wise RMSE averaged)
        rmses = []
        for col in range(3):
            mse = np.mean((val_preds[:, :, col] - val_targets[:, :, col]) ** 2)
            rmses.append(np.sqrt(mse))

        mcrmse = np.mean(rmses)

        print(
            f"Epoch {epoch+1}/{ModelConfig.num_epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {mcrmse:.10f}"
        )

        if mcrmse < best_mcrmse:
            best_mcrmse = mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Val MCRMSE: {best_mcrmse:.10f}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    submission_data = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"]

            pred = model(seq, loop, dist, mask)  # (B, 107, 3)
            pred = pred.cpu().numpy()

            for i in range(len(batch_ids)):
                ids.append(batch_ids[i])
                submission_data.append(pred[i])

    # Format Submission
    final_rows = []
    for i, sample_id in enumerate(ids):
        preds = submission_data[i]  # (107, 3)
        for seqpos in range(107):
            row_id = f"{sample_id}_{seqpos}"
            reactivity = preds[seqpos, 0]
            deg_Mg_pH10 = preds[seqpos, 1]
            deg_pH10 = 0.0
            deg_Mg_50C = preds[seqpos, 2]
            deg_50C = 0.0

            final_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    sub_df = pd.DataFrame(
        final_rows,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )
    os.makedirs(os.path.dirname(ModelConfig.submission_file), exist_ok=True)
    sub_df.to_csv(ModelConfig.submission_file, index=False)
    print(f"Submission saved to {ModelConfig.submission_file}")


# Execute the pipeline
if __name__ == "__main__":
    train_and_predict()
