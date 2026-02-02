import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# ==========================================
# CONFIGURATION
# ==========================================


class Config:
    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data
    SEQ_LENGTH = 107
    SEQ_SCORED = 68
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Model Hyperparameters
    HIDDEN_DIM = 384
    NUM_LAYERS = 6
    EMBED_DIM_CHAR = 32
    EMBED_DIM_LOOP = 32
    EMBED_DIM_DIST = 32
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 25
    EARLY_STOPPING_PATIENCE = 5
    NUM_WORKERS = 2
    SEED = 42

    def __init__(self):
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.SUBMISSION_PATH), exist_ok=True)


# ==========================================
# UTILS & DATA PROCESSING
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def get_structure_distance_matrix(structure, seq_len):
    """
    Computes the signed distance (j - i) for paired bases.
    Unpaired bases get a distance of 0.
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
            distances[i] = 0.0

    return distances


def sinusoidal_encoding(values, dim):
    """
    Computes sinusoidal encoding for signed values.
    PE(pos, 2i) = sin(pos / 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    """
    device = values.device
    batch_size, seq_len = values.shape

    # div_term: (dim/2)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(batch_size, seq_len, dim, device=device)

    pos = values.unsqueeze(-1)  # (B, L, 1)
    div_term = div_term.unsqueeze(0).unsqueeze(0)  # (1, 1, D/2)

    pe[:, :, 0::2] = torch.sin(pos * div_term)
    pe[:, :, 1::2] = torch.cos(pos * div_term)

    return pe


class RNADataset(Dataset):
    def __init__(self, df, mode="train"):
        self.mode = mode
        self.seq_ids = df["id"].values
        self.sequences = df["sequence"].values
        self.structures = df["structure"].values
        self.loops = df["predicted_loop_type"].values

        # Vocabularies
        self.token_map = {c: i for i, c in enumerate(["A", "G", "C", "U"])}
        self.loop_map = {
            c: i for i, c in enumerate(["S", "M", "I", "B", "H", "E", "X"])
        }

        # Targets
        if self.mode in ["train", "val"]:
            self.targets = []
            for col in Config.TARGET_COLS:
                # df[col] contains lists/arrays, stack them
                vals = np.vstack(df[col].values)
                self.targets.append(vals)
            self.targets = np.stack(self.targets, axis=2)  # (N, 68, 3)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        struct = self.structures[idx]
        loop = self.loops[idx]

        # 1. Sequence Tokens (Atomic)
        seq_ints = [self.token_map.get(c, 0) for c in seq]

        # 2. Loop Tokens
        loop_ints = [self.loop_map.get(c, 0) for c in loop]

        # 3. Signed Distance
        dist = get_structure_distance_matrix(struct, len(seq))

        # Convert to tensors
        seq_t = torch.tensor(seq_ints, dtype=torch.long)
        loop_t = torch.tensor(loop_ints, dtype=torch.long)
        dist_t = torch.tensor(dist, dtype=torch.float)

        out = {"seq": seq_t, "loop": loop_t, "dist": dist_t, "id": self.seq_ids[idx]}

        if self.mode in ["train", "val"]:
            # Targets: (68, 3)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            out["targets"] = y

        return out


def prepare_data(config):
    # Caching logic
    cache_file = os.path.join(config.WORKING_DIR, "processed_data.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        try:
            return torch.load(cache_file)
        except Exception as e:
            print(f"Cache load failed ({e}), reprocessing...")

    print("Processing data from scratch...")
    train_df = pd.read_parquet(config.TRAIN_PARQUET)
    val_df = pd.read_parquet(config.VAL_PARQUET)
    test_df = pd.read_parquet(config.TEST_PARQUET)

    data = {
        "train": RNADataset(train_df, mode="train"),
        "val": RNADataset(val_df, mode="val"),
        "test": RNADataset(test_df, mode="test"),
    }

    torch.save(data, cache_file)
    return data


# ==========================================
# MODEL
# ==========================================


class InputInjectedBiGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.char_emb = nn.Embedding(4, config.EMBED_DIM_CHAR)
        self.loop_emb = nn.Embedding(7, config.EMBED_DIM_LOOP)
        # Distance embedded via sinusoidal function in forward pass

        input_dim = (
            config.EMBED_DIM_CHAR + config.EMBED_DIM_LOOP + config.EMBED_DIM_DIST
        )

        # Encoder Layers with Input Injection
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(config.NUM_LAYERS):
            # Layer 0: Input only
            # Layer >0: Hidden (from prev) + Input (injected)
            gru_input_dim = input_dim if i == 0 else config.HIDDEN_DIM + input_dim

            self.layers.append(
                nn.GRU(
                    input_size=gru_input_dim,
                    hidden_size=config.HIDDEN_DIM // 2,  # Bidirectional
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Pre-LayerNorm for residual stack
            if i > 0:
                self.norms.append(nn.LayerNorm(config.HIDDEN_DIM))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(config.DROPOUT)

        # Output Head (Predicts 3 scored targets)
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 3),
        )

    def forward(self, seq, loop, dist):
        # Embeddings
        x_char = self.char_emb(seq)
        x_loop = self.loop_emb(loop)
        x_dist = sinusoidal_encoding(dist, self.config.EMBED_DIM_DIST)

        # Concatenate Features
        x_input = torch.cat([x_char, x_loop, x_dist], dim=-1)

        # Input Injection Mechanism
        current_features = x_input

        # First Layer
        out, _ = self.layers[0](x_input)
        h_prev = out

        # Subsequent Layers
        for i in range(1, self.config.NUM_LAYERS):
            h_norm = self.norms[i](h_prev)
            gru_in = torch.cat([h_norm, current_features], dim=-1)
            out, _ = self.layers[i](gru_in)
            h_prev = out + h_prev  # Residual

        logits = self.head(h_prev)
        return logits


# ==========================================
# TRAINING & EXECUTION
# ==========================================


def calculate_mcrmse(pred, target):
    """
    MCRMSE = Mean Columnwise Root Mean Squared Error.
    Calculated as mean(sqrt(mean(error^2, axis=0)))
    """
    diff = pred - target
    mse_cols = torch.mean(diff**2, dim=0)
    rmse_cols = torch.sqrt(mse_cols)
    mcrmse = torch.mean(rmse_cols)
    return mcrmse.item()


def train_model():
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    datasets = prepare_data(config)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Model
    model = InputInjectedBiGRU(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")
    patience = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            preds = model(seq, loop, dist)

            # Loss only on scored positions
            preds_scored = preds[:, : config.SEQ_SCORED, :]

            loss = criterion(preds_scored, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seq, loop, dist)
                preds_scored = preds[:, : config.SEQ_SCORED, :]

                all_preds.append(preds_scored.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        val_mcrmse = calculate_mcrmse(all_preds, all_targets)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train MSE: {train_loss/len(train_loader):.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(
                model.state_dict(), os.path.join(config.WORKING_DIR, "best_model.pth")
            )
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(
        torch.load(os.path.join(config.WORKING_DIR, "best_model.pth"))
    )
    model.eval()

    test_loader = DataLoader(
        datasets["test"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist).cpu().numpy()  # (B, 107, 3)

            for i, sample_id in enumerate(ids):
                sample_preds = preds[i]
                for pos in range(config.SEQ_LENGTH):
                    # Output format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    # Model predicts: reactivity, deg_Mg_pH10, deg_Mg_50C
                    row = [
                        f"{sample_id}_{pos}",
                        sample_preds[pos, 0],  # reactivity
                        sample_preds[pos, 1],  # deg_Mg_pH10
                        0.0,  # deg_pH10 (unscored)
                        sample_preds[pos, 2],  # deg_Mg_50C
                        0.0,  # deg_50C (unscored)
                    ]
                    submission_data.append(row)

    sub_df = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


# Execute pipeline
# train_model()
