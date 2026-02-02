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


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    CACHE_FILE = os.path.join(WORKING_DIR, "processed_data.pt")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data Dimensions
    SEQ_LENGTH = 107
    SCORABLE_LENGTH = 68

    # Targets
    # We train on these 3 columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # We must predict 5 columns in total (filling others with 0)
    UNSCORED_COLS = ["deg_pH10", "deg_50C"]

    # Model Hyperparameters
    HIDDEN_DIM = 384
    NUM_GRU_LAYERS = 5
    DROPOUT = 0.1

    # Vocabularies
    VOCAB_SIZE = 4  # A, G, U, C
    LOOP_VOCAB_SIZE = 7  # S, M, I, B, H, E, X

    # Training Settings
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================================
# UTILS & DATA PROCESSING
# ==================================================================================


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_structure_indices(structure):
    """
    Parses dot-bracket structure to find pair indices.
    Returns an array where arr[i] = j if i is paired with j, else -1.
    """
    pair_index = np.full(len(structure), -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pair_index[start] = i
                pair_index[i] = start
    return pair_index


class RNADataset(Dataset):
    def __init__(self, df, mode="train"):
        self.mode = mode
        self.seqs = df["sequence"].values
        self.structs = df["structure"].values
        self.loops = df["predicted_loop_type"].values
        self.ids = df["id"].values

        # Mappings
        self.seq_map = {c: i for i, c in enumerate("AGUC")}
        self.loop_map = {c: i for i, c in enumerate("SMIBHEX")}

        # Targets
        if mode in ["train", "val"]:
            # Stack targets: (N, 3, 68)
            # We explicitly select the 3 scored columns
            t1 = np.vstack(df["reactivity"].values)
            t2 = np.vstack(df["deg_Mg_pH10"].values)
            t3 = np.vstack(df["deg_Mg_50C"].values)
            self.targets = np.stack([t1, t2, t3], axis=2)  # (N, 68, 3)
        else:
            self.targets = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Sequence
        seq_str = self.seqs[idx]
        seq_ints = [self.seq_map.get(c, 0) for c in seq_str]

        # Loop
        loop_str = self.loops[idx]
        loop_ints = [self.loop_map.get(c, 0) for c in loop_str]

        # Structure Pairs
        struct_str = self.structs[idx]
        pair_indices = get_structure_indices(struct_str)

        # Convert to tensors
        seq_t = torch.tensor(seq_ints, dtype=torch.long)
        loop_t = torch.tensor(loop_ints, dtype=torch.long)
        pair_t = torch.tensor(pair_indices, dtype=torch.long)

        item = {"seq": seq_t, "loop": loop_t, "pair_index": pair_t, "id": self.ids[idx]}

        if self.targets is not None:
            # Targets are (68, 3)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = y

        return item


def load_and_process_data(load_cached_data=True):
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.CACHE_FILE):
        print(f"Loading cached data from {Config.CACHE_FILE}")
        return torch.load(Config.CACHE_FILE, weights_only=False)

    print("Processing data from scratch...")
    # Load Parquet files from metadata
    df_train = pd.read_parquet(os.path.join(Config.METADATA_DIR, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(Config.METADATA_DIR, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(Config.METADATA_DIR, "test.parquet"))

    # Create Datasets
    train_ds = RNADataset(df_train, mode="train")
    val_ds = RNADataset(df_val, mode="val")
    test_ds = RNADataset(df_test, mode="test")

    data = {"train": train_ds, "val": val_ds, "test": test_ds}

    # Cache
    torch.save(data, Config.CACHE_FILE)
    return data


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x contains distances. Clamp to range [0, max_len-1]
        indices = x.clamp(0, self.pe.size(0) - 1)
        return self.pe[indices]


class StructureAugmentedHybridNetwork(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM

        # 1. Embeddings
        self.seq_emb = nn.Embedding(config.VOCAB_SIZE, self.hidden_dim)
        self.loop_emb = nn.Embedding(config.LOOP_VOCAB_SIZE, self.hidden_dim)

        # 2. Distance Encoding (Sinusoidal)
        # Max distance is approx seq_length.
        self.dist_emb = SinusoidalPositionalEncoding(
            self.hidden_dim, max_len=config.SEQ_LENGTH + 5
        )

        # 3. Paired-Base Identity (Teleportation)
        # Special token for unpaired base
        self.unpaired_emb = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # 4. Input Projection
        # Concatenate: Seq(H) + Loop(H) + Dist(H) + PairedSeq(H) = 4H
        self.input_proj = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.DROPOUT),
        )

        # 5. Stage 1: Pre-LN BiGRU Stack
        self.gru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.NUM_GRU_LAYERS):
            self.norms.append(nn.LayerNorm(self.hidden_dim))
            self.gru_layers.append(
                nn.GRU(
                    self.hidden_dim,
                    self.hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # 6. Stage 2: Transformer Encoder Refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.NHEAD,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_TRANSFORMER_LAYERS
        )

        # 7. Output Head
        self.head = nn.Linear(self.hidden_dim, 3)  # Predict 3 scored targets

    def forward(self, seq, loop, pair_index):
        # seq: (B, L)
        # pair_index: (B, L), values -1 or index j

        B, L = seq.shape

        # Base Embeddings
        x_seq = self.seq_emb(seq)  # (B, L, H)
        x_loop = self.loop_emb(loop)  # (B, L, H)

        # Distance Embeddings
        # Create grid of indices (0..L-1)
        indices = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)

        # If paired, dist = abs(i - pair_index). If unpaired, dist = 0 (arbitrary, masked later implicitly or explicitly)
        # Note: pair_index is -1 for unpaired.
        is_paired = pair_index != -1

        # Calculate distance only where paired, else 0
        dist = torch.zeros_like(indices)
        dist[is_paired] = torch.abs(indices[is_paired] - pair_index[is_paired])

        x_dist = self.dist_emb(dist)  # (B, L, H)

        # Paired-Base Identity (Teleportation)
        # We want to gather the embedding of the paired base.
        # Construct gather indices. For unpaired (-1), we use 0 temporarily.
        gather_indices = pair_index.clone()
        gather_indices[~is_paired] = 0

        # Expand for gather: (B, L, H)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )
        x_paired = torch.gather(x_seq, 1, gather_indices_expanded)

        # Replace unpaired positions with special token
        x_paired = torch.where(
            is_paired.unsqueeze(-1), x_paired, self.unpaired_emb.expand(B, L, -1)
        )

        # Combine
        x = torch.cat([x_seq, x_loop, x_dist, x_paired], dim=-1)
        x = self.input_proj(x)

        # BiGRU (Residual Pre-LN)
        for norm, gru in zip(self.norms, self.gru_layers):
            x_norm = norm(x)
            x_gru, _ = gru(x_norm)
            x = x + x_gru

        # Transformer
        x = self.transformer(x)

        # Head
        logits = self.head(x)  # (B, L, 3)

        return logits


# ==================================================================================
# PIPELINE EXECUTION
# ==================================================================================


def train_model():
    seed_everything(Config.SEED)

    # Load Data
    datasets = load_and_process_data()
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize Model
    model = StructureAugmentedHybridNetwork().to(Config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_val_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0

        for batch in train_loader:
            seq = batch["seq"].to(Config.DEVICE)
            loop = batch["loop"].to(Config.DEVICE)
            pair = batch["pair_index"].to(Config.DEVICE)
            target = batch["targets"].to(Config.DEVICE)  # (B, 68, 3)

            optimizer.zero_grad()

            # Forward
            preds = model(seq, loop, pair)  # (B, 107, 3)

            # Crop to scored length (68) for loss calculation
            preds_scored = preds[:, : Config.SCORABLE_LENGTH, :]

            # Masked MSE Loss (Data is already clean/filtered by metadata gen, but good to be safe)
            loss = F.mse_loss(preds_scored, target)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)

        # Validation
        model.eval()
        val_mse_sum = torch.zeros(3).to(Config.DEVICE)
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(Config.DEVICE)
                loop = batch["loop"].to(Config.DEVICE)
                pair = batch["pair_index"].to(Config.DEVICE)
                target = batch["targets"].to(Config.DEVICE)

                preds = model(seq, loop, pair)
                preds_scored = preds[:, : Config.SCORABLE_LENGTH, :]

                # Sum of squared errors per column
                sq_err = (preds_scored - target) ** 2
                val_mse_sum += sq_err.mean(dim=1).sum(
                    dim=0
                )  # Sum over batch of (mean over seq_len)
                count += seq.size(0)

        # MCRMSE Calculation
        val_rmse_per_col = torch.sqrt(val_mse_sum / count)
        val_mcrmse = val_rmse_per_col.mean().item()

        print(
            f"Epoch {epoch+1} | Train MSE: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_val_mcrmse:
            best_val_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best MCRMSE: {best_val_mcrmse:.6f}")


def generate_submission():
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("No model found. Skipping submission.")
        return

    model = StructureAugmentedHybridNetwork().to(Config.DEVICE)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    datasets = load_and_process_data()
    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    preds_list = []
    ids_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(Config.DEVICE)
            loop = batch["loop"].to(Config.DEVICE)
            pair = batch["pair_index"].to(Config.DEVICE)
            ids = batch["id"]

            preds = model(seq, loop, pair)  # (B, 107, 3)
            preds = preds.cpu().numpy()

            preds_list.append(preds)
            ids_list.extend(ids)

    all_preds = np.concatenate(preds_list, axis=0)  # (N_test, 107, 3)

    # Format Submission
    submission_data = []
    for i, sample_id in enumerate(ids_list):
        pred_matrix = all_preds[i]  # (107, 3)
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"

            # We have predictions for: reactivity, deg_Mg_pH10, deg_Mg_50C
            # Indices: 0, 1, 2
            reactivity = pred_matrix[seqpos, 0]
            deg_Mg_pH10 = pred_matrix[seqpos, 1]
            deg_Mg_50C = pred_matrix[seqpos, 2]

            # Fill unscored columns with 0
            submission_data.append(
                [
                    row_id,
                    float(reactivity),
                    float(deg_Mg_pH10),
                    0.0,  # deg_pH10
                    float(deg_Mg_50C),
                    0.0,  # deg_50C
                ]
            )

    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_data, columns=cols)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    train_model()
    generate_submission()


if __name__ == "__main__":
    # Trigger pipeline
    run_pipeline()
