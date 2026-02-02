import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # General
    SEED = 42
    IDEA_NAME = "idea_21"
    DEBUG = False

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test.parquet")

    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    os.makedirs(WORKING_DIR, exist_ok=True)

    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Data Dimensions
    SEQ_LEN = 107
    SEQ_SCORED = 68
    INPUT_DIM = 14  # 4 (ACGU) + 3 (Structure) + 7 (Loop)
    OUTPUT_DIM = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Model Hyperparameters
    HIDDEN_DIM = 768  # High capacity backbone
    NUM_LAYERS = 3
    DROPOUT = 0.1
    CNN_FILTERS = 256
    CNN_KERNEL = 3

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 25
    PATIENCE = 5
    GRAD_CLIP_NORM = 1.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


# ==========================================
# DATA PROCESSING & CACHING
# ==========================================
def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string into an adjacency array.
    adj[i] = j if base i is paired with base j.
    adj[i] = -1 if base i is unpaired.
    """
    n = len(structure)
    adj = np.full(n, -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i
    return adj


def process_dataframe(df, mode="train"):
    # 1. Sequence Encoding (One-Hot)
    # A:0, G:1, C:2, U:3
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    seqs = np.array([[seq_map.get(c, 0) for c in s] for s in df["sequence"]])

    # 2. Structure Encoding (One-Hot)
    # .:0, (:1, ):2
    struct_map = {".": 0, "(": 1, ")": 2}
    structs = np.array([[struct_map.get(c, 0) for c in s] for s in df["structure"]])

    # 3. Loop Type Encoding (One-Hot)
    # S:0, M:1, I:2, B:3, H:4, E:5, X:6
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    loops = np.array(
        [[loop_map.get(c, 0) for c in s] for s in df["predicted_loop_type"]]
    )

    # 4. Adjacency Matrix for Structural Injection
    adj = np.array([get_structure_adj(s) for s in df["structure"]])

    # Combine Features
    N, L = seqs.shape
    features = np.zeros((N, L, Config.INPUT_DIM), dtype=np.float32)

    # Fill Sequence (0-3)
    for i in range(4):
        features[:, :, i] = seqs == i

    # Fill Structure (4-6)
    for i in range(3):
        features[:, :, 4 + i] = structs == i

    # Fill Loop (7-13)
    for i in range(7):
        features[:, :, 7 + i] = loops == i

    data = {"features": features, "adj": adj, "ids": df["id"].values}

    # Process Targets if not test
    if mode != "test":
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        y = np.zeros((N, Config.SEQ_SCORED, 5), dtype=np.float32)
        for i, col in enumerate(target_cols):
            # df[col] is a Series of lists/arrays
            # Stack them into a matrix
            col_data = np.array(df[col].tolist())
            y[:, :, i] = col_data
        data["targets"] = y

    return data


def get_dataset(mode="train", load_cached=True):
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    if load_cached and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        loaded = np.load(cache_file, allow_pickle=True)
        data_dict = {k: loaded[k] for k in loaded.files}
    else:
        print(f"Processing {mode} data from metadata...")
        if mode == "train":
            df = pd.read_parquet(Config.TRAIN_META)
        elif mode == "val":
            df = pd.read_parquet(Config.VAL_META)
        else:
            df = pd.read_parquet(Config.TEST_META)

        data_dict = process_dataframe(df, mode)
        np.savez(cache_file, **data_dict)

    return RNADataset(data_dict, mode)


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.features = torch.from_numpy(data_dict["features"]).float()
        self.adj = torch.from_numpy(data_dict["adj"]).long()
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = torch.from_numpy(data_dict["targets"]).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {"x": self.features[idx], "adj": self.adj[idx], "id": self.ids[idx]}
        if self.mode != "test":
            item["y"] = self.targets[idx]
        return item


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class StructuralInteractionModule(nn.Module):
    """
    Non-Linear Channel-Gated Structural Interaction Module.
    Gathers hidden states from paired bases, applies non-linear projection,
    and selectively injects information via a learned gate.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Value Projection (Non-Linear)
        self.val_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(Config.DROPOUT)
        )

        # Gating Mechanism
        # Input: Concat(h_i, h_j) -> Gate
        self.gate_mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj):
        # h: (B, L, D)
        # adj: (B, L) indices of partners, -1 if unpaired
        B, L, D = h.shape

        # 1. Gather Neighbor States (h_j)
        # Handle -1 by clamping to 0 and masking later
        mask = (adj != -1).unsqueeze(-1).float()  # (B, L, 1)
        adj_clamped = adj.clone()
        adj_clamped[adj == -1] = 0

        # Create batch indices for gathering
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, L)
        h_j = h[batch_idx, adj_clamped]  # (B, L, D)

        # 2. Non-Linear Value Projection
        v_ij = self.val_proj(h_j)

        # 3. Channel-Wise Gating
        cat = torch.cat([h, h_j], dim=-1)  # (B, L, 2D)
        gate_hidden = F.gelu(self.gate_mix(cat))
        z_ij = torch.sigmoid(self.gate_proj(gate_hidden))  # (B, L, D)

        # 4. Selective Injection
        injection = z_ij * v_ij * mask

        # 5. Residual + Norm
        return self.norm(h + injection)


class BiGRU_Block(nn.Module):
    def __init__(self, hidden_dim, use_interaction=True):
        super().__init__()
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.use_interaction = use_interaction
        if use_interaction:
            self.interaction = StructuralInteractionModule(hidden_dim)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x, adj):
        # x: (B, L, D)
        out, _ = self.gru(x)  # (B, L, D)
        out = self.dropout(out)

        if self.use_interaction:
            out = self.interaction(out, adj)

        return out


class RNAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                Config.INPUT_DIM,
                Config.CNN_FILTERS,
                kernel_size=Config.CNN_KERNEL,
                padding=Config.CNN_KERNEL // 2,
            ),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )
        self.proj = nn.Linear(Config.CNN_FILTERS, Config.HIDDEN_DIM)

        # High-Capacity Backbone (3 Blocks)
        # Interaction in first 2 blocks, standard BiGRU in last (as per Idea)
        self.blocks = nn.ModuleList(
            [
                BiGRU_Block(Config.HIDDEN_DIM, use_interaction=True),
                BiGRU_Block(Config.HIDDEN_DIM, use_interaction=True),
                BiGRU_Block(Config.HIDDEN_DIM, use_interaction=False),
            ]
        )

        # Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.OUTPUT_DIM)

    def forward(self, x, adj):
        # x: (B, L, 14)
        x = x.permute(0, 2, 1)  # (B, 14, L)
        x = self.stem(x)
        x = x.permute(0, 2, 1)  # (B, L, Filters)
        x = self.proj(x)

        for block in self.blocks:
            x = block(x, adj)

        out = self.head(x)
        return out


# ==========================================
# TRAINING & INFERENCE
# ==========================================
def mcrmse_loss(pred, target):
    # pred, target: (B, Seq_Scored, 5)
    mse = torch.mean((pred - target) ** 2, dim=1)  # (B, 5)
    rmse = torch.sqrt(mse)
    return torch.mean(rmse)  # Mean over batch and columns


def train_fn(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            y = batch["y"].to(Config.DEVICE)

            optimizer.zero_grad()
            pred = model(x, adj)

            # Slice to scored length
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            loss = mcrmse_loss(pred_scored, y)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_score, val_scored_cols = validate_fn(model, val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val MCRMSE: {val_score:.5f} | Scored Cols: {val_scored_cols:.5f}"
        )

        if val_scored_cols < best_score:
            best_score = val_scored_cols
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break


def validate_fn(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            y = batch["y"].to(Config.DEVICE)

            pred = model(x, adj)
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Global MCRMSE
    mse = np.mean((preds - targets) ** 2, axis=(0, 1))
    rmse = np.sqrt(mse)
    mcrmse = np.mean(rmse)

    # Scored Columns Only (Reactivity, Deg_Mg_pH10, Deg_Mg_50C -> Indices 0, 1, 3)
    scored_indices = [0, 1, 3]
    mcrmse_scored = np.mean(rmse[scored_indices])

    return mcrmse, mcrmse_scored


def predict_fn(model, loader):
    model.eval()
    ids = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(Config.DEVICE)
            adj = batch["adj"].to(Config.DEVICE)
            ids.extend(batch["id"])

            pred = model(x, adj)  # (B, 107, 5)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return ids, preds


def generate_submission():
    # Load Data
    test_ds = get_dataset("test")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = RNAModel().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # Predict
    ids, preds = predict_fn(model, test_loader)

    # Format Submission
    # Rows: id_seqpos
    # Cols: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    id_seqpos = []
    flat_preds = []

    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        # We must predict for ALL sequence positions (length 107) as per prompt
        # "For each sample id in the test set, you must predict targets for each sequence position"
        sample_pred = preds[i]  # (107, 5)

        for j in range(Config.SEQ_LEN):
            id_seqpos.append(f"{sample_id}_{j}")
            flat_preds.append(sample_pred[j])

    submission_df = pd.DataFrame(flat_preds, columns=cols)
    submission_df.insert(0, "id_seqpos", id_seqpos)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
