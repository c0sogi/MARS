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
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_93"
    SUBMISSION_DIR = "./submission"

    # Data
    SEQ_LEN = 107
    SEQ_SCORED = 68
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Model Hyperparameters
    INPUT_CHANNELS = 14  # 4 (seq) + 3 (struct) + 7 (loop)
    CONV_FILTERS = 256
    GRU_HIDDEN = 384  # Per direction
    GRU_BIDIRECTIONAL = True
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 240
    LEARNING_RATE = 1e-3
    MAX_GRAD_NORM = 1.0
    SEED = 42

    # Caching
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")


# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# ==================================================================================
# UTILS & DATA PROCESSING
# ==================================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def get_structure_adj(structure, seq_len):
    # Returns an array where adj[i] = j if i pairs with j, else 0 (masked later)
    # Also returns a mask where mask[i] = 1 if paired, 0 if unpaired
    adj = np.zeros(seq_len, dtype=np.int32)
    mask = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return adj, mask


def process_data(df, mode="train"):
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    N = len(df)
    L = Config.SEQ_LEN

    X = np.zeros((N, L, 14), dtype=np.float32)
    adj = np.zeros((N, L), dtype=np.int32)
    mask = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        seq = sequences[i]
        struct = structures[i]
        loop = loops[i]

        # One-hot encoding
        for j, char in enumerate(seq):
            if char in seq_map:
                X[i, j, seq_map[char]] = 1.0

        for j, char in enumerate(struct):
            if char in struct_map:
                X[i, j, 4 + struct_map[char]] = 1.0

        for j, char in enumerate(loop):
            if char in loop_map:
                X[i, j, 7 + loop_map[char]] = 1.0

        # Adjacency
        a, m = get_structure_adj(struct, L)
        adj[i] = a
        mask[i] = m

    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros((N, Config.SEQ_SCORED, 5), dtype=np.float32)
        for t_idx, col in enumerate(Config.TARGET_COLS):
            # Flatten the series of lists and stack
            vals = np.vstack(df[col].values)
            targets[:, :, t_idx] = vals

    return {"X": X, "adj": adj, "mask": mask, "ids": ids}, targets


def get_dataset(mode="train", load_cached_data=True):
    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache...")
        data = np.load(cache_path, allow_pickle=True)
        inputs = {k: data[k] for k in ["X", "adj", "mask", "ids"]}
        targets = data["targets"] if "targets" in data else None
        if mode != "test":
            return inputs, targets
        else:
            return inputs

    print(f"Processing {mode} data from scratch...")
    meta_path = os.path.join(Config.METADATA_DIR, f"{mode}.parquet")
    df = pd.read_parquet(meta_path)

    inputs, targets = process_data(df, mode)

    save_dict = {**inputs}
    if targets is not None:
        save_dict["targets"] = targets
    np.savez(cache_path, **save_dict)

    if mode != "test":
        return inputs, targets
    else:
        return inputs


class RNADataset(Dataset):
    def __init__(self, inputs, targets=None):
        self.X = inputs["X"]
        self.adj = inputs["adj"]
        self.mask = inputs["mask"]
        self.targets = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        adj = torch.tensor(self.adj[idx], dtype=torch.long)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)

        item = {"X": x, "adj": adj, "mask": mask}

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return item, y
        return item


# ==================================================================================
# MODEL
# ==================================================================================


class InteractionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message components
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Gate components (Wide Stabilized MLP)
        # Input is [h_i; h_j] -> 2 * hidden_dim
        self.gate_in = nn.Linear(hidden_dim * 2, hidden_dim)  # Project to wide (768)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj, mask):
        # h: (B, L, H)
        # adj: (B, L) indices of pairs
        # mask: (B, L) 1 if paired, 0 if unpaired

        B, L, H = h.shape

        # 1. Gather h_j
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(h, 1, adj_expanded)  # (B, L, H)

        # 2. Input Zero-Masking
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask_expanded

        # 3. GLU Message (Bias-Refined)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # 4. Wide Stabilized MLP Gate
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)
        z_raw = self.gate_in(cat_input)  # (B, L, H)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)
        g_ij = torch.sigmoid(self.gate_out(z_act))

        # 5. Injection
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNA_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stem
        self.conv = nn.Conv1d(
            config.INPUT_CHANNELS, config.CONV_FILTERS, kernel_size=3, padding=1
        )
        self.act = nn.GELU()

        # Backbone
        self.blocks = nn.ModuleList()
        gru_input_dim = config.CONV_FILTERS
        gru_hidden = config.GRU_HIDDEN
        gru_out_dim = gru_hidden * 2  # 768

        for i in range(config.NUM_LAYERS):
            # BiGRU
            gru = nn.GRU(
                input_size=gru_input_dim if i == 0 else gru_out_dim,
                hidden_size=gru_hidden,
                batch_first=True,
                bidirectional=True,
            )

            # Interaction (for first 3 layers, i.e., indices 0, 1, 2)
            interaction = None
            if i < config.NUM_LAYERS - 1:
                interaction = InteractionModule(gru_out_dim)

            self.blocks.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))

        self.dropout = nn.Dropout(config.DROPOUT)
        self.head = nn.Linear(gru_out_dim, 5)

    def forward(self, x, adj, mask):
        # x: (B, L, 14)
        # Permute for Conv1d: (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.act(self.conv(x))
        x = x.permute(0, 2, 1)  # (B, L, C)

        h = x

        for i, block in enumerate(self.blocks):
            h, _ = block["gru"](h)

            if block["interaction"] is not None:
                h = block["interaction"](h, adj, mask)

            if i < len(self.blocks) - 1:
                h = self.dropout(h)

        out = self.head(h)  # (B, L, 5)
        return out


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================


def mcrmse(y_true, y_pred):
    # y_true, y_pred: (B, L, C)
    # Flatten B and L to compute RMSE per column
    y_t = y_true.reshape(-1, y_true.shape[-1])
    y_p = y_pred.reshape(-1, y_pred.shape[-1])
    mse = torch.mean((y_t - y_p) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    return torch.mean(rmse)


def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_inputs, train_targets = get_dataset("train")
    val_inputs, val_targets = get_dataset("val")

    train_dataset = RNADataset(train_inputs, train_targets)
    val_dataset = RNADataset(val_inputs, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = RNA_Model(Config).to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_score = float("inf")
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            item, y = batch
            x = item["X"].to(device)
            adj = item["adj"].to(device)
            mask = item["mask"].to(device)
            y = y.to(device)  # (B, 68, 5)

            optimizer.zero_grad()

            # Forward
            pred = model(x, adj, mask)  # (B, 107, 5)

            # Slice to scored positions for loss calculation
            pred_sliced = pred[:, : Config.SEQ_SCORED, :]

            # Loss on all 5 columns (Multi-Task)
            loss = mcrmse(y, pred_sliced)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()

            train_loss_accum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_truths = []

        with torch.no_grad():
            for batch in val_loader:
                item, y = batch
                x = item["X"].to(device)
                adj = item["adj"].to(device)
                mask = item["mask"].to(device)
                y = y.to(device)

                pred = model(x, adj, mask)
                pred_sliced = pred[:, : Config.SEQ_SCORED, :]

                val_preds.append(pred_sliced.cpu())
                val_truths.append(y.cpu())

        val_preds = torch.cat(val_preds, dim=0)
        val_truths = torch.cat(val_truths, dim=0)

        # Calculate metric on scored columns only
        val_preds_scored = val_preds[:, :, scored_indices]
        val_truths_scored = val_truths[:, :, scored_indices]

        val_score = mcrmse(val_truths_scored, val_preds_scored).item()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )

    print(f"Best Val Score: {best_score}")


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_inputs = get_dataset("test")
    test_dataset = RNADataset(test_inputs, targets=None)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Model
    model = RNA_Model(Config).to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("No model found, skipping submission.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            item = batch
            x = item["X"].to(device)
            adj = item["adj"].to(device)
            mask = item["mask"].to(device)

            pred = model(x, adj, mask)  # (B, 107, 5)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 5)

    # Format Submission
    ids = test_inputs["ids"]
    submission_data = []

    for i, sample_id in enumerate(ids):
        pred_sample = all_preds[i]  # (107, 5)
        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            vals = pred_sample[j]
            submission_data.append([row_id] + list(vals))

    cols = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_data, columns=cols)
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run_pipeline():
    train_model()
    generate_submission()


# Execute pipeline
run_pipeline()
