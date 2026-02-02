import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================================================
# CONFIGURATION
# =========================================================================================


class Config:
    # Data Paths
    TRAIN_PATH = "./metadata/train.parquet"
    VAL_PATH = "./metadata/val.parquet"
    TEST_PATH = "./metadata/test.parquet"
    CACHE_DIR = "./working/idea_77/"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Model Hyperparameters
    SEQ_LEN = 107
    PRED_LEN = 68
    INPUT_CHANNELS = 14  # 4 (seq) + 3 (struct) + 7 (loop)
    HIDDEN_DIM = 384  # Per direction, total 768
    NUM_LAYERS = 4
    DROPOUT = 0.1
    KERNEL_SIZE = 3

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 5

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    SEED = 2024


# =========================================================================================
# DATA PROCESSING & DATASET
# =========================================================================================


def parse_structure(structure_str):
    """Parses dot-bracket structure to find pair indices."""
    stack = []
    indices = np.full(len(structure_str), -1, dtype=np.int32)
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
    return indices


def process_data(load_cached_data=True, mode="train"):
    """
    Processes dataframe into tensors with caching.
    Args:
        load_cached_data (bool): Whether to load from cache if available.
        mode (str): 'train', 'val', or 'test'.
    Returns:
        dict: Dictionary containing processed tensors.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        loaded = np.load(cache_file)
        data = {k: torch.from_numpy(v) for k, v in loaded.items()}
        return data

    print(f"Processing data for {mode}...")

    # Load source dataframe
    if mode == "train":
        df = pd.read_parquet(Config.TRAIN_PATH)
    elif mode == "val":
        df = pd.read_parquet(Config.VAL_PATH)
    else:
        df = pd.read_parquet(Config.TEST_PATH)

    # Dictionaries for one-hot encoding
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    N = len(df)
    L = Config.SEQ_LEN

    # Features
    sequence_enc = np.zeros((N, L, 4), dtype=np.float32)
    structure_enc = np.zeros((N, L, 3), dtype=np.float32)
    loop_enc = np.zeros((N, L, 7), dtype=np.float32)

    # Adjacency (Pair indices)
    pair_indices = np.zeros((N, L), dtype=np.int64)
    pair_mask = np.zeros((N, L, 1), dtype=np.float32)

    # Targets
    targets = None
    if mode in ["train", "val"]:
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        targets = np.zeros((N, Config.PRED_LEN, 5), dtype=np.float32)

    for i, row in df.iterrows():
        # Sequence
        for j, char in enumerate(row["sequence"]):
            if char in seq_map:
                sequence_enc[i, j, seq_map[char]] = 1.0

        # Structure
        struct_str = row["structure"]
        for j, char in enumerate(struct_str):
            if char in struct_map:
                structure_enc[i, j, struct_map[char]] = 1.0

        # Loop Type
        for j, char in enumerate(row["predicted_loop_type"]):
            if char in loop_map:
                loop_enc[i, j, loop_map[char]] = 1.0

        # Pairs
        pairs = parse_structure(struct_str)
        # If unpaired (-1), set to 0 (dummy index), but mask is 0.
        p_idx = pairs.copy()
        p_idx[p_idx == -1] = 0
        pair_indices[i, :] = p_idx
        pair_mask[i, pairs != -1, 0] = 1.0

        # Targets
        if targets is not None:
            for k, col in enumerate(target_cols):
                val = row[col]
                if isinstance(val, (list, np.ndarray)):
                    targets[i, :, k] = np.array(val, dtype=np.float32)

    # Concatenate inputs: (N, L, 14)
    inputs = np.concatenate([sequence_enc, structure_enc, loop_enc], axis=2)

    data_dict = {"inputs": inputs, "pair_indices": pair_indices, "pair_mask": pair_mask}

    if targets is not None:
        data_dict["targets"] = targets

    # Save to cache
    np.savez(cache_file, **data_dict)

    return {k: torch.from_numpy(v) for k, v in data_dict.items()}


class RNADataset(Dataset):
    def __init__(self, data_dict, ids=None):
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_mask = data_dict["pair_mask"]
        self.targets = data_dict.get("targets")
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            "inputs": self.inputs[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_mask": self.pair_mask[idx],
        }
        if self.targets is not None:
            item["targets"] = self.targets[idx]
        if self.ids is not None:
            item["id"] = self.ids[idx]
        return item


# =========================================================================================
# MODEL ARCHITECTURE
# =========================================================================================


class StabilizedInteractionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message components
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Wide Gate components
        self.W_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm_gate = nn.LayerNorm(hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

        # Output Norm
        self.layer_norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_mask):
        """
        h: (B, L, H)
        pair_indices: (B, L)
        pair_mask: (B, L, 1)
        """
        B, L, H = h.shape

        # 1. Gather h_j
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, H)  # (B, L, H)
        h_j = torch.gather(h, 1, idx)  # (B, L, H)

        # 2. Mask Unpaired (Force unpaired h_j to 0)
        h_j = h_j * pair_mask

        # 3. GLU Message (Bias-Refined for unpaired)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # 4. Wide Stabilized MLP Gate
        cat_input = torch.cat([h, h_j], dim=-1)
        z_raw = self.W_in(cat_input)
        z_norm = self.layer_norm_gate(z_raw)
        z_act = F.gelu(z_norm)
        g_ij = torch.sigmoid(self.W_out(z_act))

        # 5. Injection
        h_struct = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.layer_norm_out(h_struct)

        return h_out


class RNAModel(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Stem: Conv1d projection
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.INPUT_CHANNELS,
                256,
                kernel_size=config.KERNEL_SIZE,
                padding=config.KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )
        self.stem_norm = nn.LayerNorm(256)

        # Backbone
        self.layers = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        input_size = 256
        hidden_total = config.HIDDEN_DIM * 2  # 768

        for i in range(config.NUM_LAYERS):
            # Layer 1 takes stem output (256), others take hidden_total (768)
            rnn_in = input_size if i == 0 else hidden_total

            self.layers.append(
                nn.GRU(
                    input_size=rnn_in,
                    hidden_size=config.HIDDEN_DIM,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            self.interactions.append(StabilizedInteractionModule(hidden_total))
            self.dropouts.append(nn.Dropout(config.DROPOUT))

        # Head
        self.head = nn.Linear(hidden_total, 5)

    def forward(self, x, pair_indices, pair_mask):
        # x: (B, L, 14) -> Permute for Conv: (B, 14, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        # Permute back: (B, L, 256)
        x = x.permute(0, 2, 1)
        x = self.stem_norm(x)

        h = x

        for i in range(self.config.NUM_LAYERS):
            # Vertical Residual Logic
            # Layer 1: h_new = BiGRU(stem)
            # Layer >1: h_new = h_prev + Dropout(BiGRU(h_prev))

            h_rnn, _ = self.layers[i](h)

            if i > 0:
                h_rnn = h + self.dropouts[i](h_rnn)

            # Interleaved Interaction
            h = self.interactions[i](h_rnn, pair_indices, pair_mask)

        out = self.head(h)  # (B, L, 5)
        return out


# =========================================================================================
# TRAINING & INFERENCE
# =========================================================================================


def mcrmse_loss(pred, target):
    """Calculates MCRMSE loss."""
    mse = torch.mean((pred - target) ** 2, dim=0)  # Mean over batch
    rmse = torch.sqrt(torch.mean(mse, dim=0))  # Mean over length
    return torch.mean(rmse)  # Mean over targets


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        preds = model(inputs, pair_indices, pair_mask)
        preds_sliced = preds[:, : Config.PRED_LEN, :]

        loss = mcrmse_loss(preds_sliced, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, pair_indices, pair_mask)
            preds_sliced = preds[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    mse = torch.mean((all_preds - all_targets) ** 2, dim=1)  # (N, 5)
    rmse = torch.sqrt(torch.mean(mse, dim=0))  # (5,)

    final_metric = torch.mean(rmse[scored_indices]).item()
    return final_metric


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)

            preds = model(inputs, pair_indices, pair_mask)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def main():
    # Set seed
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed(Config.SEED)

    # 1. Load Data
    train_data = process_data(mode="train")
    val_data = process_data(mode="val")
    test_data = process_data(mode="test")

    # Load test IDs for submission
    test_df = pd.read_parquet(Config.TEST_PATH)
    test_ids = test_df["id"].values

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data, ids=test_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model Setup
    model = RNAModel(Config).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 3. Training
    best_metric = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_metric = validate(model, val_loader, Config.DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_metric:.10f}"
        )

        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print("  New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 4. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    preds = predict(model, test_loader, Config.DEVICE)  # (N_test, 107, 5)

    # 5. Submission Generation
    print("Generating submission...")
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])
            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
