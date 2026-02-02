import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast
import random
import gc

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Paths
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"
    CACHE_DIR = "./working/idea_13/"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data Dimensions
    SEQ_LEN = 107
    SEQ_SCORED = 68
    INPUT_CHANNELS = 14  # 4 (Seq) + 3 (Struct) + 7 (Loop)

    # Targets
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Model Hyperparameters
    NUM_CHANNELS = 64
    BOTTLENECK_DIM = 32
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    STAGE1_DILATIONS = [1, 2, 4, 8, 16, 32]
    STAGE2_DILATIONS = [1, 2, 4, 8, 16, 32]

    # Training Settings
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 25
    PATIENCE = 5
    NUM_WORKERS = 2
    SEED = 42

    # Cache Files
    TRAIN_CACHE = "train_data_residual_v1.npz"
    VAL_CACHE = "val_data_residual_v1.npz"
    TEST_CACHE = "test_data_residual_v1.npz"


# ==================================================================================
# UTILITIES & PREPROCESSING
# ==================================================================================


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_array(x):
    """Parses stringified list to numpy array."""
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except:
            return np.zeros(0, dtype=np.float32)
    return np.array(x, dtype=np.float32)


def get_structure_indices(structure):
    """
    Returns an array where arr[i] is the index of the partner of base i.
    If unpaired, arr[i] = -1.
    """
    partner = np.full(len(structure), -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner[i] = j
                partner[j] = i
    return partner


def preprocess_data(df, is_test=False):
    """
    Generates inputs and targets.
    Input channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14.
    """
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    features = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    loss_mask = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    for i, row in df.iterrows():
        # Sequence
        seq = row["sequence"]
        for j, char in enumerate(seq):
            if char in seq_map:
                features[i, j, seq_map[char]] = 1.0

        # Structure
        struct = row["structure"]
        for j, char in enumerate(struct):
            if char in struct_map:
                features[i, j, 4 + struct_map[char]] = 1.0

        # Loop
        loop = row["predicted_loop_type"]
        for j, char in enumerate(loop):
            if char in loop_map:
                features[i, j, 7 + loop_map[char]] = 1.0

        # Pair Indices
        pair_indices[i] = get_structure_indices(struct)

        # Targets
        if not is_test:
            for k, col in enumerate(Config.ALL_TARGETS):
                val = parse_array(row[col])
                length = len(val)
                if length > 0:
                    targets[i, :length, k] = val
                    if col in Config.SCORED_COLS:
                        loss_mask[i, :length, k] = 1.0

    return features, pair_indices, targets, loss_mask


def load_or_process_data(mode="train", load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if mode == "train":
        meta_path = Config.TRAIN_METADATA
        cache_name = Config.TRAIN_CACHE
    elif mode == "val":
        meta_path = Config.VAL_METADATA
        cache_name = Config.VAL_CACHE
    else:
        meta_path = Config.TEST_METADATA
        cache_name = Config.TEST_CACHE

    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["features"],
            data["pair_indices"],
            data["targets"],
            data["loss_mask"],
            data["ids"],
        )

    print(f"Processing {mode} data from scratch...")
    df = pd.read_csv(meta_path)

    features, pair_indices, targets, loss_mask = preprocess_data(
        df, is_test=(mode == "test")
    )
    ids = df["id"].values

    np.savez_compressed(
        cache_path,
        features=features,
        pair_indices=pair_indices,
        targets=targets,
        loss_mask=loss_mask,
        ids=ids,
    )

    return features, pair_indices, targets, loss_mask, ids


# ==================================================================================
# DATASET
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, features, pair_indices, targets, loss_mask):
        self.features = features
        self.pair_indices = pair_indices
        self.targets = targets
        self.loss_mask = loss_mask

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.pair_indices[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.loss_mask[idx], dtype=torch.float32),
        )


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class DilatedDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=Config.KERNEL_SIZE,
            padding=dilation,
            dilation=dilation,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class StackingDenseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Stage 1
        self.stage1_blocks = nn.ModuleList()
        current_dim = Config.INPUT_CHANNELS
        for d in Config.STAGE1_DILATIONS:
            blk = DilatedDenseBlock(current_dim, Config.GROWTH_RATE, d, Config.DROPOUT)
            self.stage1_blocks.append(blk)
            current_dim += Config.GROWTH_RATE

        self.stage1_out_dim = current_dim

        # Compression
        self.compress = nn.Conv1d(self.stage1_out_dim, Config.HIDDEN_DIM, kernel_size=1)

        # Stage 2
        # Input is 2 * HIDDEN_DIM (Self + Paired)
        self.stage2_blocks = nn.ModuleList()
        current_dim = Config.HIDDEN_DIM * 2
        for d in Config.STAGE2_DILATIONS:
            blk = DilatedDenseBlock(current_dim, Config.GROWTH_RATE, d, Config.DROPOUT)
            self.stage2_blocks.append(blk)
            current_dim += Config.GROWTH_RATE

        self.stage2_out_dim = current_dim

        # Global Aggregation
        self.gru = nn.GRU(
            input_size=self.stage2_out_dim,
            hidden_size=self.stage2_out_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Head
        self.head = nn.Linear(self.stage2_out_dim, 5)

    def forward(self, x, pair_indices):
        # x: (B, L, 14) -> (B, 14, L)
        x = x.permute(0, 2, 1)
        B, _, L = x.shape

        # Stage 1
        out = x
        for blk in self.stage1_blocks:
            out = blk(out)

        # Compression: (B, 64, L)
        compressed = self.compress(out)

        # Pair Gather
        # Prepare for gathering: (B, L, 64)
        comp_trans = compressed.permute(0, 2, 1)

        # Pad with zero vector for unpaired indices (-1 maps to L)
        padding = torch.zeros(B, 1, Config.HIDDEN_DIM, device=x.device, dtype=x.dtype)
        comp_padded = torch.cat([comp_trans, padding], dim=1)  # (B, L+1, 64)

        # Adjust indices
        gather_idx = pair_indices.clone()
        gather_idx[gather_idx == -1] = L
        gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, Config.HIDDEN_DIM)

        # Gather paired features
        pair_features = torch.gather(comp_padded, 1, gather_idx_expanded)  # (B, L, 64)

        # Concatenate: (B, L, 128)
        combined = torch.cat([comp_trans, pair_features], dim=2)

        # Stage 2: (B, 128, L)
        out2 = combined.permute(0, 2, 1)
        for blk in self.stage2_blocks:
            out2 = blk(out2)

        # Global Aggregation: (B, L, C)
        out2 = out2.permute(0, 2, 1)
        gru_out, _ = self.gru(out2)

        # Head
        logits = self.head(gru_out)
        return logits


# ==================================================================================
# TRAINING & INFERENCE
# ==================================================================================


def mcrmse_loss(pred, target, mask):
    # pred, target, mask: (B, L, 5)
    mse = (pred - target) ** 2
    losses = []
    for k in range(5):
        m_k = mask[:, :, k]
        count = m_k.sum()
        if count > 0:
            rmse = torch.sqrt((mse[:, :, k] * m_k).sum() / count)
            losses.append(rmse)
    if len(losses) > 0:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=pred.device, requires_grad=True)


def validate(model, loader, device):
    model.eval()
    total_diff_sq = torch.zeros(5, device=device)
    total_count = torch.zeros(5, device=device)

    with torch.no_grad():
        for features, pair_indices, targets, mask in loader:
            features, pair_indices = features.to(device), pair_indices.to(device)
            targets, mask = targets.to(device), mask.to(device)

            preds = model(features, pair_indices)
            mse = (preds - targets) ** 2

            for k in range(5):
                m_k = mask[:, :, k]
                total_diff_sq[k] += (mse[:, :, k] * m_k).sum()
                total_count[k] += m_k.sum()

    scored_indices = [
        i for i, col in enumerate(Config.ALL_TARGETS) if col in Config.SCORED_COLS
    ]
    rmses = []
    for k in scored_indices:
        if total_count[k] > 0:
            rmses.append(torch.sqrt(total_diff_sq[k] / total_count[k]))

    if len(rmses) > 0:
        return torch.stack(rmses).mean().item()
    return 0.0


def train_model():
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, idx_train, y_train, mask_train, _ = load_or_process_data("train")
    X_val, idx_val, y_val, mask_val, _ = load_or_process_data("val")

    train_loader = DataLoader(
        RNADataset(X_train, idx_train, y_train, mask_train),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        RNADataset(X_val, idx_val, y_val, mask_val),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model = StackingDenseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for features, pair_indices, targets, mask in train_loader:
            features, pair_indices = features.to(device), pair_indices.to(device)
            targets, mask = targets.to(device), mask.to(device)

            optimizer.zero_grad()
            preds = model(features, pair_indices)
            loss = mcrmse_loss(preds, targets, mask)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

        val_loss = validate(model, val_loader, device)
        print(
            f"Epoch {epoch+1} | Train: {train_loss_accum/len(train_loader):.6f} | Val: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break


def generate_submission():
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, idx_test, _, _, ids = load_or_process_data("test")
    test_loader = DataLoader(
        RNADataset(
            X_test,
            idx_test,
            np.zeros((len(X_test), Config.SEQ_LEN, 5)),
            np.zeros((len(X_test), Config.SEQ_LEN, 5)),
        ),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model = StackingDenseNet().to(device)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for features, pair_indices, _, _ in test_loader:
            features, pair_indices = features.to(device), pair_indices.to(device)
            preds = model(features, pair_indices)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    submission_data = []
    for i, sample_id in enumerate(ids):
        for j in range(Config.SEQ_LEN):
            submission_data.append([f"{sample_id}_{j}"] + all_preds[i, j].tolist())

    sub_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + Config.ALL_TARGETS)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    train_model()
    generate_submission()


main()
