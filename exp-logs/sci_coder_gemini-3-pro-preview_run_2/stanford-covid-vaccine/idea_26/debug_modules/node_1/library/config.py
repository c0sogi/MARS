import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================================================================================
# CONFIGURATION CONSTANTS
# ==================================================================================

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_26"
CACHE_DIR = WORKING_DIR

TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# Sequence Parameters
SEQ_LEN = 107
PRED_LEN = 68

# Targets
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SCORED_INDICES = [0, 1, 3]  # Indices corresponding to SCORED_COLS in TARGET_COLS

# Model Hyperparameters
# Input: 4(Seq) + 3(Struct) + 7(Loop) + 4(Partner) + 5(Recycled) = 23
INPUT_DIM = 23
GROWTH_RATE = 64
DROPOUT = 0.1
HIDDEN_DIM = 64
N_LAYERS = 6  # Dilations: 1, 2, 4, 8, 16, 32
KERNEL_SIZE = 3

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find base pairs.
    Returns a mapping {index: partner_index}. Unpaired bases are not in the map.
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


def process_data(mode="train", load_cached_data=True):
    """
    Loads and processes data, using caching.
    Generates static features (Seq, Struct, Loop, Partner) and parses targets.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{mode}_data_recurrent_v1.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        return np.load(cache_path, allow_pickle=True)

    print(f"Processing {mode} data from scratch...")

    if mode == "train":
        df = pd.read_csv(TRAIN_CSV)
    elif mode == "val":
        df = pd.read_csv(VAL_CSV)
    else:
        df = pd.read_csv(TEST_CSV)

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    n_samples = len(df)

    # Static features: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18 channels
    # We construct this as a dense array.
    X_static = np.zeros((n_samples, SEQ_LEN, 18), dtype=np.float32)

    # Partner indices for gathering operations in the model
    # Initialize with -1 (unpaired)
    partner_indices = np.full((n_samples, SEQ_LEN), -1, dtype=np.int32)

    # Targets
    y = np.zeros((n_samples, SEQ_LEN, 5), dtype=np.float32)

    ids = df["id"].values

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        pairs = get_structure_pairs(struct)

        for i in range(SEQ_LEN):
            if i < len(seq):
                # 1. Sequence One-Hot (0-3)
                X_static[idx, i, seq_map.get(seq[i], 0)] = 1.0

                # 2. Structure One-Hot (4-6)
                X_static[idx, i, 4 + struct_map.get(struct[i], 2)] = 1.0

                # 3. Loop One-Hot (7-13)
                X_static[idx, i, 7 + loop_map.get(loop[i], 6)] = 1.0

                # 4. Partner Identity (14-17) & Index
                if i in pairs:
                    j = pairs[i]
                    partner_indices[idx, i] = j
                    # Inject partner's base identity
                    X_static[idx, i, 14 + seq_map.get(seq[j], 0)] = 1.0
                else:
                    # Unpaired: partner_indices remains -1, partner identity remains 0
                    pass

        # Parse Targets (only for train/val)
        if mode in ["train", "val"]:
            for t_i, col in enumerate(TARGET_COLS):
                try:
                    val_str = row[col]
                    if isinstance(val_str, str):
                        val_list = ast.literal_eval(val_str)
                    else:
                        val_list = val_str

                    # Targets are usually length 68
                    length = len(val_list)
                    y[idx, :length, t_i] = val_list
                except Exception:
                    pass

    # Save to cache
    np.savez(
        cache_path, X_static=X_static, partner_indices=partner_indices, y=y, ids=ids
    )
    print(f"Saved processed data to {cache_path}")

    return {
        "X_static": X_static,
        "partner_indices": partner_indices,
        "y": y,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.X_static = torch.from_numpy(data_dict["X_static"]).float()
        self.partner_indices = torch.from_numpy(data_dict["partner_indices"]).long()
        self.y = torch.from_numpy(data_dict["y"]).float()
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.X_static)

    def __getitem__(self, idx):
        return self.X_static[idx], self.partner_indices[idx], self.y[idx]


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class DenseDilatedBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size, padding=dilation, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class RecurrentDenseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = INPUT_DIM
        self.growth_rate = GROWTH_RATE
        self.hidden_dim = HIDDEN_DIM

        # Dense Backbone
        # Input: 23 channels (18 static + 5 recycled)
        self.blocks = nn.ModuleList()
        current_dim = self.input_dim
        dilations = [1, 2, 4, 8, 16, 32]

        for d in dilations:
            self.blocks.append(
                DenseDilatedBlock(
                    current_dim, self.growth_rate, KERNEL_SIZE, d, DROPOUT
                )
            )
            current_dim += self.growth_rate

        self.backbone_out_dim = current_dim

        # Latent Interaction
        self.compress = nn.Conv1d(self.backbone_out_dim, 32, 1)

        # BiGRU
        # Input: 32 (local) + 32 (partner) = 64
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        self.head = nn.Linear(self.hidden_dim * 2, 5)

    def forward(self, x_static, x_recycled, partner_indices):
        """
        Args:
            x_static: [Batch, Length, 18]
            x_recycled: [Batch, Length, 5]
            partner_indices: [Batch, Length]
        """
        # 1. Input Construction
        x = torch.cat([x_static, x_recycled], dim=2)  # [B, L, 23]
        x = x.permute(0, 2, 1)  # [B, 23, L]

        # 2. Dense Backbone
        features = [x]
        for block in self.blocks:
            # Dense connection: concat all previous features
            out = block(torch.cat(features, dim=1))
            features.append(out)

        backbone_out = torch.cat(features, dim=1)  # [B, Total_Dim, L]

        # 3. Latent Interaction
        compressed = self.compress(backbone_out)  # [B, 32, L]
        compressed = compressed.permute(0, 2, 1)  # [B, L, 32]

        # Gather partner features
        B, L, C = compressed.shape
        flat_compressed = compressed.reshape(B * L, C)

        # Calculate flat indices: batch_offset + partner_index
        batch_offsets = torch.arange(B, device=x.device).unsqueeze(1) * L
        flat_indices = partner_indices + batch_offsets
        flat_indices = flat_indices.view(-1)  # [B*L]

        # Mask for unpaired bases (-1)
        mask = partner_indices.view(-1) != -1

        # Safe gather
        safe_indices = flat_indices.clone()
        safe_indices[~mask] = 0

        gathered = flat_compressed[safe_indices]  # [B*L, 32]
        gathered[~mask] = 0.0  # Zero out unpaired

        gathered = gathered.view(B, L, C)

        # Fuse
        interaction_out = torch.cat([compressed, gathered], dim=2)  # [B, L, 64]

        # 4. BiGRU
        gru_out, _ = self.gru(interaction_out)  # [B, L, 128]

        # 5. Head
        logits = self.head(gru_out)  # [B, L, 5]

        return logits


# ==================================================================================
# TRAINING & EVALUATION UTILS
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    """
    Mean Columnwise RMSE on scored columns only.
    """
    mse_list = []
    for idx in SCORED_INDICES:
        p = pred[:, :, idx]
        t = target[:, :, idx]

        diff_sq = (p - t) ** 2

        if mask is not None:
            diff_sq = diff_sq * mask
            count = mask.sum() + 1e-8
            mse = diff_sq.sum() / count
        else:
            mse = diff_sq.mean()

        mse_list.append(torch.sqrt(mse))

    return torch.mean(torch.stack(mse_list))


def train_model(debug=False):
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load Data
    train_data = process_data("train")
    val_data = process_data("val")

    if debug:
        for k in ["X_static", "partner_indices", "y"]:
            train_data[k] = train_data[k][:100]
            val_data[k] = val_data[k][:20]
        train_data["ids"] = train_data["ids"][:100]
        val_data["ids"] = val_data["ids"][:20]

    train_ds = RNADataset(train_data)
    val_ds = RNADataset(val_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RecurrentDenseNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Mask for scoring (first 68 positions)
    def get_mask(batch_size, device):
        m = torch.zeros((batch_size, SEQ_LEN), device=device)
        m[:, :PRED_LEN] = 1.0
        return m

    best_val_loss = float("inf")

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for x_static, partner_idx, y in train_loader:
            x_static = x_static.to(DEVICE)
            partner_idx = partner_idx.to(DEVICE)
            y = y.to(DEVICE)
            B = x_static.shape[0]

            # Pass 1: Zero recycling
            x_recycled_1 = torch.zeros((B, SEQ_LEN, 5), device=DEVICE)
            pred_1 = model(x_static, x_recycled_1, partner_idx)

            # Pass 2: Recycle predictions (with gradients)
            x_recycled_2 = pred_1
            pred_2 = model(x_static, x_recycled_2, partner_idx)

            mask = get_mask(B, DEVICE)
            loss = mcrmse_loss(pred_2, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for x_static, partner_idx, y in val_loader:
                x_static = x_static.to(DEVICE)
                partner_idx = partner_idx.to(DEVICE)
                y = y.to(DEVICE)
                B = x_static.shape[0]

                # Recurrent Inference
                x_recycled_1 = torch.zeros((B, SEQ_LEN, 5), device=DEVICE)
                pred_1 = model(x_static, x_recycled_1, partner_idx)

                x_recycled_2 = pred_1
                pred_2 = model(x_static, x_recycled_2, partner_idx)

                mask = get_mask(B, DEVICE)
                loss = mcrmse_loss(pred_2, y, mask)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(WORKING_DIR, "best_model.pth"))

    print(f"Best Val Loss: {best_val_loss:.6f}")


def generate_submission():
    # Load Test Data
    test_data = process_data("test")
    test_ds = RNADataset(test_data)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RecurrentDenseNetwork().to(DEVICE)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Model not found, skipping submission.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for x_static, partner_idx, _ in test_loader:
            x_static = x_static.to(DEVICE)
            partner_idx = partner_idx.to(DEVICE)
            B = x_static.shape[0]

            # Recurrent Inference
            x_recycled_1 = torch.zeros((B, SEQ_LEN, 5), device=DEVICE)
            pred_1 = model(x_static, x_recycled_1, partner_idx)

            x_recycled_2 = pred_1
            pred_2 = model(x_static, x_recycled_2, partner_idx)

            preds.append(pred_2.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # [N, 107, 5]
    all_ids = test_data["ids"]

    submission_rows = []

    for i, sample_id in enumerate(all_ids):
        sample_preds = preds[i]  # [107, 5]

        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {
                "id_seqpos": row_id,
                "reactivity": row_vals[0],
                "deg_Mg_pH10": row_vals[1],
                "deg_pH10": row_vals[2],
                "deg_Mg_50C": row_vals[3],
                "deg_50C": row_vals[4],
            }
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)
    cols = ["id_seqpos"] + TARGET_COLS
    sub_df = sub_df[cols]

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
