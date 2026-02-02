import os
import ast
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Paths
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/val.csv"
    TEST_CSV = "./metadata/test.csv"
    SUBMISSION_SAMPLE = "./input/sample_submission.csv"
    CACHE_DIR = "./working/idea_6/"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data
    SEQ_LEN = 107
    PRED_LEN = 68

    # Model Hyperparameters
    HIDDEN_DIM = 192
    KERNEL_SIZE = 3
    DROPOUT = 0.1
    NUM_LAYERS = 12  # Dilations: 2^0 to 2^11

    # Training
    BATCH_SIZE = 64
    EPOCHS = 25
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature Dimensions
    # Sequence (4) + Structure (3) + Loop (7) + Partner Triplet (15)
    INPUT_DIM = 4 + 3 + 7 + 15
    OUTPUT_DIM = 5


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed(Config.SEED)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

# ==================================================================================
# DATA PROCESSING & FEATURE ENGINEERING
# ==================================================================================


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a mapping {index: paired_index}. Unpaired indices are not in the dict.
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


def one_hot(idx, num_classes):
    vec = np.zeros(num_classes, dtype=np.float32)
    if 0 <= idx < num_classes:
        vec[idx] = 1.0
    return vec


def process_data(csv_path, mode="train", load_cached_data=True):
    """
    Processes raw CSV data into numpy tensors.
    Features:
    1. Sequence One-Hot (A, G, C, U)
    2. Structure One-Hot ((, ), .)
    3. Loop Type One-Hot (S, M, I, B, H, E, X)
    4. Partner Triplet: For paired base i->j, one-hot of seq[j-1], seq[j], seq[j+1].
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_processed.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        data = np.load(cache_file)
        return data["inputs"], data["targets"], data["ids"]

    print(f"Processing {mode} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    inputs = np.zeros((num_samples, Config.SEQ_LEN, Config.INPUT_DIM), dtype=np.float32)
    # Targets: 5 channels. Initialize with zeros.
    targets = np.zeros(
        (num_samples, Config.SEQ_LEN, Config.OUTPUT_DIM), dtype=np.float32
    )
    ids = df["id"].values

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        pairs = get_structure_pairs(struct)

        # 1. Basic Features
        for i in range(Config.SEQ_LEN):
            # Sequence
            if i < len(seq):
                s_char = seq[i]
                if s_char in seq_map:
                    inputs[idx, i, seq_map[s_char]] = 1.0

            # Structure
            if i < len(struct):
                st_char = struct[i]
                if st_char in struct_map:
                    inputs[idx, i, 4 + struct_map[st_char]] = 1.0

            # Loop
            if i < len(loop):
                l_char = loop[i]
                if l_char in loop_map:
                    inputs[idx, i, 4 + 3 + loop_map[l_char]] = 1.0

            # 2. Partner Triplet Features
            # Offset: 4 + 3 + 7 = 14
            base_offset = 14
            if i in pairs:
                j = pairs[i]
                # Triplet: j-1, j, j+1
                triplet_indices = [j - 1, j, j + 1]
                for k, t_idx in enumerate(triplet_indices):
                    # 5 classes per position: A, G, C, U, Pad
                    # 0-3 for bases, 4 for pad
                    feat_idx = 4  # Default Pad
                    if 0 <= t_idx < len(seq):
                        t_char = seq[t_idx]
                        if t_char in seq_map:
                            feat_idx = seq_map[t_char]

                    # Set bit in the 15-dim vector (3 positions * 5 classes)
                    # Position k (0,1,2) * 5 classes + feat_idx
                    inputs[idx, i, base_offset + (k * 5) + feat_idx] = 1.0

        # 3. Targets
        if mode in ["train", "val"]:
            for t_i, col in enumerate(target_cols):
                try:
                    val_list = ast.literal_eval(row[col])
                    # Targets are usually length 68, but we pad to 107
                    length = len(val_list)
                    targets[idx, :length, t_i] = val_list
                except:
                    pass  # Should not happen with clean data

    np.savez(cache_file, inputs=inputs, targets=targets, ids=ids)
    print(f"Saved processed {mode} data to {cache_file}")
    return inputs, targets, ids


# ==================================================================================
# DATASET
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, targets, ids, mode="train"):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.ids[idx]


# ==================================================================================
# MODEL: Stacking-Aware Gated Hybrid Network
# ==================================================================================


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, dropout):
        super(GatedBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

        # Residual connection if channels match
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        out = filter_out * gate_out
        out = self.dropout(out)
        return out + self.residual(x)


class StackingAwareGatedNet(nn.Module):
    def __init__(self, config):
        super(StackingAwareGatedNet, self).__init__()

        self.embedding = nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM)

        # Gated Dilated TCN Backbone
        self.blocks = nn.ModuleList()
        for i in range(config.NUM_LAYERS):
            dilation = 2**i
            self.blocks.append(
                GatedBlock(
                    config.HIDDEN_DIM,
                    config.HIDDEN_DIM,
                    dilation,
                    config.KERNEL_SIZE,
                    config.DROPOUT,
                )
            )

        # Global Aggregation
        self.bigru = nn.GRU(
            config.HIDDEN_DIM,
            config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        self.head = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        x = self.embedding(x)

        # Permute for Conv1d: [Batch, Dim, Seq]
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)

        # Permute back for GRU: [Batch, Seq, Dim]
        x = x.permute(0, 2, 1)

        x, _ = self.bigru(x)

        out = self.head(x)
        return out


# ==================================================================================
# METRICS & LOSS
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    """
    Mean Columnwise RMSE.
    Only scores columns 0 (reactivity), 1 (deg_Mg_pH10), and 3 (deg_Mg_50C).
    """
    # pred, target: [Batch, Seq, 5]
    scored_indices = [0, 1, 3]

    loss = 0.0
    count = 0

    for idx in scored_indices:
        p = pred[:, :, idx]
        t = target[:, :, idx]

        mse = (p - t) ** 2

        if mask is not None:
            mse = mse * mask
            # Average over valid positions
            rmse = torch.sqrt(mse.sum() / (mask.sum() + 1e-8))
        else:
            rmse = torch.sqrt(mse.mean())

        loss += rmse
        count += 1

    return loss / count


# ==================================================================================
# TRAINING LOOP
# ==================================================================================


def train_model():
    # Load Data
    train_inputs, train_targets, train_ids = process_data(Config.TRAIN_CSV, "train")
    val_inputs, val_targets, val_ids = process_data(Config.VAL_CSV, "val")

    train_dataset = RNADataset(train_inputs, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_targets, val_ids)

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

    # Model
    model = StackingAwareGatedNet(Config).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    # Mask for scoring: only first 68 positions are valid for loss
    # Create a mask tensor [1, 107]
    mask = torch.zeros((1, Config.SEQ_LEN), device=Config.DEVICE)
    mask[:, : Config.PRED_LEN] = 1.0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()
            preds = model(inputs)

            loss = mcrmse_loss(preds, targets, mask)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)
                preds = model(inputs)
                loss = mcrmse_loss(preds, targets, mask)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MCRMSE: {avg_train_loss:.6f} | Val MCRMSE: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MCRMSE: {best_loss:.6f}")
    return best_model_path


# ==================================================================================
# INFERENCE
# ==================================================================================


def generate_submission(model_path):
    print("Generating submission...")
    test_inputs, _, test_ids = process_data(Config.TEST_CSV, "test")
    test_dataset = RNADataset(
        test_inputs, np.zeros((len(test_inputs), Config.SEQ_LEN, 5)), test_ids
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model = StackingAwareGatedNet(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    preds_list = []
    ids_list = []

    with torch.no_grad():
        for inputs, _, ids in test_loader:
            inputs = inputs.to(Config.DEVICE)
            output = model(inputs)  # [Batch, 107, 5]
            preds_list.append(output.cpu().numpy())
            ids_list.extend(ids)

    all_preds = np.concatenate(preds_list, axis=0)

    # Format for submission
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # [107, 5]
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos].tolist()
            submission_data.append([row_id] + row_vals)

    sub_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + target_cols)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# ==================================================================================
# MAIN
# ==================================================================================

if __name__ == "__main__":
    best_model = train_model()
    generate_submission(best_model)
