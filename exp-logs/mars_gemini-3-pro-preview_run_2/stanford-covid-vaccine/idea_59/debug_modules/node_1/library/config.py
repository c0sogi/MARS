import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ==================================================================================
# CONFIGURATION CONSTANTS
# ==================================================================================

TRAIN_JSON = "./input/train.json"
TEST_JSON = "./input/test.json"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_59"
SUBMISSION_PATH = "./submission/submission.csv"

# Structural Constants
SEQ_LEN = 107
PRED_LEN = 68
NUM_TARGETS = 5
SCORED_COLS_INDICES = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

# Training Hyperparameters
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 15  # Sufficient for convergence with this architecture
HIDDEN_DIM = 64
GROWTH_RATE = 64
FEEDBACK_GROWTH_RATE = 16
DROPOUT = 0.1
SEED = 42


# Ensure reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string and returns a mapping of paired indices.
    Returns:
        pairs: numpy array of shape (SEQ_LEN,) where pairs[i] = j if i is paired with j, else -1.
    """
    pairs = np.full(len(structure), -1, dtype=np.int32)
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


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    Generates One-Hot encodings and Partner Identity features.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_files = {
        "train": os.path.join(WORKING_DIR, "train_data_ds_rdn_v1.npz"),
        "val": os.path.join(WORKING_DIR, "val_data_ds_rdn_v1.npz"),
        "test": os.path.join(WORKING_DIR, "test_data_ds_rdn_v1.npz"),
    }

    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data...")
        data = {}
        for key, path in cache_files.items():
            data[key] = np.load(path, allow_pickle=True)
        return data

    print("Processing data from scratch...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Dictionaries for One-Hot Encoding
    seq_map = {c: i for i, c in enumerate("AGCU")}
    struct_map = {c: i for i, c in enumerate("().")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    def encode_sample(row, is_test=False):
        # 1. Sequence One-Hot (4 channels)
        seq_idx = [seq_map.get(c, 0) for c in row["sequence"]]
        seq_oh = np.eye(4)[seq_idx]

        # 2. Structure One-Hot (3 channels)
        struct_idx = [struct_map.get(c, 2) for c in row["structure"]]
        struct_oh = np.eye(3)[struct_idx]

        # 3. Loop Type One-Hot (7 channels)
        loop_idx = [loop_map.get(c, 6) for c in row["predicted_loop_type"]]
        loop_oh = np.eye(7)[loop_idx]

        # 4. Partner Identity (4 channels)
        pairs = get_structure_pairs(row["structure"])
        partner_oh = np.zeros((SEQ_LEN, 4), dtype=np.float32)
        for i, p_idx in enumerate(pairs):
            if p_idx != -1:
                # Get identity of the partner base
                partner_char = row["sequence"][p_idx]
                partner_oh[i] = np.eye(4)[seq_map.get(partner_char, 0)]

        # Concatenate all features: 4 + 3 + 7 + 4 = 18 channels
        inputs = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_oh], axis=1
        ).astype(np.float32)

        # Targets
        targets = np.zeros((SEQ_LEN, NUM_TARGETS), dtype=np.float32)
        if not is_test:
            # Parse stringified lists
            t_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
            for i, col in enumerate(t_cols):
                val_list = ast.literal_eval(row[col])
                # Pad to SEQ_LEN (targets are usually length 68)
                length = len(val_list)
                targets[:length, i] = val_list

        return inputs, targets, pairs, row["id"]

    def process_df(df, is_test=False):
        inputs_list, targets_list, pairs_list, ids_list = [], [], [], []
        for _, row in df.iterrows():
            inp, tar, par, sid = encode_sample(row, is_test)
            inputs_list.append(inp)
            targets_list.append(tar)
            pairs_list.append(par)
            ids_list.append(sid)
        return {
            "inputs": np.array(inputs_list),
            "targets": np.array(targets_list),
            "pairs": np.array(pairs_list),
            "ids": np.array(ids_list),
        }

    train_data = process_df(train_df)
    val_data = process_df(val_df)
    test_data = process_df(test_df, is_test=True)

    np.savez(cache_files["train"], **train_data)
    np.savez(cache_files["val"], **val_data)
    np.savez(cache_files["test"], **test_data)

    # Reload to return consistent format
    data = {}
    for key, path in cache_files.items():
        data[key] = np.load(path, allow_pickle=True)
    return data


class RNADataset(Dataset):
    def __init__(self, data):
        self.inputs = torch.from_numpy(data["inputs"]).float()
        self.targets = torch.from_numpy(data["targets"]).float()
        self.pairs = torch.from_numpy(data["pairs"]).long()
        self.ids = data["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.pairs[idx]


# ==================================================================================
# MODEL ARCHITECTURE: Dual-Stem Recurrent Dense Network (DS-RDN)
# ==================================================================================


class DenseDilatedBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding="same", dilation=dilation
        )
        self.ln1 = nn.LayerNorm(growth_rate)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(growth_rate)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        # Post-activation structure
        out = self.conv1(x)
        out = self.ln1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)

        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)

        # Dense connection: Concatenate input and output
        return torch.cat([x, out], dim=1)


class DualStemRDN(nn.Module):
    def __init__(self, in_channels=18, num_targets=5):
        super().__init__()

        # --- Static Branch ---
        self.static_stem = nn.Sequential(
            nn.Conv1d(in_channels, HIDDEN_DIM, kernel_size=3, padding="same"),
            nn.LayerNorm(HIDDEN_DIM),
            nn.SiLU(),
        )

        # Static Backbone (Dense TCN)
        self.static_blocks = nn.ModuleList()
        curr_dim = HIDDEN_DIM
        dilations = [1, 2, 4, 8, 16, 32]
        for d in dilations:
            blk = DenseDilatedBlock(curr_dim, GROWTH_RATE, dilation=d, dropout=DROPOUT)
            self.static_blocks.append(blk)
            curr_dim += GROWTH_RATE

        self.static_proj = nn.Conv1d(curr_dim, HIDDEN_DIM, kernel_size=1)

        # --- Feedback Branch ---
        # Feedback Stem: Processes recycled targets
        self.feedback_stem = nn.Sequential(
            nn.Conv1d(num_targets, 32, kernel_size=3, padding="same"),
            nn.LayerNorm(32),
            nn.SiLU(),
        )

        # Feedback Backbone (Lightweight Dense TCN)
        self.feedback_blocks = nn.ModuleList()
        curr_dim_fb = 32
        for d in [1, 2, 4, 8]:
            blk = DenseDilatedBlock(
                curr_dim_fb, FEEDBACK_GROWTH_RATE, dilation=d, dropout=DROPOUT
            )
            self.feedback_blocks.append(blk)
            curr_dim_fb += FEEDBACK_GROWTH_RATE

        self.feedback_proj = nn.Conv1d(curr_dim_fb, 32, kernel_size=1)

        # --- Interaction & Aggregation ---
        # Input to RNN: (Static(64) + Feedback(32)) * 2 (Self + Partner) = 192
        self.rnn = nn.GRU(
            input_size=(HIDDEN_DIM + 32) * 2,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(HIDDEN_DIM * 2, num_targets)

    def forward_static(self, x):
        # x: [B, C, L]
        # Static Stem
        feat = self.static_stem(x.transpose(1, 2)).transpose(
            1, 2
        )  # LN expects [B, L, C]

        # Static Backbone
        for blk in self.static_blocks:
            feat = blk(feat)

        # Projection
        z = self.static_proj(feat)  # [B, 64, L]
        return z

    def forward_feedback(self, y_prev):
        # y_prev: [B, 5, L]
        # Mask unscored channels (indices 2 and 4: deg_pH10, deg_50C)
        # We keep indices 0, 1, 3. Zero out 2, 4.
        mask = torch.tensor([1, 1, 0, 1, 0], device=y_prev.device).view(1, 5, 1)
        y_masked = y_prev * mask

        feat = self.feedback_stem(y_masked.transpose(1, 2)).transpose(1, 2)

        for blk in self.feedback_blocks:
            feat = blk(feat)

        e_fb = self.feedback_proj(feat)  # [B, 32, L]
        return e_fb

    def forward_interaction(self, z, e_fb, pairs):
        # z: [B, 64, L], e_fb: [B, 32, L]
        # pairs: [B, L]

        # Concatenate self features
        self_feat = torch.cat([z, e_fb], dim=1)  # [B, 96, L]

        # Gather partner features
        B, C, L = self_feat.shape
        # Create batch indices
        batch_idx = torch.arange(B, device=z.device).view(B, 1).expand(B, L)

        # Handle unpaired bases (-1)
        # Replace -1 with 0 for gathering, then mask result
        valid_mask = (pairs != -1).unsqueeze(1)  # [B, 1, L]
        safe_pairs = pairs.clone()
        safe_pairs[pairs == -1] = 0

        # Gather: [B, C, L]
        # We need to gather from dimension 2 based on indices in safe_pairs
        # View as [B, L, C] for easier gathering? No, gather works on dim.
        # Let's use advanced indexing
        partner_feat = self_feat[batch_idx, :, safe_pairs].transpose(1, 2)  # [B, C, L]

        # Apply mask to zero out unpaired partners
        partner_feat = partner_feat * valid_mask

        # Concatenate Self and Partner: [B, 192, L]
        combined = torch.cat([self_feat, partner_feat], dim=1)

        # RNN
        combined = combined.transpose(1, 2)  # [B, L, 192]
        rnn_out, _ = self.rnn(combined)  # [B, L, 128]

        # Head
        preds = self.head(rnn_out)  # [B, L, 5]
        return preds.transpose(1, 2)  # [B, 5, L]

    def forward(self, x, pairs, y_prev=None):
        if y_prev is None:
            y_prev = torch.zeros((x.shape[0], NUM_TARGETS, x.shape[2]), device=x.device)

        z = self.forward_static(x)
        e_fb = self.forward_feedback(y_prev)
        preds = self.forward_interaction(z, e_fb, pairs)
        return preds, z  # Return z to reuse in recycling


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    # pred, target: [B, 5, L]
    # mask: [B, L] or None

    # Only scored columns: 0, 1, 3
    scored_indices = torch.tensor(SCORED_COLS_INDICES, device=pred.device)
    pred_scored = torch.index_select(pred, 1, scored_indices)
    target_scored = torch.index_select(target, 1, scored_indices)

    mse = (pred_scored - target_scored) ** 2

    if mask is not None:
        # Expand mask to [B, 3, L]
        mask_exp = mask.unsqueeze(1).expand_as(mse)
        mse = mse * mask_exp
        # Sum over L, divide by count of valid positions
        # Per column RMSE
        loss_per_col = torch.sqrt(
            mse.sum(dim=(0, 2)) / (mask_exp.sum(dim=(0, 2)) + 1e-6)
        )
    else:
        loss_per_col = torch.sqrt(mse.mean(dim=(0, 2)))

    return loss_per_col.mean()


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = process_data(load_cached_data=True)
    train_dataset = RNADataset(data["train"])
    val_dataset = RNADataset(data["val"])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = DualStemRDN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    # Mask for loss calculation (only first 68 positions)
    # We construct it dynamically in the loop based on batch size

    for epoch in range(EPOCHS):
        model.train()
        train_loss_accum = 0

        for inputs, targets, pairs in train_loader:
            inputs, targets, pairs = (
                inputs.to(device),
                targets.to(device),
                pairs.to(device),
            )
            B, _, L = inputs.shape

            # Create mask for scored positions (0 to 67)
            mask = torch.zeros((B, L), device=device)
            mask[:, :PRED_LEN] = 1.0

            optimizer.zero_grad()

            # --- Recycling Loop ---
            # Pass 1: Initial feedback is zero
            y_prev = torch.zeros((B, NUM_TARGETS, L), device=device)

            # We can optimize by computing static Z once
            z = model.forward_static(inputs)

            # Pass 1
            e_fb_1 = model.forward_feedback(y_prev)
            preds_1 = model.forward_interaction(z, e_fb_1, pairs)

            # Pass 2
            # Detach gradients from Pass 1 predictions to stop gradient explosion/loops
            y_prev_2 = preds_1.detach()
            e_fb_2 = model.forward_feedback(y_prev_2)
            preds_2 = model.forward_interaction(z, e_fb_2, pairs)

            # Loss
            loss1 = mcrmse_loss(preds_1, targets, mask)
            loss2 = mcrmse_loss(preds_2, targets, mask)

            total_loss = loss2 + 0.5 * loss1

            total_loss.backward()
            optimizer.step()

            train_loss_accum += total_loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_loss_accum = 0

        # For correct global RMSE, we should accumulate SSE and counts, but batch avg is approx fine for monitoring
        with torch.no_grad():
            for inputs, targets, pairs in val_loader:
                inputs, targets, pairs = (
                    inputs.to(device),
                    targets.to(device),
                    pairs.to(device),
                )
                B, _, L = inputs.shape
                mask = torch.zeros((B, L), device=device)
                mask[:, :PRED_LEN] = 1.0

                # Inference: Pass 1 -> Pass 2
                z = model.forward_static(inputs)

                # Pass 1
                y_prev = torch.zeros((B, NUM_TARGETS, L), device=device)
                e_fb_1 = model.forward_feedback(y_prev)
                preds_1 = model.forward_interaction(z, e_fb_1, pairs)

                # Pass 2
                e_fb_2 = model.forward_feedback(preds_1)  # No detach needed in eval
                preds_2 = model.forward_interaction(z, e_fb_2, pairs)

                loss = mcrmse_loss(preds_2, targets, mask)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(WORKING_DIR, "best_model.pth"))

    print(f"Best Validation Loss: {best_val_loss:.6f}")
    return model


def generate_submission(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = DualStemRDN().to(device)
        model.load_state_dict(torch.load(os.path.join(WORKING_DIR, "best_model.pth")))

    model.eval()

    data = process_data(load_cached_data=True)
    test_dataset = RNADataset(data["test"])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds_list = []
    ids_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, _, pairs in test_loader:
            inputs, pairs = inputs.to(device), pairs.to(device)
            B, _, L = inputs.shape

            # Inference Recycling
            z = model.forward_static(inputs)

            # Pass 1
            y_prev = torch.zeros((B, NUM_TARGETS, L), device=device)
            e_fb_1 = model.forward_feedback(y_prev)
            preds_1 = model.forward_interaction(z, e_fb_1, pairs)

            # Pass 2
            e_fb_2 = model.forward_feedback(preds_1)
            preds_2 = model.forward_interaction(z, e_fb_2, pairs)

            # Move to CPU
            preds_np = preds_2.transpose(1, 2).cpu().numpy()  # [B, L, 5]
            preds_list.append(preds_np)

            # Get IDs for this batch (reconstruct from dataset or loader if possible, but dataset has them)
            # Since loader is sequential and no shuffle, we can just iterate dataset ids
            pass

    # Concatenate all predictions
    all_preds = np.concatenate(preds_list, axis=0)
    all_ids = data["test"]["ids"]

    # Prepare submission dataframe
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # [107, 5]
        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]
            row_dict = {"id_seqpos": row_id}
            for j, col in enumerate(cols):
                row_dict[col] = float(row_vals[j])
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


# ==================================================================================
# EXECUTION ENTRY POINT (Simulated Main)
# ==================================================================================


def run_pipeline():
    train_model()
    generate_submission()


# To run:
# from config import run_pipeline
# run_pipeline()
