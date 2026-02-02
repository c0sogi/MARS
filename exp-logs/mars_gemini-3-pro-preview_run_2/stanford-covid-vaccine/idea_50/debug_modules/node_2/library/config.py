import os
import gc
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Model Hyperparameters
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_LAYERS = 6
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
    LATENT_DIM = 64

    FEEDBACK_INPUT_DIM = 5
    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_LAYERS = 4
    FEEDBACK_EMBED_DIM = 32

    RNN_HIDDEN = 64
    RNN_LAYERS = 1

    # Training
    BATCH_SIZE = 16  # Adjusted for memory safety
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    NUM_WORKERS = 2
    SEED = 42

    # Paths
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_50"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Columns
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


# Ensure reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_adj(structure, seq_len):
    """Parses dot-bracket structure to find pair indices."""
    pairs = np.full(seq_len, -1, dtype=np.int32)
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


def process_data(load_cached_data=True, debug_size=None):
    """
    Loads, processes, and caches data.
    Returns: (train_dict, val_dict, test_dict)
    Each dict contains: 'inputs', 'pair_indices', 'targets', 'ids'
    """
    cache_file = os.path.join(Config.CACHE_DIR, "data_cache_v1.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            {
                k: data[f"train_{k}"]
                for k in ["inputs", "pair_indices", "targets", "ids"]
            },
            {k: data[f"val_{k}"] for k in ["inputs", "pair_indices", "targets", "ids"]},
            {k: data[f"test_{k}"] for k in ["inputs", "pair_indices", "ids"]},
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    if debug_size:
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    # Helper for feature generation
    def get_features(df, is_test=False):
        # Dictionaries
        seq_map = {c: i for i, c in enumerate("AGCU")}
        struct_map = {c: i for i, c in enumerate(".()")}
        loop_map = {c: i for i, c in enumerate("SMIBHEX")}

        n_samples = len(df)
        seq_len = Config.SEQ_LEN

        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner Identity) = 18
        inputs = np.zeros((n_samples, seq_len, 18), dtype=np.float32)
        pair_indices = np.full((n_samples, seq_len), -1, dtype=np.int32)
        targets = (
            np.zeros((n_samples, seq_len, 5), dtype=np.float32) if not is_test else None
        )
        ids = df["id"].values

        for idx, row in df.iterrows():
            idx_loc = (
                idx if isinstance(idx, int) else 0
            )  # Reset index handling if needed
            # Re-align index for numpy array filling
            # Since we iterate df, we need a counter or use enumerate on df.itertuples
            pass

        # Correct iteration
        for i, row in enumerate(df.itertuples()):
            seq = row.sequence
            struct = row.structure
            loop = row.predicted_loop_type

            # 1. Basic One-Hot
            for j, char in enumerate(seq):
                if char in seq_map:
                    inputs[i, j, seq_map[char]] = 1.0

            for j, char in enumerate(struct):
                if char in struct_map:
                    inputs[i, j, 4 + struct_map[char]] = 1.0

            for j, char in enumerate(loop):
                if char in loop_map:
                    inputs[i, j, 7 + loop_map[char]] = 1.0

            # 2. Partner Info
            pairs = get_structure_adj(struct, seq_len)
            pair_indices[i] = pairs

            # Partner Identity
            for j, p_idx in enumerate(pairs):
                if p_idx != -1:
                    partner_char = seq[p_idx]
                    if partner_char in seq_map:
                        inputs[i, j, 14 + seq_map[partner_char]] = 1.0

            # 3. Targets
            if not is_test:
                # Parse stringified lists
                for t_idx, col in enumerate(Config.ALL_TARGET_COLS):
                    val_list = ast.literal_eval(getattr(row, col))
                    # Targets provided for first 68 bases
                    len_t = len(val_list)
                    targets[i, :len_t, t_idx] = val_list

        return inputs, pair_indices, targets, ids

    train_inputs, train_pairs, train_targets, train_ids = get_features(train_df)
    val_inputs, val_pairs, val_targets, val_ids = get_features(val_df)
    test_inputs, test_pairs, _, test_ids = get_features(test_df, is_test=True)

    # Save to cache
    np.savez_compressed(
        cache_file,
        train_inputs=train_inputs,
        train_pair_indices=train_pairs,
        train_targets=train_targets,
        train_ids=train_ids,
        val_inputs=val_inputs,
        val_pair_indices=val_pairs,
        val_targets=val_targets,
        val_ids=val_ids,
        test_inputs=test_inputs,
        test_pair_indices=test_pairs,
        test_ids=test_ids,
    )

    print("Data processed and cached.")
    return (
        {
            "inputs": train_inputs,
            "pair_indices": train_pairs,
            "targets": train_targets,
            "ids": train_ids,
        },
        {
            "inputs": val_inputs,
            "pair_indices": val_pairs,
            "targets": val_targets,
            "ids": val_ids,
        },
        {"inputs": test_inputs, "pair_indices": test_pairs, "ids": test_ids},
    )


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.ids = data_dict["ids"]
        self.is_test = is_test
        if not is_test:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (L, C) -> (C, L)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).transpose(0, 1)
        pairs = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        if self.is_test:
            return x, pairs, self.ids[idx]
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, pairs, y, self.ids[idx]


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.ln2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.conv_pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, C, L) -> LN expects (B, L, C)
        out = x.transpose(1, 2)
        out = self.ln1(out)
        out = self.act1(out).transpose(1, 2)

        out = self.conv_dilated(out)

        out_ln = out.transpose(1, 2)
        out_ln = self.ln2(out_ln)
        out_ln = self.act2(out_ln).transpose(1, 2)

        out = self.conv_pointwise(out_ln)
        out = self.drop(out)
        return out


class FeedbackTCN(nn.Module):
    def __init__(self, in_dim, growth_rate, layers, out_dim):
        super().__init__()
        self.embedding = nn.Conv1d(in_dim, growth_rate, kernel_size=1)
        self.blocks = nn.ModuleList()
        current_dim = growth_rate

        for i in range(layers):
            dilation = 2**i
            # Dense connection: input grows
            self.blocks.append(DilatedBlock(current_dim, growth_rate, dilation))
            current_dim += growth_rate

        self.out_proj = nn.Conv1d(current_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 5, L)
        features = [self.embedding(x)]

        for block in self.blocks:
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        final_concat = torch.cat(features, dim=1)
        return self.out_proj(final_concat)


class REID_FN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Input Embedding
        # 18 input channels
        self.input_embed = nn.Conv1d(18, Config.BACKBONE_GROWTH_RATE, kernel_size=1)

        # 2. Backbone (Dense Dilated TCN)
        self.backbone_blocks = nn.ModuleList()
        current_dim = Config.BACKBONE_GROWTH_RATE

        for d in Config.BACKBONE_DILATIONS:
            self.backbone_blocks.append(
                DilatedBlock(current_dim, Config.BACKBONE_GROWTH_RATE, d)
            )
            current_dim += Config.BACKBONE_GROWTH_RATE

        self.latent_proj = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)

        # 3. Pure-Feedback Module
        self.feedback_module = FeedbackTCN(
            in_dim=Config.FEEDBACK_INPUT_DIM,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            layers=Config.FEEDBACK_LAYERS,
            out_dim=Config.FEEDBACK_EMBED_DIM,
        )

        # 4. Interaction & Aggregation
        # Input to RNN: (Latent + Feedback) * 2 (Self + Partner)
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_EMBED_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Head
        self.head = nn.Linear(Config.RNN_HIDDEN * 2, 5)

    def forward(self, x, pair_indices, prev_preds=None):
        """
        x: (B, 18, L)
        pair_indices: (B, L)
        prev_preds: (B, L, 5) or None
        """
        B, C, L = x.shape

        # --- 1. Backbone (Static) ---
        # Embed
        embed = self.input_embed(x)
        features = [embed]

        # Dense Blocks
        for block in self.backbone_blocks:
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Project to Z
        z = self.latent_proj(torch.cat(features, dim=1))  # (B, Latent, L)
        z = z.transpose(1, 2)  # (B, L, Latent)

        # --- 2. Feedback Processing ---
        if prev_preds is None:
            prev_preds = torch.zeros((B, L, 5), device=x.device)

        # Strict Masking: Only keep reactivity, deg_Mg_pH10, deg_Mg_50C
        # Indices in 5-vector: 0, 1, 3 are scored. 2 (deg_pH10) and 4 (deg_50C) are not.
        # Mask: [1, 1, 0, 1, 0]
        mask = torch.tensor([1, 1, 0, 1, 0], device=x.device).view(1, 1, 5)
        masked_preds = prev_preds * mask

        # Feedback Module expects (B, 5, L)
        fb_in = masked_preds.transpose(1, 2)
        e_fb = self.feedback_module(fb_in).transpose(1, 2)  # (B, L, FB_Dim)

        # --- 3. Interaction ---
        # Self Vector
        h_self = torch.cat([z, e_fb], dim=2)  # (B, L, Latent+FB)

        # Partner Vector Gathering
        # pair_indices is (B, L). -1 indicates no pair.
        # We need to gather from h_self.

        # Create batch indices
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)

        # Handle -1 by clamping to 0 temporarily, then masking
        valid_mask = pair_indices != -1
        safe_indices = pair_indices.clone()
        safe_indices[~valid_mask] = 0

        h_partner = h_self[batch_idx, safe_indices]  # (B, L, Dim)
        h_partner = h_partner * valid_mask.unsqueeze(2).float()  # Zero out unpaired

        # Fusion
        h_combined = torch.cat([h_self, h_partner], dim=2)  # (B, L, Dim*2)

        # --- 4. Aggregation ---
        rnn_out, _ = self.rnn(h_combined)

        # --- 5. Head ---
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits


# ==================================================================================
# METRICS & LOSS
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    """
    Column-wise RMSE.
    pred, target: (B, L, 5)
    mask: (B, L) - 1 for valid positions, 0 for padding/unscored
    """
    # Scored columns indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]

    pred_scored = pred[:, :, scored_indices]
    target_scored = target[:, :, scored_indices]

    mse = (pred_scored - target_scored) ** 2

    if mask is not None:
        mask = mask.unsqueeze(2)  # (B, L, 1)
        mse = mse * mask
        # Count valid elements per column
        n_valid = mask.sum()
        # Avoid div by zero
        mse_col_mean = mse.sum(dim=(0, 1)) / (n_valid + 1e-8)
    else:
        mse_col_mean = mse.mean(dim=(0, 1))

    rmse_col = torch.sqrt(mse_col_mean)
    return rmse_col.mean()


# ==================================================================================
# TRAINING LOOP
# ==================================================================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, pairs, y, _ in loader:
        x, pairs, y = x.to(device), pairs.to(device), y.to(device)

        # Create mask for scored positions (0 to 67)
        # Sequence length is 107.
        mask = torch.zeros((x.shape[0], x.shape[2]), device=device)
        mask[:, : Config.PRED_LEN] = 1.0

        optimizer.zero_grad()

        # Pass 1: Zero Feedback
        pred1 = model(x, pairs, prev_preds=None)

        # Pass 2: Feedback from Pass 1 (Detached)
        pred2 = model(x, pairs, prev_preds=pred1.detach())

        # Loss
        loss1 = mcrmse_loss(pred1, y, mask)
        loss2 = mcrmse_loss(pred2, y, mask)

        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_mse = torch.zeros(3, device=device)
    total_count = 0

    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for x, pairs, y, _ in loader:
            x, pairs, y = x.to(device), pairs.to(device), y.to(device)

            # Pass 1
            pred1 = model(x, pairs, prev_preds=None)
            # Pass 2
            pred2 = model(x, pairs, prev_preds=pred1)

            # Select scored positions and columns
            pred_scored = pred2[:, : Config.PRED_LEN, scored_indices]
            target_scored = y[:, : Config.PRED_LEN, scored_indices]

            # Accumulate SSE
            se = (pred_scored - target_scored) ** 2
            total_mse += se.sum(dim=(0, 1))
            total_count += x.shape[0] * Config.PRED_LEN

    rmse_per_col = torch.sqrt(total_mse / total_count)
    return rmse_per_col.mean().item()


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_dict, val_dict, test_dict = process_data(load_cached_data=True)

    train_ds = RNADataset(train_dict)
    val_ds = RNADataset(val_dict)
    test_ds = RNADataset(test_dict, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model Setup
    model = REID_FN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_score = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        scheduler.step(val_score)

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score:.6f}")

    print(f"Training complete. Best Validation Score: {best_score:.6f}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds_list = []
    ids_list = []

    with torch.no_grad():
        for x, pairs, ids in test_loader:
            x, pairs = x.to(device), pairs.to(device)

            # Two pass inference
            pred1 = model(x, pairs, prev_preds=None)
            pred2 = model(x, pairs, prev_preds=pred1)

            preds_list.append(pred2.cpu().numpy())
            ids_list.extend(ids)

    preds_arr = np.concatenate(preds_list, axis=0)  # (N, L, 5)

    # Format Submission
    # Need one row per id_seqpos
    submission_rows = []
    for i, sample_id in enumerate(ids_list):
        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            # Values: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            vals = preds_arr[i, j]
            # Clip to valid range if necessary (usually not strictly required but good practice)
            # vals = np.clip(vals, -10, 10)

            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
    sub_df = pd.DataFrame(submission_rows, columns=cols)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_training()
