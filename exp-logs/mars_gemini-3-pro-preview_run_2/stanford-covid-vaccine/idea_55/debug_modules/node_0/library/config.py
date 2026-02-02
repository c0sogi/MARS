import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Hyperparameters
    BATCH_SIZE = 16  # Strictly enforced based on lessons
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    HIDDEN_DIM = 64
    NUM_LAYERS = 6
    DROPOUT = 0.1

    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68
    NUM_TARGETS = 5
    # Indices for: reactivity, deg_Mg_pH10, deg_Mg_50C
    SCORED_TARGETS = [0, 1, 3]

    # Paths
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_55"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Mappings
    BASE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    STRUCT_MAP = {".": 0, "(": 1, ")": 2}
    LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_adj(structure):
    """
    Parses dot-bracket structure to find partner indices.
    Returns array of length L where arr[i] is index of partner, or -1 if unpaired.
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


def one_hot(seq_arr, num_classes):
    """One-hot encodes an integer array."""
    res = np.zeros((len(seq_arr), num_classes), dtype=np.float32)
    for i, val in enumerate(seq_arr):
        if 0 <= val < num_classes:
            res[i, val] = 1.0
    return res


def process_dataframe(df, mode="train"):
    """
    Extracts features and targets from dataframe.
    Returns: X (N, L, 18), partners (N, L), y (N, L, 5) or None, ids
    """
    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    # --- Process Targets ---
    y_data = None
    if mode in ["train", "val"]:
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        num_samples = len(df)
        y_data = np.zeros((num_samples, Config.SEQ_LEN, 5), dtype=np.float32)

        for t_idx, col in enumerate(target_cols):
            # Parse stringified lists
            values = (
                df[col]
                .apply(
                    lambda x: (
                        np.array(ast.literal_eval(x), dtype=np.float32)
                        if isinstance(x, str)
                        else np.array(x, dtype=np.float32)
                    )
                )
                .values
            )
            for i, val in enumerate(values):
                length = min(len(val), Config.SEQ_LEN)
                y_data[i, :length, t_idx] = val[:length]

    # --- Process Features ---
    num_samples = len(df)
    # Channels: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    X_static = np.zeros((num_samples, Config.SEQ_LEN, 18), dtype=np.float32)
    partner_indices = np.full((num_samples, Config.SEQ_LEN), -1, dtype=int)

    for i in range(num_samples):
        seq = sequences[i]
        struc = structures[i]
        lp = loops[i]

        # Integer encoding
        seq_ints = [Config.BASE_MAP.get(c, 0) for c in seq]
        struc_ints = [Config.STRUCT_MAP.get(c, 0) for c in struc]
        loop_ints = [Config.LOOP_MAP.get(c, 0) for c in lp]

        # One-hot encoding
        oh_seq = one_hot(seq_ints, 4)
        oh_struc = one_hot(struc_ints, 3)
        oh_loop = one_hot(loop_ints, 7)

        # Partner Mapping
        p_idx = get_structure_adj(struc)
        partner_indices[i] = p_idx

        # Explicit Partner Identity
        oh_partner = np.zeros((Config.SEQ_LEN, 4), dtype=np.float32)
        valid_mask = p_idx != -1
        # For paired bases, copy the one-hot vector of the partner
        oh_partner[valid_mask] = oh_seq[p_idx[valid_mask]]

        # Concatenate all features
        X_static[i] = np.concatenate([oh_seq, oh_struc, oh_loop, oh_partner], axis=1)

    return X_static, partner_indices, y_data, ids


def get_data(mode="train", load_cached=True):
    """
    Loads data, using cache if available.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data_dda_rn.npz")

    if load_cached and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["X"],
                data["partners"],
                data["y"] if "y" in data else None,
                data["ids"],
            )
        except Exception as e:
            print(f"Cache load failed ({e}), reprocessing...")

    print(f"Processing {mode} data from scratch...")
    df_path = os.path.join(Config.METADATA_DIR, f"{mode}.csv")
    df = pd.read_csv(df_path)

    X, partners, y, ids = process_dataframe(df, mode)

    save_dict = {"X": X, "partners": partners, "ids": ids}
    if y is not None:
        save_dict["y"] = y

    np.savez(cache_path, **save_dict)
    return X, partners, y, ids


class RNADataset(Dataset):
    def __init__(self, X, partners, y=None):
        # X: (N, L, C) -> Permute to (N, C, L) for Conv1d
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.partners = torch.tensor(partners, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.partners[idx], self.y[idx]
        return self.X[idx], self.partners[idx]


# ==================================================================================
# MODEL ARCHITECTURE: Dual Direct-Access Recurrent Network (DDA-RN)
# ==================================================================================


class DirectAccessBlock(nn.Module):
    """
    Dilated Convolutional Block with Post-Activation structure.
    """

    def __init__(self, in_channels, hidden_dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.pw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, x):
        # x: (N, C_in, L)
        out = self.conv(x)

        # LayerNorm expects (N, L, C)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.pw(out)

        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)
        return out


class FeedbackEncoder(nn.Module):
    """
    Encodes recycled predictions with Direct Access to Raw Topology.
    """

    def __init__(self, hidden_dim=32):
        super().__init__()
        # Input: 5 (pred) + 10 (topo: 3 struct + 7 loop) = 15
        self.stem = nn.Conv1d(15, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        dilations = [1, 2, 4]
        current_dim = hidden_dim

        for d in dilations:
            # Input to block is concat(prev_out, raw_topo)
            in_dim = current_dim + 10
            self.blocks.append(DirectAccessBlock(in_dim, hidden_dim, d))

    def forward(self, pred, raw_topo):
        # pred: (N, 5, L)
        # raw_topo: (N, 10, L)
        x = torch.cat([pred, raw_topo], dim=1)
        x = self.stem(x)

        for block in self.blocks:
            inp = torch.cat([x, raw_topo], dim=1)
            x = block(inp)

        return x


class DDARNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.raw_dim = 18
        self.topo_indices = slice(4, 14)  # Indices for Struct(3) + Loop(7) in X
        self.hidden_dim = Config.HIDDEN_DIM
        self.fb_dim = 32

        # --- Backbone ---
        self.stem = nn.Conv1d(self.raw_dim, self.hidden_dim, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16, 32]

        # Direct Access Wiring: Input to block k is concat(all_prev_outputs, raw_input)
        current_in_dim = self.hidden_dim + self.raw_dim
        self.out_dims = [self.hidden_dim]  # Stem output

        for d in dilations:
            self.blocks.append(DirectAccessBlock(current_in_dim, self.hidden_dim, d))
            self.out_dims.append(self.hidden_dim)
            current_in_dim += self.hidden_dim  # Accumulate output width

        # Latent Projection (1x1 Conv)
        total_backbone_out = sum(self.out_dims)
        self.proj = nn.Conv1d(total_backbone_out, self.hidden_dim, kernel_size=1)

        # --- Feedback ---
        self.feedback = FeedbackEncoder(self.fb_dim)

        # --- Interaction & Aggregation ---
        # Input: Self(Hidden + FB) + Partner(Hidden + FB)
        rnn_input_dim = (self.hidden_dim + self.fb_dim) * 2
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Head ---
        self.head = nn.Linear(self.hidden_dim * 2, 5)

    def forward(self, x, partners, prev_pred=None):
        # x: (N, 18, L)
        # partners: (N, L)
        # prev_pred: (N, 5, L)

        B, C, L = x.shape
        raw_topo = x[:, self.topo_indices, :]  # (N, 10, L)

        # 1. Backbone Pass
        stem_out = self.stem(x)
        outputs = [stem_out]

        current_in = torch.cat([stem_out, x], dim=1)

        for block in self.blocks:
            out = block(current_in)
            outputs.append(out)
            current_in = torch.cat([current_in, out], dim=1)

        z_all = torch.cat(outputs, dim=1)
        z = self.proj(z_all)  # (N, 64, L)

        # 2. Feedback Pass
        if prev_pred is None:
            prev_pred = torch.zeros((B, 5, L), device=x.device, dtype=x.dtype)

        fb_emb = self.feedback(prev_pred, raw_topo)  # (N, 32, L)

        # 3. Interaction
        node_feat = torch.cat([z, fb_emb], dim=1)  # (N, 96, L)

        # Gather partner features
        p_idx = partners.clone()
        mask_unpaired = p_idx == -1
        p_idx[mask_unpaired] = 0  # Safe index for gather

        idx_expanded = p_idx.unsqueeze(1).expand(-1, node_feat.size(1), -1)
        partner_feat = torch.gather(node_feat, 2, idx_expanded)

        # Mask unpaired
        partner_feat = partner_feat * (~mask_unpaired.unsqueeze(1))

        # Concatenate
        combined = torch.cat([node_feat, partner_feat], dim=1)  # (N, 192, L)

        # 4. RNN & Head
        combined = combined.permute(0, 2, 1)  # (N, L, C)
        rnn_out, _ = self.rnn(combined)
        logits = self.head(rnn_out)  # (N, L, 5)

        return logits.permute(0, 2, 1)  # (N, 5, L)


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    """
    Calculates MCRMSE on the scored columns (0, 1, 3) for the first 68 positions.
    """
    # pred: (N, 5, L)
    # target: (N, L, 5) -> Transpose to (N, 5, L)
    target = target.permute(0, 2, 1)

    # Select scored columns
    pred_s = pred[:, Config.SCORED_TARGETS, :]
    target_s = target[:, Config.SCORED_TARGETS, :]

    mse = (pred_s - target_s) ** 2

    if mask is None:
        # Create mask for first 68 positions
        mask = torch.zeros_like(mse)
        mask[:, :, : Config.PRED_LEN] = 1.0

    mse = mse * mask
    # Sum over batch and sequence
    count = mask.sum(dim=(0, 2))

    # RMSE per column
    rmse = torch.sqrt(mse.sum(dim=(0, 2)) / (count + 1e-8))

    # Mean over columns
    return rmse.mean()


def train_model():
    Config.set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    X_train, p_train, y_train, _ = get_data("train")
    X_val, p_val, y_val, _ = get_data("val")

    train_ds = RNADataset(X_train, p_train, y_train)
    val_ds = RNADataset(X_val, p_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Model & Optimizer
    model = DDARNModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss_avg = 0

        for X, P, y in train_loader:
            X, P, y = X.to(device), P.to(device), y.to(device)
            optimizer.zero_grad()

            # Pass 1: Zero Feedback
            pred_1 = model(X, P, prev_pred=None)

            # Pass 2: Recycled Feedback (Detached)
            # Mask unscored positions in feedback to prevent leakage/noise
            fb_input = pred_1.detach().clone()
            fb_input[:, :, Config.PRED_LEN :] = 0

            pred_2 = model(X, P, prev_pred=fb_input)

            # Loss: Weighted sum
            loss1 = mcrmse_loss(pred_1, y)
            loss2 = mcrmse_loss(pred_2, y)
            loss = loss2 + 0.5 * loss1

            loss.backward()
            optimizer.step()

            train_loss_avg += loss.item()

        train_loss_avg /= len(train_loader)

        # Validation
        model.eval()
        val_loss_avg = 0
        with torch.no_grad():
            for X, P, y in val_loader:
                X, P, y = X.to(device), P.to(device), y.to(device)

                # Inference: 2 Passes
                pred_1 = model(X, P, prev_pred=None)

                fb_input = pred_1.clone()
                fb_input[:, :, Config.PRED_LEN :] = 0

                pred_2 = model(X, P, prev_pred=fb_input)

                val_loss = mcrmse_loss(pred_2, y)
                val_loss_avg += val_loss.item()

        val_loss_avg /= len(val_loader)
        scheduler.step(val_loss_avg)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}"
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )

    print(f"Best Validation Loss: {best_val_loss:.6f}")


def predict_and_submit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    X_test, p_test, _, ids = get_data("test")
    test_ds = RNADataset(X_test, p_test)
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Model
    model = DDARNModel().to(device)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Best model not found, using random weights.")

    model.eval()
    preds = []

    with torch.no_grad():
        for X, P in test_loader:
            X, P = X.to(device), P.to(device)

            # Inference: 2 Passes
            pred_1 = model(X, P, prev_pred=None)

            fb_input = pred_1.clone()
            fb_input[:, :, Config.PRED_LEN :] = 0

            pred_2 = model(X, P, prev_pred=fb_input)

            # Output is (N, 5, L) -> Transpose to (N, L, 5)
            preds.append(pred_2.permute(0, 2, 1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N_samples, 107, 5)

    # Generate Submission CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_preds = preds[i, seq_pos, :]

            # Clip negative values as degradation cannot be negative (physically)
            # though dataset allows negatives (noise), usually clipping helps metric
            row_preds = np.clip(row_preds, 0, None)

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])
            submission_data.append(row_dict)

    pd.DataFrame(submission_data).to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # This block is for testing the module independently
    train_model()
    predict_and_submit()
