import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Configuration
# ==========================================


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_74"

    # Data Dimensions
    SEQ_LEN = 107
    SCORED_LEN = 68
    NUM_TARGETS = 5
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    SCORED_TARGET_INDICES = [0, 1, 3]

    # Training Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    SEED = 42

    # Model Hyperparameters (HC-HSGFN)
    STEM_KERNEL = 3
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_LAYERS = 6
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
    LATENT_DIM = 64

    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_LAYERS = 4
    FEEDBACK_OUT_DIM = 32

    RNN_HIDDEN_DIM = 64
    DROPOUT = 0.1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


# ==========================================
# Data Processing & Caching
# ==========================================


def get_structure_map(structure):
    """Parses dot-bracket structure to find pairing partners."""
    map_arr = np.full(len(structure), -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                map_arr[i] = j
                map_arr[j] = i
    return map_arr


def process_data(load_cached_data=True):
    """
    Processes raw metadata into features.
    Implements strict caching logic using .npz files.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_files = {
        "train": os.path.join(Config.CACHE_DIR, "train_data_hc_hsgfn_v1.npz"),
        "val": os.path.join(Config.CACHE_DIR, "val_data_hc_hsgfn_v1.npz"),
        "test": os.path.join(Config.CACHE_DIR, "test_data_hc_hsgfn_v1.npz"),
    }

    # 1. Try to load cache
    if load_cached_data and all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data...")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(v, allow_pickle=True)
        return data

    # 2. Process from scratch
    print("Processing data from scratch...")

    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    def process_df(df, is_test=False):
        seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
        struct_map = {".": 0, "(": 1, ")": 2}
        loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

        n_samples = len(df)
        seq_len = Config.SEQ_LEN

        # Channels: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
        X_seq = np.zeros((n_samples, seq_len, 4), dtype=np.float32)
        X_struct = np.zeros((n_samples, seq_len, 3), dtype=np.float32)
        X_loop = np.zeros((n_samples, seq_len, 7), dtype=np.float32)
        X_partner = np.zeros((n_samples, seq_len, 4), dtype=np.float32)
        X_pair_idx = np.full((n_samples, seq_len), -1, dtype=np.int32)

        if not is_test:
            Y = np.zeros((n_samples, seq_len, 5), dtype=np.float32)
        else:
            Y = None

        for i, row in df.iterrows():
            # Sequence
            for j, char in enumerate(row["sequence"]):
                if char in seq_map:
                    X_seq[i, j, seq_map[char]] = 1.0

            # Structure & Partner
            pairs = get_structure_map(row["structure"])
            X_pair_idx[i] = pairs
            for j, char in enumerate(row["structure"]):
                if char in struct_map:
                    X_struct[i, j, struct_map[char]] = 1.0
                if pairs[j] != -1:
                    partner_idx = pairs[j]
                    p_char = row["sequence"][partner_idx]
                    if p_char in seq_map:
                        X_partner[i, j, seq_map[p_char]] = 1.0

            # Loop
            for j, char in enumerate(row["predicted_loop_type"]):
                if char in loop_map:
                    X_loop[i, j, loop_map[char]] = 1.0

            # Targets
            if not is_test:
                t_cols = [
                    "reactivity",
                    "deg_Mg_pH10",
                    "deg_pH10",
                    "deg_Mg_50C",
                    "deg_50C",
                ]
                for k, col_name in enumerate(t_cols):
                    val_list = ast.literal_eval(row[col_name])
                    length = len(val_list)
                    Y[i, :length, k] = val_list

        X = np.concatenate([X_seq, X_struct, X_loop, X_partner], axis=-1)
        return X, X_pair_idx, Y, df["id"].values

    X_train, P_train, Y_train, ids_train = process_df(train_df)
    X_val, P_val, Y_val, ids_val = process_df(val_df)
    X_test, P_test, _, ids_test = process_df(test_df, is_test=True)

    # 3. Save to cache
    np.savez(
        cache_files["train"],
        inputs=X_train,
        pairs=P_train,
        targets=Y_train,
        ids=ids_train,
    )
    np.savez(cache_files["val"], inputs=X_val, pairs=P_val, targets=Y_val, ids=ids_val)
    np.savez(cache_files["test"], inputs=X_test, pairs=P_test, ids=ids_test)

    # Reload to ensure consistent return format
    data = {}
    for k, v in cache_files.items():
        data[k] = np.load(v, allow_pickle=True)
    return data


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.inputs = data_dict["inputs"]
        self.pairs = data_dict["pairs"]
        self.mode = mode
        self.targets = data_dict["targets"] if mode != "test" else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Permute to (C, L) for Conv1d
        x = torch.tensor(self.inputs[idx], dtype=torch.float32).permute(1, 0)
        p = torch.tensor(self.pairs[idx], dtype=torch.long)

        if self.mode != "test":
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p, y
        return x, p


# ==========================================
# Model Architecture (HC-HSGFN)
# ==========================================


class DenseDilatedBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.ln1 = nn.LayerNorm(Config.SEQ_LEN)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(Config.SEQ_LEN)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.drop(out)
        return out


class HC_HSGFN(nn.Module):
    def __init__(self):
        super().__init__()

        # Hybrid Stem
        self.stem_conv = nn.Conv1d(18, 32, kernel_size=3, padding=1)
        self.stem_ln = nn.LayerNorm(Config.SEQ_LEN)
        self.stem_act = nn.SiLU()
        self.backbone_in_dim = 18 + 32  # 50

        # Dense Backbone
        self.blocks = nn.ModuleList()
        current_dim = self.backbone_in_dim
        for d in Config.BACKBONE_DILATIONS:
            blk = DenseDilatedBlock(
                current_dim, Config.BACKBONE_GROWTH_RATE, d, Config.DROPOUT
            )
            self.blocks.append(blk)
            current_dim += Config.BACKBONE_GROWTH_RATE

        self.latent_proj = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)

        # Feedback Module
        self.fb_stem = nn.Conv1d(5, 16, kernel_size=3, padding=1)
        self.fb_ln = nn.LayerNorm(Config.SEQ_LEN)
        self.fb_act = nn.SiLU()

        self.fb_blocks = nn.ModuleList()
        curr_fb_dim = 16
        for _ in range(Config.FEEDBACK_LAYERS):
            self.fb_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        curr_fb_dim,
                        Config.FEEDBACK_GROWTH_RATE,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.LayerNorm(Config.SEQ_LEN),
                    nn.SiLU(),
                )
            )
            curr_fb_dim = Config.FEEDBACK_GROWTH_RATE
        self.fb_out = nn.Conv1d(curr_fb_dim, Config.FEEDBACK_OUT_DIM, kernel_size=1)

        # Interaction & Head
        # Input: Self(Latent+FB) + Partner(Latent+FB) = (64+32)*2 = 192
        self.rnn = nn.GRU(
            192,
            Config.RNN_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(Config.RNN_HIDDEN_DIM * 2, 5)

    def forward_backbone(self, x):
        # Branch A (Raw) + Branch B (Conv)
        branch_b = self.stem_act(self.stem_ln(self.stem_conv(x)))
        feat = torch.cat([x, branch_b], dim=1)

        features = [feat]
        for block in self.blocks:
            out = block(torch.cat(features, dim=1))
            features.append(out)

        return self.latent_proj(torch.cat(features, dim=1))  # (B, 64, L)

    def forward_feedback(self, y_prev):
        # Mask unscored channels (indices 2, 4) to prevent noise
        mask = torch.zeros_like(y_prev)
        mask[:, Config.SCORED_TARGET_INDICES, :] = 1.0
        y_masked = y_prev * mask

        out = self.fb_act(self.fb_ln(self.fb_stem(y_masked)))
        for blk in self.fb_blocks:
            out = blk(out)
        return self.fb_out(out)  # (B, 32, L)

    def forward_head(self, z, e_fb, pairs):
        B, _, L = z.shape
        self_vec = torch.cat([z, e_fb], dim=1)  # (B, 96, L)

        # Gather partner vectors
        batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, L)
        valid_mask = pairs != -1
        safe_pairs = pairs.clone()
        safe_pairs[~valid_mask] = 0

        self_vec_t = self_vec.permute(0, 2, 1)  # (B, L, 96)
        partner_vec_t = self_vec_t[batch_idx, safe_pairs]
        partner_vec_t[~valid_mask] = 0.0

        combined = torch.cat([self_vec_t, partner_vec_t], dim=2)  # (B, L, 192)
        rnn_out, _ = self.rnn(combined)
        return self.head(rnn_out)

    def forward(self, x, pairs, y_prev=None):
        z = self.forward_backbone(x)
        if y_prev is None:
            y_prev = torch.zeros((x.shape[0], 5, x.shape[2]), device=x.device)
        e_fb = self.forward_feedback(y_prev)
        preds = self.forward_head(z, e_fb, pairs)
        return preds, z


# ==========================================
# Training & Submission
# ==========================================


def mcrmse_loss(pred, target):
    """Calculates MCRMSE only on scored columns and scored positions."""
    pred_s = pred[:, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES]
    target_s = target[:, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES]
    mse = torch.mean((pred_s - target_s) ** 2, dim=(0, 1))
    return torch.mean(torch.sqrt(mse))


def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = process_data(load_cached_data=True)
    train_loader = DataLoader(
        RNADataset(data["train"]),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        RNADataset(data["val"]), batch_size=Config.BATCH_SIZE, shuffle=False
    )

    model = HC_HSGFN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for x, p, y in train_loader:
            x, p, y = x.to(device), p.to(device), y.to(device)
            optimizer.zero_grad()

            # Iterative Refinement
            pred1, _ = model(x, p, y_prev=None)
            pred1_detached = pred1.detach().permute(0, 2, 1)  # Feedback input format
            pred2, _ = model(x, p, y_prev=pred1_detached)

            loss = mcrmse_loss(pred2, y) + 0.5 * mcrmse_loss(pred1, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_sse = torch.zeros(3, device=device)
        val_count = 0
        with torch.no_grad():
            for x, p, y in val_loader:
                x, p, y = x.to(device), p.to(device), y.to(device)
                pred1, _ = model(x, p, y_prev=None)
                pred2, _ = model(x, p, y_prev=pred1.permute(0, 2, 1))

                p_s = pred2[:, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES]
                t_s = y[:, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES]
                val_sse += torch.sum((p_s - t_s) ** 2, dim=(0, 1))
                val_count += x.shape[0] * Config.SCORED_LEN

        val_mcrmse = torch.mean(torch.sqrt(val_sse / val_count)).item()
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        scheduler.step(val_mcrmse)
        if val_mcrmse < best_val_loss:
            best_val_loss = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break

    return best_model_path


def generate_submission(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = process_data(load_cached_data=True)
    loader = DataLoader(
        RNADataset(data["test"], mode="test"),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )
    ids = data["test"]["ids"]

    model = HC_HSGFN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x, p in loader:
            x, p = x.to(device), p.to(device)
            pred1, _ = model(x, p, y_prev=None)
            pred2, _ = model(x, p, y_prev=pred1.permute(0, 2, 1))
            preds.append(pred2.cpu().numpy())

    all_preds = np.concatenate(preds, axis=0)

    sub_data = []
    for i, sid in enumerate(ids):
        for pos in range(Config.SEQ_LEN):
            row = [f"{sid}_{pos}"] + all_preds[i, pos].tolist()
            sub_data.append(row)

    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    pd.DataFrame(sub_data, columns=cols).to_csv(
        "./submission/submission.csv", index=False
    )
    print("Submission saved.")
