import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------


class Config:
    # File Paths
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"
    CACHE_DIR = "./working/idea_78"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache Keys
    CACHE_KEY_TRAIN = "train_data_hc_sdrn_v1.npz"
    CACHE_KEY_VAL = "val_data_hc_sdrn_v1.npz"
    CACHE_KEY_TEST = "test_data_hc_sdrn_v1.npz"

    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features
    # Sequence (4) + Structure (3) + LoopType (7) + PartnerSeq (4) = 18
    INPUT_DIM = 18

    # Model Architecture
    BACKBONE_GROWTH = 64
    BACKBONE_LAYERS = 6
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
    LATENT_DIM = 64

    FEEDBACK_CHANNELS = 5  # Number of target columns
    FEEDBACK_GROWTH = 16

    RNN_HIDDEN = 64
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 25
    LR = 1e-3
    NUM_WORKERS = 2
    SEED = 42

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Indices of scored columns within TARGET_COLS: 0, 1, 3
    SCORED_INDICES = [0, 1, 3]


# ------------------------------------------------------------------------------
# DATA PROCESSING & CACHING
# ------------------------------------------------------------------------------


def get_structure_adj(structure, seq_len):
    """Parses dot-bracket structure to find partner indices."""
    adj = np.full(seq_len, -1, dtype=np.int32)
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


def process_data(metadata_path, is_test=False):
    """
    Processes metadata CSV into numpy arrays for training/inference.
    Generates One-Hot features and Partner Indices.
    """
    df = pd.read_csv(metadata_path)

    # Mappings
    seq_map = {c: i for i, c in enumerate("AGUC")}
    struct_map = {c: i for i, c in enumerate(".()")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    ids = df["id"].values
    seq_len = Config.SEQ_LEN
    n_samples = len(df)

    # Initialize arrays
    # Features: Sequence(4), Structure(3), Loop(7), PartnerIdentity(4)
    X = np.zeros((n_samples, seq_len, Config.INPUT_DIM), dtype=np.float32)
    partner_indices = np.full((n_samples, seq_len), -1, dtype=np.int32)

    # Targets
    if not is_test:
        Y = np.zeros((n_samples, seq_len, 5), dtype=np.float32)
    else:
        Y = None

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Base Features
        for i, char in enumerate(seq):
            if char in seq_map:
                X[idx, i, seq_map[char]] = 1.0

        for i, char in enumerate(struct):
            if char in struct_map:
                X[idx, i, 4 + struct_map[char]] = 1.0

        for i, char in enumerate(loop):
            if char in loop_map:
                X[idx, i, 4 + 3 + loop_map[char]] = 1.0

        # 2. Partner Indices & Partner Identity
        adj = get_structure_adj(struct, seq_len)
        partner_indices[idx] = adj

        for i, p_idx in enumerate(adj):
            if p_idx != -1:
                # Partner identity is the sequence char at p_idx
                p_char = seq[p_idx]
                if p_char in seq_map:
                    X[idx, i, 4 + 3 + 7 + seq_map[p_char]] = 1.0

        # 3. Targets
        if not is_test:
            for t_i, col in enumerate(Config.TARGET_COLS):
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(row[col])
                    # Pad to 107 with 0 (though loss is masked, we need shape)
                    length = len(val_list)
                    Y[idx, :length, t_i] = val_list
                except:
                    pass

    return ids, X, partner_indices, Y


def load_or_process_data(mode="train", load_cached_data=True):
    """
    Caching mechanism for dataset.
    mode: 'train', 'val', 'test'
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if mode == "train":
        path = Config.TRAIN_METADATA
        cache_key = Config.CACHE_KEY_TRAIN
        is_test = False
    elif mode == "val":
        path = Config.VAL_METADATA
        cache_key = Config.CACHE_KEY_VAL
        is_test = False
    else:
        path = Config.TEST_METADATA
        cache_key = Config.CACHE_KEY_TEST
        is_test = True

    cache_path = os.path.join(Config.CACHE_DIR, cache_key)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if is_test:
            return data["ids"], data["X"], data["partner_indices"]
        else:
            return data["ids"], data["X"], data["partner_indices"], data["Y"]

    print(f"Processing {mode} data from {path}...")
    result = process_data(path, is_test=is_test)

    print(f"Saving {mode} data to cache...")
    if is_test:
        np.savez_compressed(
            cache_path, ids=result[0], X=result[1], partner_indices=result[2]
        )
        return result[0], result[1], result[2]
    else:
        np.savez_compressed(
            cache_path,
            ids=result[0],
            X=result[1],
            partner_indices=result[2],
            Y=result[3],
        )

    return result


class RNADataset(Dataset):
    def __init__(self, X, partner_indices, Y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.partner_indices = torch.tensor(partner_indices, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32) if Y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.partner_indices[idx], self.Y[idx]
        return self.X[idx], self.partner_indices[idx]


# ------------------------------------------------------------------------------
# MODEL: HC-SDRN
# ------------------------------------------------------------------------------


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.LayerNorm(Config.SEQ_LEN),
            nn.SiLU(),
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            nn.LayerNorm(Config.SEQ_LEN),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
        )

    def forward(self, x):
        # Permute for LayerNorm (N, C, L) -> (N, L, C)
        out = self.net[0](x)  # Conv
        out = out.permute(0, 2, 1)  # N, L, C
        out = self.net[1](out)  # LN
        out = out.permute(0, 2, 1)  # N, C, L
        out = self.net[2](out)  # SiLU

        out = self.net[3](out)  # Conv1x1
        out = out.permute(0, 2, 1)
        out = self.net[4](out)  # LN
        out = out.permute(0, 2, 1)
        out = self.net[5](out)  # SiLU
        out = self.net[6](out)  # Dropout

        return out


class FeedbackModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, padding=1),
            nn.LayerNorm(Config.SEQ_LEN),
            nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=2, dilation=2),
            nn.LayerNorm(Config.SEQ_LEN),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (N, L, 5) -> Permute to (N, 5, L)
        x = x.permute(0, 2, 1)

        x = self.stem[0](x)
        x = x.permute(0, 2, 1)
        x = self.stem[1](x)
        x = x.permute(0, 2, 1)
        x = self.stem[2](x)

        x = self.block[0](x)
        x = x.permute(0, 2, 1)
        x = self.block[1](x)
        x = x.permute(0, 2, 1)
        x = self.block[2](x)

        # Output: (N, 16, L) -> Permute to (N, L, 16)
        return x.permute(0, 2, 1)


class HC_SDRN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Spatial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                Config.INPUT_DIM, Config.BACKBONE_GROWTH, kernel_size=3, padding=1
            ),
            nn.LayerNorm(Config.SEQ_LEN),
            nn.SiLU(),
        )

        # 2. Dense Backbone
        self.blocks = nn.ModuleList()
        curr_dim = Config.BACKBONE_GROWTH

        for d in Config.BACKBONE_DILATIONS:
            blk = DenseBlock(curr_dim, Config.BACKBONE_GROWTH, d)
            self.blocks.append(blk)
            curr_dim += Config.BACKBONE_GROWTH

        self.latent_proj = nn.Conv1d(curr_dim, Config.LATENT_DIM, kernel_size=1)

        # 3. Feedback
        self.feedback_net = FeedbackModule()

        # 4. Interaction & Aggregation
        # Input to RNN: Latent(64) + Feedback(16) + PartnerLatent(64) + PartnerFeedback(16) = 160
        rnn_input_dim = (Config.LATENT_DIM + 16) * 2
        self.gru = nn.GRU(
            rnn_input_dim, Config.RNN_HIDDEN, batch_first=True, bidirectional=True
        )

        self.head = nn.Linear(Config.RNN_HIDDEN * 2, 5)

    def forward_backbone(self, x):
        # x: (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)

        # Stem
        out = self.stem[0](x)
        out = out.permute(0, 2, 1)
        out = self.stem[1](out)
        out = out.permute(0, 2, 1)
        out = self.stem[2](out)

        features = [out]

        # Dense Blocks
        for blk in self.blocks:
            inp = torch.cat(features, dim=1)
            new_feat = blk(inp)
            features.append(new_feat)

        # Final Projection
        all_feats = torch.cat(features, dim=1)
        z = self.latent_proj(all_feats)  # (N, 64, L)
        z = z.permute(0, 2, 1)  # (N, L, 64)
        return z

    def forward_head(self, z, prev_pred, partner_indices):
        # z: (N, L, 64)
        # prev_pred: (N, L, 5)

        # Mask unscored channels in feedback
        # Scored indices: 0, 1, 3. Unscored: 2, 4.
        mask = torch.zeros_like(prev_pred)
        mask[:, :, [0, 1, 3]] = 1.0
        masked_pred = prev_pred * mask

        # Feedback embedding
        e_fb = self.feedback_net(masked_pred)  # (N, L, 16)

        # Combine Self
        h_self = torch.cat([z, e_fb], dim=-1)  # (N, L, 80)

        # Gather Partner
        batch_size, seq_len, _ = h_self.shape

        # Create batch indices
        batch_idx = (
            torch.arange(batch_size, device=z.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Handle -1 in partner_indices (unpaired) -> map to 0 temporarily then mask
        p_idx_safe = partner_indices.clone()
        p_idx_safe[p_idx_safe == -1] = 0

        h_partner = h_self[batch_idx, p_idx_safe]  # (N, L, 80)

        # Mask unpaired
        mask_pair = (partner_indices != -1).unsqueeze(-1).float()
        h_partner = h_partner * mask_pair

        # Fuse
        rnn_in = torch.cat([h_self, h_partner], dim=-1)  # (N, L, 160)

        # GRU
        rnn_out, _ = self.gru(rnn_in)

        # Head
        logits = self.head(rnn_out)
        return logits

    def forward(self, x, partner_indices, prev_pred=None):
        z = self.forward_backbone(x)

        if prev_pred is None:
            prev_pred = torch.zeros((x.shape[0], Config.SEQ_LEN, 5), device=x.device)

        # Pass 1
        y1 = self.forward_head(z, prev_pred, partner_indices)

        # Pass 2 (Recycling)
        y2 = self.forward_head(z, y1.detach(), partner_indices)

        return y1, y2


# ------------------------------------------------------------------------------
# TRAINING UTILS
# ------------------------------------------------------------------------------


def mcrmse_loss(
    pred, target, scored_indices=Config.SCORED_INDICES, pred_len=Config.PRED_LEN
):
    # pred, target: (N, L, 5)
    # Only first pred_len positions
    pred = pred[:, :pred_len, :]
    target = target[:, :pred_len, :]

    # Only scored columns
    pred = pred[:, :, scored_indices]
    target = target[:, :, scored_indices]

    # Flatten samples and length
    pred_flat = pred.reshape(-1, len(scored_indices))
    target_flat = target.reshape(-1, len(scored_indices))

    mse_cols = torch.mean((pred_flat - target_flat) ** 2, dim=0)
    rmse_cols = torch.sqrt(mse_cols)
    return torch.mean(rmse_cols)


def validate(model, loader, device):
    model.eval()
    total_mse = torch.zeros(len(Config.SCORED_INDICES), device=device)
    count = 0

    with torch.no_grad():
        for x, p_idx, y in loader:
            x, p_idx, y = x.to(device), p_idx.to(device), y.to(device)
            _, y_pred = model(x, p_idx)

            # Slice
            y_pred = y_pred[:, : Config.PRED_LEN, Config.SCORED_INDICES]
            y_true = y[:, : Config.PRED_LEN, Config.SCORED_INDICES]

            # Accumulate SSE per column
            diff = (y_pred - y_true).reshape(-1, len(Config.SCORED_INDICES))
            total_mse += torch.sum(diff**2, dim=0)
            count += diff.shape[0]

    rmse = torch.sqrt(total_mse / count)
    return torch.mean(rmse).item()


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_ids, train_X, train_P, train_Y = load_or_process_data("train")
    val_ids, val_X, val_P, val_Y = load_or_process_data("val")

    train_ds = RNADataset(train_X, train_P, train_Y)
    val_ds = RNADataset(val_X, val_P, val_Y)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model
    model = HC_SDRN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_score = float("inf")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_acc = 0

        for x, p_idx, y in train_loader:
            x, p_idx, y = x.to(device), p_idx.to(device), y.to(device)

            optimizer.zero_grad()
            y1, y2 = model(x, p_idx)

            loss1 = mcrmse_loss(y1, y)
            loss2 = mcrmse_loss(y2, y)
            loss = loss2 + 0.5 * loss1

            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item()

        avg_train_loss = train_loss_acc / len(train_loader)
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        scheduler.step(val_score)

        if val_score < best_score:
            best_score = val_score
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )

    print(f"Best Validation Score: {best_score:.6f}")

    # Inference
    print("Generating Submission...")
    test_ids, test_X, test_P = load_or_process_data("test")
    test_ds = RNADataset(test_X, test_P)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.load_state_dict(
        torch.load(
            os.path.join(Config.CACHE_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    preds = []
    with torch.no_grad():
        for x, p_idx in test_loader:
            x, p_idx = x.to(device), p_idx.to(device)
            _, y_pred = model(x, p_idx)
            preds.append(y_pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N, 107, 5)

    # Format Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    sub_ids = []
    sub_data = []

    for i, sample_id in enumerate(test_ids):
        seq_len = Config.SEQ_LEN
        for pos in range(seq_len):
            sub_ids.append(f"{sample_id}_{pos}")
            sub_data.append(preds[i, pos])

    sub_df = pd.DataFrame(sub_data, columns=Config.TARGET_COLS)
    sub_df.insert(0, "id_seqpos", sub_ids)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
