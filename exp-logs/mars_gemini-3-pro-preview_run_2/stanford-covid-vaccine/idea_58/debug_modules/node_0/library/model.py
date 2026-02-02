import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, MCRMSELoss, GlobalMetrics

# =========================================================================
# Data Processing & Caching
# =========================================================================


def get_couples(structure):
    """
    Converts dot-bracket structure to a list of paired indices.
    Returns a mapping where map[i] = j if i is paired with j, else -1.
    """
    mapping = np.full(len(structure), -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                mapping[i] = j
                mapping[j] = i
    return mapping


def process_data(df):
    """
    Generates inputs and targets from the dataframe.
    """
    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Mappings
    base_map = {c: i for i, c in enumerate(Config.BASES)}
    struct_map = {c: i for i, c in enumerate(Config.STRUCTS)}
    loop_map = {c: i for i, c in enumerate(Config.LOOPS)}

    # Pre-allocate
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner) = 18
    features = np.zeros((n_samples, Config.INPUT_DIM, seq_len), dtype=np.float32)
    bpp_indices = np.full((n_samples, seq_len), -1, dtype=np.int32)

    for idx in range(n_samples):
        seq = sequences[idx]
        struct = structures[idx]
        loop = loops[idx]

        # Parse pairs
        pairs = get_couples(struct)
        bpp_indices[idx] = pairs

        for i in range(seq_len):
            # One-Hot Sequence (0-3)
            if seq[i] in base_map:
                features[idx, base_map[seq[i]], i] = 1.0

            # One-Hot Structure (4-6)
            if struct[i] in struct_map:
                features[idx, 4 + struct_map[struct[i]], i] = 1.0

            # One-Hot Loop (7-13)
            if loop[i] in loop_map:
                features[idx, 7 + loop_map[loop[i]], i] = 1.0

            # Partner Identity (14-17)
            partner_idx = pairs[i]
            if partner_idx != -1:
                partner_base = seq[partner_idx]
                if partner_base in base_map:
                    features[idx, 14 + base_map[partner_base], i] = 1.0

    # Parse Targets (if available)
    targets = None
    # Check if the first scored column exists to determine if this is training data
    if Config.SCORED_COLS[0] in df.columns:
        targets = np.zeros(
            (n_samples, seq_len, len(Config.TARGET_COLS)), dtype=np.float32
        )

        for t_i, col in enumerate(Config.TARGET_COLS):
            # Helper to safely parse stringified lists
            def parse_val(x):
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except:
                        return [0.0] * seq_len
                elif isinstance(x, (list, np.ndarray)):
                    return x
                return [0.0] * seq_len

            values = df[col].apply(parse_val).tolist()

            for idx, val_list in enumerate(values):
                length = min(len(val_list), seq_len)
                targets[idx, :length, t_i] = val_list[:length]

    return {
        "ids": ids,
        "features": features,
        "bpp_indices": bpp_indices,
        "targets": targets,
    }


def load_dataset(mode="train", load_cached_data=True):
    """
    Loads data with caching mechanism.
    """
    cache_path = getattr(Config, f"CACHE_{mode.upper()}")
    csv_path = getattr(Config, f"{mode.upper()}_CSV")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "ids": data["ids"],
            "features": data["features"],
            "bpp_indices": data["bpp_indices"],
            "targets": data["targets"] if "targets" in data else None,
        }

    # 2. Process from Scratch
    print(f"Processing {mode} data from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    # Debug subset
    if Config.DEBUG and mode == "train":
        df = df.head(Config.MAX_DEBUG_SAMPLES)

    processed = process_data(df)

    # 3. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {
        "ids": processed["ids"],
        "features": processed["features"],
        "bpp_indices": processed["bpp_indices"],
    }
    if processed["targets"] is not None:
        save_dict["targets"] = processed["targets"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved cache to {cache_path}")

    return processed


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.features = data["features"]
        self.bpp_indices = data["bpp_indices"]
        self.targets = data["targets"]
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (18, 107)
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        # BPP: (107,)
        bpp = torch.tensor(self.bpp_indices[idx], dtype=torch.long)

        if self.targets is not None:
            # Targets: (107, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, bpp, y
        else:
            return x, bpp


# =========================================================================
# Model Architecture
# =========================================================================


class PermuteLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) -> LN -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x


class SpatialStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            PermuteLayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.op(x)


class PostActDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        # Decoupled Spatial Aggregation
        self.spatial = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            PermuteLayerNorm(growth_rate),
            nn.SiLU(),
        )
        # Channel Mixing
        self.mixing = nn.Sequential(
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            PermuteLayerNorm(growth_rate),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
        )

    def forward(self, x):
        out = self.spatial(x)
        out = self.mixing(out)
        return out


class GC_SSN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Spatial Stem
        self.stem = SpatialStem(Config.INPUT_DIM, Config.HIDDEN_DIM)

        # 2. Backbone (Dense Dilated TCN)
        self.backbone_blocks = nn.ModuleList()
        curr_channels = Config.HIDDEN_DIM

        for d in Config.DILATIONS:
            blk = PostActDenseBlock(curr_channels, Config.GROWTH_RATE, d)
            self.backbone_blocks.append(blk)
            curr_channels += Config.GROWTH_RATE

        self.latent_proj = nn.Conv1d(curr_channels, Config.HIDDEN_DIM, kernel_size=1)

        # 3. Feedback Module
        self.feedback_stem = SpatialStem(
            Config.FEEDBACK_INPUT_DIM, Config.FEEDBACK_GROWTH_RATE
        )
        self.feedback_blocks = nn.ModuleList()
        curr_fb_channels = Config.FEEDBACK_GROWTH_RATE

        for d in Config.DILATIONS:
            blk = PostActDenseBlock(curr_fb_channels, Config.FEEDBACK_GROWTH_RATE, d)
            self.feedback_blocks.append(blk)
            curr_fb_channels += Config.FEEDBACK_GROWTH_RATE

        self.feedback_proj = nn.Conv1d(
            curr_fb_channels, Config.FEEDBACK_DIM, kernel_size=1
        )

        # 4. Interaction & Aggregation
        # Input to RNN: Z (64) + Z_partner (64) + E_fb (32) + E_fb_partner (32) = 192
        self.rnn_input_dim = (Config.HIDDEN_DIM + Config.FEEDBACK_DIM) * 2

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(Config.RNN_HIDDEN * 2, len(Config.TARGET_COLS))

        # Mask for feedback channels: Scored=1, Unscored=0
        # TARGET_COLS = [reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C]
        # Mask = [1, 1, 1, 0, 0]
        self.register_buffer(
            "channel_mask",
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32).view(1, 5, 1),
        )

    def forward_backbone(self, x):
        feat = self.stem(x)
        features = [feat]

        for blk in self.backbone_blocks:
            inp = torch.cat(features, dim=1)
            out = blk(inp)
            features.append(out)

        total_feat = torch.cat(features, dim=1)
        z = self.latent_proj(total_feat)  # (B, 64, L)
        return z

    def forward_feedback(self, y_prev):
        # y_prev: (B, L, 5) -> (B, 5, L)
        y_in = y_prev.permute(0, 2, 1)
        y_in = y_in * self.channel_mask

        feat = self.feedback_stem(y_in)
        features = [feat]

        for blk in self.feedback_blocks:
            inp = torch.cat(features, dim=1)
            out = blk(inp)
            features.append(out)

        total_feat = torch.cat(features, dim=1)
        e_fb = self.feedback_proj(total_feat)  # (B, 32, L)
        return e_fb

    def predict_step(self, z, e_fb, bpp_indices):
        # z: (B, 64, L), e_fb: (B, 32, L)
        self_vec = torch.cat([z, e_fb], dim=1)  # (B, 96, L)

        B, C, L = self_vec.shape

        # Gather Partner Vector
        gather_idx = bpp_indices.clone()
        mask_unpaired = gather_idx == -1
        gather_idx[mask_unpaired] = 0

        gather_idx_exp = gather_idx.unsqueeze(1).expand(-1, C, -1)
        partner_vec = torch.gather(self_vec, 2, gather_idx_exp)  # (B, 96, L)

        # Null-Masking
        partner_vec = partner_vec * (~mask_unpaired.unsqueeze(1))

        # Fusion
        rnn_in = torch.cat([self_vec, partner_vec], dim=1)  # (B, 192, L)

        # Aggregation
        rnn_in = rnn_in.permute(0, 2, 1)  # (B, L, 192)
        rnn_out, _ = self.rnn(rnn_in)  # (B, L, 128)

        y_pred = self.head(rnn_out)  # (B, L, 5)
        return y_pred

    def forward(self, x, bpp_indices):
        z = self.forward_backbone(x)

        B, _, L = x.shape
        y_curr = torch.zeros((B, L, 5), device=x.device)

        # Pass 1
        e_fb_1 = self.forward_feedback(y_curr)
        y_1 = self.predict_step(z, e_fb_1, bpp_indices)

        # Pass 2
        y_curr_2 = y_1.detach()
        e_fb_2 = self.forward_feedback(y_curr_2)
        y_2 = self.predict_step(z, e_fb_2, bpp_indices)

        return y_2, y_1


# =========================================================================
# Training & Inference Logic
# =========================================================================


def train_model():
    set_seed(Config.SEED)

    train_data = load_dataset("train")
    val_data = load_dataset("val")

    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")

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

    model = GC_SSN().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = MCRMSELoss()

    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_avg = 0.0

        for x, bpp, y in train_loader:
            x, bpp, y = x.to(Config.DEVICE), bpp.to(Config.DEVICE), y.to(Config.DEVICE)

            optimizer.zero_grad()
            y_final, y_aux = model(x, bpp)

            # Loss on both passes
            loss = criterion(y_final, y) + 0.5 * criterion(y_aux, y)
            loss.backward()
            optimizer.step()

            train_loss_avg += loss.item()

        train_loss_avg /= len(train_loader)

        model.eval()
        global_metrics = GlobalMetrics()

        with torch.no_grad():
            for x, bpp, y in val_loader:
                x, bpp, y = (
                    x.to(Config.DEVICE),
                    bpp.to(Config.DEVICE),
                    y.to(Config.DEVICE),
                )
                y_final, _ = model(x, bpp)
                global_metrics.update(y_final, y)

        val_score = global_metrics.compute()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss_avg:.6f} | Val MCRMSE: {val_score:.10f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step(val_score)

        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_score:.10f}")


def predict_and_submit():
    set_seed(Config.SEED)

    test_data = load_dataset("test")
    test_ds = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model = GC_SSN().to(Config.DEVICE)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No trained model found. Skipping inference.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    print("Generating predictions...")
    all_preds = []

    with torch.no_grad():
        for x, bpp in test_loader:
            x, bpp = x.to(Config.DEVICE), bpp.to(Config.DEVICE)
            y_final, _ = model(x, bpp)
            all_preds.append(y_final.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)

    # Prepare Submission
    ids = test_data["ids"]
    submission_rows = []

    # Target Cols: [reactivity, deg_Mg_pH10, deg_Mg_50C, deg_pH10, deg_50C]
    # Submission:  [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    # Indices:     0, 1, 3, 2, 4

    for i, sample_id in enumerate(ids):
        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"
            preds = all_preds[i, pos]

            vals = [
                preds[0],  # reactivity
                preds[1],  # deg_Mg_pH10
                preds[3],  # deg_pH10
                preds[2],  # deg_Mg_50C
                preds[4],  # deg_50C
            ]
            submission_rows.append([row_id] + vals)

    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=cols)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
