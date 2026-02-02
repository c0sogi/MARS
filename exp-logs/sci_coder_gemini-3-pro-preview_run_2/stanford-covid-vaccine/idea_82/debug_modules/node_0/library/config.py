import os
import random
import ast
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
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_82")
    SUBMISSION_DIR = "./submission"

    # Data Dimensions
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Scored columns for metric: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 3)
    SCORED_COLS_INDICES = [0, 1, 3]

    # Model Hyperparameters
    HIDDEN_DIM = 64
    GROWTH_RATE = 64
    FEEDBACK_GROWTH_RATE = 16
    LATENT_DIM = 64
    FEEDBACK_DIM = 32
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 16
    LR = 1e-3
    EPOCHS = 15
    PATIENCE = 5
    SEED = 42


# Ensure directories exist
os.makedirs(Config.IDEA_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed(Config.SEED)

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_map(structure):
    """Parses dot-bracket structure to get paired indices."""
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


def process_data(df, is_test=False):
    # 1. Sequences (One-Hot)
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    sequences = []
    for seq in df["sequence"]:
        vec = np.zeros((Config.SEQ_LENGTH, 4), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in seq_map:
                vec[i, seq_map[char]] = 1.0
        sequences.append(vec)
    sequences = np.array(sequences)

    # 2. Structures (One-Hot) & Pair Maps
    struct_map = {".": 0, "(": 1, ")": 2}
    structures = []
    pair_maps = []
    for struct in df["structure"]:
        vec = np.zeros((Config.SEQ_LENGTH, 3), dtype=np.float32)
        for i, char in enumerate(struct):
            if char in struct_map:
                vec[i, struct_map[char]] = 1.0
        structures.append(vec)
        pair_maps.append(get_structure_map(struct))
    structures = np.array(structures)
    pair_maps = np.array(pair_maps)

    # 3. Loop Types (One-Hot)
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    loops = []
    for lp in df["predicted_loop_type"]:
        vec = np.zeros((Config.SEQ_LENGTH, 7), dtype=np.float32)
        for i, char in enumerate(lp):
            if char in loop_map:
                vec[i, loop_map[char]] = 1.0
        loops.append(vec)
    loops = np.array(loops)

    # 4. Partner Identity
    partner_identities = []
    for i in range(len(sequences)):
        p_id = np.zeros((Config.SEQ_LENGTH, 4), dtype=np.float32)
        pm = pair_maps[i]
        seq = sequences[i]
        valid = pm != -1
        p_id[valid] = seq[pm[valid]]
        partner_identities.append(p_id)
    partner_identities = np.array(partner_identities)

    # 5. Targets (Anchored)
    targets = None
    if not is_test:
        targets = np.zeros((len(df), Config.SEQ_LENGTH, 5), dtype=np.float32)
        for idx, col in enumerate(Config.TARGET_COLS):
            # Parse stringified lists safely
            vals = df[col].apply(
                lambda x: (
                    np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
                )
            )
            for i, val in enumerate(vals):
                length = min(len(val), Config.SEQ_LENGTH)
                targets[i, :length, idx] = val[:length]
                # Tail (68-107) remains 0.0 for Anchoring

    return {
        "sequence": sequences,
        "structure": structures,
        "loop": loops,
        "partner_identity": partner_identities,
        "pair_map": pair_maps,
        "targets": targets,
        "ids": df["id"].values,
    }


def get_dataset(mode="train", load_cached_data=True):
    cache_file = os.path.join(Config.IDEA_DIR, f"{mode}_data_ahc_hdn_v1.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {mode} data from scratch...")
    if mode == "test":
        df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
        data = process_data(df, is_test=True)
    else:
        df = pd.read_csv(os.path.join(Config.METADATA_DIR, f"{mode}.csv"))
        data = process_data(df, is_test=False)

    print(f"Saving {mode} data to cache...")
    np.savez(cache_file, **data)
    return data


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.sequence = torch.FloatTensor(data["sequence"])
        self.structure = torch.FloatTensor(data["structure"])
        self.loop = torch.FloatTensor(data["loop"])
        self.partner_identity = torch.FloatTensor(data["partner_identity"])
        self.pair_map = torch.LongTensor(data["pair_map"])
        self.mode = mode
        if mode != "test":
            self.targets = torch.FloatTensor(data["targets"])

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        # Concatenate static features: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
        features = torch.cat(
            [
                self.sequence[idx],
                self.structure[idx],
                self.loop[idx],
                self.partner_identity[idx],
            ],
            dim=1,
        )

        pair_map = self.pair_map[idx]

        if self.mode == "test":
            return features, pair_map
        else:
            return features, pair_map, self.targets[idx]


# ==================================================================================
# MODEL
# ==================================================================================


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
        )
        self.pointwise = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        # Post-Activation structure: Conv -> LN -> SiLU
        out = self.net(x)

        # LN 1
        out = out.permute(0, 2, 1)
        out = F.layer_norm(out, out.shape[2:])
        out = out.permute(0, 2, 1)
        out = self.act(out)

        out = self.pointwise(out)

        # LN 2
        out = out.permute(0, 2, 1)
        out = F.layer_norm(out, out.shape[2:])
        out = out.permute(0, 2, 1)
        out = self.act(out)

        out = self.dropout(out)
        return out


class AHCHDN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Hybrid Input Stem
        self.stem_conv = nn.Conv1d(18, 32, kernel_size=3, padding=1)

        # 2. Main Backbone
        self.backbone_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16, 32]
        curr_dim = 50  # 18 (raw) + 32 (context)
        for d in dilations:
            blk = DenseBlock(curr_dim, Config.GROWTH_RATE, d, Config.DROPOUT)
            self.backbone_blocks.append(blk)
            curr_dim += Config.GROWTH_RATE

        self.latent_proj = nn.Conv1d(curr_dim, Config.LATENT_DIM, kernel_size=1)

        # 3. Feedback Module
        self.fb_stem = nn.Conv1d(5, 16, kernel_size=3, padding=1)
        self.fb_blocks = nn.ModuleList()
        fb_dilations = [1, 2, 4, 8]
        fb_curr = 16
        for d in fb_dilations:
            blk = DenseBlock(fb_curr, Config.FEEDBACK_GROWTH_RATE, d, Config.DROPOUT)
            self.fb_blocks.append(blk)
            fb_curr += Config.FEEDBACK_GROWTH_RATE

        self.fb_proj = nn.Conv1d(fb_curr, Config.FEEDBACK_DIM, kernel_size=1)

        # 4. Aggregation
        self.rnn = nn.GRU(
            input_size=(Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

    def forward_backbone(self, x):
        # Hybrid Stem
        ctx = self.stem_conv(x)
        ctx = ctx.permute(0, 2, 1)
        ctx = F.layer_norm(ctx, ctx.shape[2:])
        ctx = F.silu(ctx)
        ctx = ctx.permute(0, 2, 1)

        h = torch.cat([x, ctx], dim=1)

        features = [h]
        for blk in self.backbone_blocks:
            out = blk(torch.cat(features, dim=1))
            features.append(out)

        z = self.latent_proj(torch.cat(features, dim=1))
        return z

    def forward_feedback(self, y_prev):
        # Mask channels: keep 0, 1, 3
        mask = torch.tensor(
            [1, 1, 0, 1, 0], device=y_prev.device, dtype=y_prev.dtype
        ).view(1, 5, 1)
        y_masked = y_prev * mask

        h = self.fb_stem(y_masked)
        h = h.permute(0, 2, 1)
        h = F.layer_norm(h, h.shape[2:])
        h = F.silu(h)
        h = h.permute(0, 2, 1)

        features = [h]
        for blk in self.fb_blocks:
            out = blk(torch.cat(features, dim=1))
            features.append(out)

        e_fb = self.fb_proj(torch.cat(features, dim=1))
        return e_fb

    def forward(self, x, pair_map, y_prev=None):
        x = x.permute(0, 2, 1)  # (N, 18, L)
        N, C, L = x.shape

        z = self.forward_backbone(x)  # (N, 64, L)

        if y_prev is None:
            y_prev = torch.zeros((N, 5, L), device=x.device, dtype=x.dtype)
        else:
            y_prev = y_prev.permute(0, 2, 1)

        e_fb = self.forward_feedback(y_prev)  # (N, 32, L)

        # Interaction
        h_self = torch.cat([z, e_fb], dim=1)  # (N, 96, L)
        h_self = h_self.permute(0, 2, 1)  # (N, L, 96)

        # Gather partner
        valid_mask = (pair_map != -1).unsqueeze(-1)
        safe_indices = pair_map.clone()
        safe_indices[safe_indices == -1] = 0

        batch_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, L)
        h_partner = h_self[batch_indices, safe_indices]
        h_partner = h_partner * valid_mask

        rnn_in = torch.cat([h_self, h_partner], dim=2)
        rnn_out, _ = self.rnn(rnn_in)
        preds = self.head(rnn_out)

        return preds


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================


def mcrmse_loss(pred, target):
    mse = F.mse_loss(pred, target, reduction="none")
    rmse = torch.sqrt(mse.mean(dim=(0, 1)))
    return rmse.mean()


def validate(model, loader, device):
    model.eval()
    total_sse = torch.zeros(5, device=device)
    total_count = 0

    with torch.no_grad():
        for features, pair_map, targets in loader:
            features = features.to(device)
            pair_map = pair_map.to(device)
            targets = targets.to(device)

            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1)

            # Score only valid positions and columns
            pred_scored = y2[:, : Config.SCORED_LENGTH, Config.SCORED_COLS_INDICES]
            targ_scored = targets[:, : Config.SCORED_LENGTH, Config.SCORED_COLS_INDICES]

            sse = ((pred_scored - targ_scored) ** 2).sum(dim=(0, 1))
            total_sse += sse
            total_count += pred_scored.shape[0] * pred_scored.shape[1]

    mse = total_sse / total_count
    rmse = torch.sqrt(mse)
    return rmse.mean().item()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = get_dataset("train", load_cached_data=True)
    val_data = get_dataset("val", load_cached_data=True)

    train_ds = RNADataset(train_data, "train")
    val_ds = RNADataset(val_data, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = AHCHDN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=Config.PATIENCE // 2
    )

    best_score = float("inf")
    best_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0

        for features, pair_map, targets in train_loader:
            features = features.to(device)
            pair_map = pair_map.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1.detach())

            # Anchored Loss on full length
            loss = mcrmse_loss(y2, targets) + 0.5 * mcrmse_loss(y1, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_score = validate(model, val_loader, device)
        scheduler.step(val_score)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss} | Val MCRMSE: {val_score}")

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break


def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AHCHDN().to(device)

    model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Model not found, skipping inference.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_data = get_dataset("test", load_cached_data=True)
    test_ds = RNADataset(test_data, "test")
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    preds_list = []
    with torch.no_grad():
        for features, pair_map in test_loader:
            features = features.to(device)
            pair_map = pair_map.to(device)

            y1 = model(features, pair_map, y_prev=None)
            y2 = model(features, pair_map, y_prev=y1)
            preds_list.append(y2.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    ids = test_data["ids"]

    submission_data = []
    for i, sample_id in enumerate(ids):
        for j in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{j}"
            row_vals = preds[i, j, :]
            row = [row_id] + row_vals.tolist()
            submission_data.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_data, columns=columns)

    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(out_path, index=False)
