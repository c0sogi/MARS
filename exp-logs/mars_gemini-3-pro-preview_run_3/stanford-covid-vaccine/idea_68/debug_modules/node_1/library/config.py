import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ==========================================
# Configuration
# ==========================================
class Config:
    # Directories
    TRAIN_META = "./metadata/train.parquet"
    VAL_META = "./metadata/val.parquet"
    TEST_META = "./metadata/test.parquet"
    CACHE_DIR = "./working/idea_68/"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data Dimensions
    SEQ_LEN = 107
    SEQ_SCORED = 68
    INPUT_CHANNELS = 14  # 4 (seq) + 3 (struct) + 7 (loop)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Model Hyperparameters
    STEM_FILTERS = 256
    HIDDEN_DIM = 384  # Dimension per direction
    TOTAL_HIDDEN = 768  # 384 * 2 (Bidirectional)
    LAYERS = 4

    # Training Settings
    SEED = 42
    EPOCHS = 25
    BATCH_SIZE = 32
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    PATIENCE = 5
    NUM_WORKERS = 4


# ==========================================
# Utils & Data Processing
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_pair_index_and_mask(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns:
        indices: Array where indices[i] = j if (i, j) are paired.
        mask: Array where mask[i] = 1 if paired, 0 if unpaired.
    """
    L = len(structure)
    indices = np.zeros(L, dtype=np.int64)
    mask = np.zeros(L, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
    return indices, mask


def one_hot_encode(seq, struct, loop):
    """
    One-hot encodes Sequence, Structure, and Loop Type.
    """
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    L = len(seq)
    enc = np.zeros((L, 14), dtype=np.float32)

    for i in range(L):
        if seq[i] in seq_map:
            enc[i, seq_map[seq[i]]] = 1.0
        if struct[i] in struct_map:
            enc[i, 4 + struct_map[struct[i]]] = 1.0
        if loop[i] in loop_map:
            enc[i, 7 + loop_map[loop[i]]] = 1.0

    return enc


def process_dataframe(df, cache_name, load_cached_data=True):
    """
    Processes dataframe into numpy arrays with caching.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["features"],
            data["pair_indices"],
            data["pair_masks"],
            data["targets"],
            data["ids"],
        )

    print(f"Processing data for {cache_name}...")
    features = []
    pair_indices = []
    pair_masks = []
    targets = []
    ids = df["id"].values

    for idx, row in df.iterrows():
        # Features
        feat = one_hot_encode(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        features.append(feat)

        # Pairs
        p_idx, p_mask = get_pair_index_and_mask(row["structure"])
        pair_indices.append(p_idx)
        pair_masks.append(p_mask)

        # Targets
        if "reactivity" in row:
            t = np.zeros((Config.SEQ_LEN, 5), dtype=np.float32)
            for k, col in enumerate(Config.TARGET_COLS):
                val = row[col]
                length = len(val)
                t[:length, k] = val
            targets.append(t)
        else:
            targets.append(np.zeros((Config.SEQ_LEN, 5), dtype=np.float32))

    features = np.array(features, dtype=np.float32)
    pair_indices = np.array(pair_indices, dtype=np.int64)
    pair_masks = np.array(pair_masks, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    np.savez(
        cache_path,
        features=features,
        pair_indices=pair_indices,
        pair_masks=pair_masks,
        targets=targets,
        ids=ids,
    )
    return features, pair_indices, pair_masks, targets, ids


class RNADataset(Dataset):
    def __init__(self, features, pair_indices, pair_masks, targets, ids):
        self.features = torch.from_numpy(features)
        self.pair_indices = torch.from_numpy(pair_indices)
        self.pair_masks = torch.from_numpy(pair_masks)
        self.targets = torch.from_numpy(targets)
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_masks": self.pair_masks[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


# ==========================================
# Model Architecture
# ==========================================
class GLUInteraction(nn.Module):
    """
    GLU-Decoupled Structural Injection Module with Deep Stabilized Gate.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Deep Stabilized MLP Gate
        # Input: [h_i; h_j] -> LayerNorm -> GELU -> Linear -> Sigmoid
        self.gate_norm = nn.LayerNorm(2 * hidden_dim)
        self.gate_w1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_w2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_masks):
        B, L, H = h.shape

        # Gather h_j (Neighbor state)
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, L)
        h_j = h[batch_idx, pair_indices]  # (B, L, H)

        # Mask h_j: Force to 0 if unpaired
        h_j = h_j * pair_masks.unsqueeze(-1)

        # 1. GLU Message Generation
        # If h_j is 0 (unpaired), this becomes bias_c * sigma(bias_g) (Bias-Driven Refinement)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # 2. Deep Stabilized Gate Calculation
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)
        z = self.gate_norm(cat_input)
        z = F.gelu(self.gate_w1(z))
        g_ij = torch.sigmoid(self.gate_w2(z))

        # 3. Injection & Normalization
        h_res = h + g_ij * m_ij
        h_out = self.out_norm(h_res)

        return h_out


class HighCapacityModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Convolutional Stem
        self.stem_conv = nn.Conv1d(
            config.INPUT_CHANNELS, config.STEM_FILTERS, kernel_size=3, padding=1
        )
        self.stem_act = nn.GELU()

        # Backbone: 4 Blocks
        self.blocks = nn.ModuleList()
        curr_dim = config.STEM_FILTERS

        for i in range(config.LAYERS):
            # BiGRU
            gru = nn.GRU(
                input_size=curr_dim,
                hidden_size=config.HIDDEN_DIM,
                batch_first=True,
                bidirectional=True,
            )
            curr_dim = config.TOTAL_HIDDEN  # 768

            # Interaction Module (Layers 1, 2, 3 only)
            interaction = None
            if i < config.LAYERS - 1:
                interaction = GLUInteraction(curr_dim)

            self.blocks.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))

        # Output Head
        self.head = nn.Linear(config.TOTAL_HIDDEN, 5)

    def forward(self, x, pair_indices, pair_masks):
        # x: (B, L, 14) -> Conv1d needs (B, 14, L)
        x = x.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = x.permute(0, 2, 1)  # (B, L, C)

        h = x
        for block in self.blocks:
            h, _ = block["gru"](h)
            if block["interaction"] is not None:
                h = block["interaction"](h, pair_indices, pair_masks)

        out = self.head(h)
        return out


# ==========================================
# Training & Execution
# ==========================================
def mcrmse_loss(pred, target):
    """
    Mean Columnwise Root Mean Squared Error.
    Averages over batch and sequence length first, then takes sqrt, then averages over columns.
    """
    mse = (pred - target) ** 2
    loss_per_col = torch.sqrt(mse.mean(dim=(0, 1)))
    return loss_per_col.mean()


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            p_idx = batch["pair_indices"].to(device)
            p_mask = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features, p_idx, p_mask)

            # Slice to scored length (68)
            preds = preds[:, : Config.SEQ_SCORED, :]
            targets = targets[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE for specific scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    mse = (all_preds - all_targets) ** 2
    rmse_per_col = torch.sqrt(mse.mean(dim=(0, 1)))
    selected_rmse = rmse_per_col[[0, 1, 3]]

    return selected_rmse.mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        features = batch["features"].to(device)
        p_idx = batch["pair_indices"].to(device)
        p_mask = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        preds = model(features, p_idx, p_mask)

        # Loss on all 5 targets, sliced to 68 positions
        preds_scored = preds[:, : Config.SEQ_SCORED, :]
        targets_scored = targets[:, : Config.SEQ_SCORED, :]

        loss = mcrmse_loss(preds_scored, targets_scored)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def run_training():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_df = pd.read_parquet(Config.TRAIN_META)
    val_df = pd.read_parquet(Config.VAL_META)

    t_feat, t_pidx, t_pmask, t_targ, t_ids = process_dataframe(train_df, "train_data")
    v_feat, v_pidx, v_pmask, v_targ, v_ids = process_dataframe(val_df, "val_data")

    train_ds = RNADataset(t_feat, t_pidx, t_pmask, t_targ, t_ids)
    val_ds = RNADataset(v_feat, v_pidx, v_pmask, v_targ, v_ids)

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

    # Initialize Model
    model = HighCapacityModel(Config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mcrmse = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")
    return model


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = HighCapacityModel(Config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CACHE_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    # Load Test Data
    test_df = pd.read_parquet(Config.TEST_META)
    te_feat, te_pidx, te_pmask, te_targ, te_ids = process_dataframe(
        test_df, "test_data"
    )
    test_ds = RNADataset(te_feat, te_pidx, te_pmask, te_targ, te_ids)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            p_idx = batch["pair_indices"].to(device)
            p_mask = batch["pair_masks"].to(device)
            ids = batch["id"]

            preds = model(features, p_idx, p_mask)  # (B, 107, 5)
            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)

    # Format submission
    submission_data = []
    for i, sample_id in enumerate(all_ids):
        preds_i = all_preds[i]  # (107, 5)
        for pos in range(Config.SEQ_LEN):
            id_seqpos = f"{sample_id}_{pos}"
            row = [id_seqpos] + preds_i[pos].tolist()
            submission_data.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_data, columns=columns)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_training()
    generate_submission()
