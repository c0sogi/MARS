import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# ==================================================================================
# CONFIGURATION CONSTANTS
# ==================================================================================

# Paths
TRAIN_METADATA = "./metadata/train.parquet"
VAL_METADATA = "./metadata/val.parquet"
TEST_METADATA = "./metadata/test.parquet"
SAMPLE_SUBMISSION = "./input/sample_submission.csv"
WORKING_DIR = "./working/idea_37"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"

# Data Dimensions & Vocab
SEQ_LEN = 107
SEQ_SCORED = 68
VOCAB_SIZE_SEQ = 4  # A, G, C, U
VOCAB_SIZE_LOOP = 7  # S, M, I, B, H, E, X

# Mappings
TOKEN_MAP_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# Targets
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
ERROR_COLS = ["reactivity_error", "deg_error_Mg_pH10", "deg_error_Mg_50C"]
SUBMISSION_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

# Model Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 384
NUM_LAYERS = 6
DROPOUT = 0.2

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def process_structure_to_pairs(structure):
    """Parses dot-bracket structure to find pairs and computes signed distance."""
    pairs = np.zeros(len(structure), dtype=np.float32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Signed distance: j - i (positive for opening, negative for closing)
                # We assign the distance to both positions
                pairs[i] = (
                    i - j
                )  # Closing bracket: looks back (negative distance?) No, j < i. i-j is positive.
                # Let's follow the prompt: "For every nucleotide i paired with j, we encode the signed scalar distance j-i"
                # If i is '(', it pairs with j > i. j-i is positive.
                # If i is ')', it pairs with j < i. j-i is negative.
                pairs[j] = i - j  # Opening at j pairs with i. Dist = i - j (positive)
                pairs[i] = j - i  # Closing at i pairs with j. Dist = j - i (negative)
    return pairs


class RNADataset(Dataset):
    def __init__(self, df, mode="train"):
        self.mode = mode
        self.ids = df["id"].values

        # 1. Sequence Tokenization
        self.seqs = np.array(
            [[TOKEN_MAP_SEQ.get(c, 0) for c in seq] for seq in df["sequence"].values],
            dtype=np.int64,
        )

        # 2. Loop Type Tokenization
        self.loops = np.array(
            [
                [TOKEN_MAP_LOOP.get(c, 0) for c in loop]
                for loop in df["predicted_loop_type"].values
            ],
            dtype=np.int64,
        )

        # 3. Structure Pairing Distance
        self.pair_dists = np.array(
            [process_structure_to_pairs(struct) for struct in df["structure"].values],
            dtype=np.float32,
        )

        if mode in ["train", "val"]:
            # Targets
            self.targets = (
                np.vstack(df[TARGET_COLS].values.tolist())
                .reshape(-1, SEQ_SCORED, len(TARGET_COLS))
                .astype(np.float32)
            )

            # Errors (Log Transformed)
            # y_err = log(sigma + epsilon)
            raw_errors = (
                np.vstack(df[ERROR_COLS].values.tolist())
                .reshape(-1, SEQ_SCORED, len(ERROR_COLS))
                .astype(np.float32)
            )
            self.errors = np.log(np.maximum(raw_errors, 1e-6))

            # Signal to Noise (for potential filtering/analysis, though not explicitly used in model input)
            self.sn_filter = (
                df["SN_filter"].values
                if "SN_filter" in df.columns
                else np.ones(len(df))
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "seq": torch.from_numpy(self.seqs[idx]),
            "loop": torch.from_numpy(self.loops[idx]),
            "pair_dist": torch.from_numpy(self.pair_dists[idx]),
        }

        if self.mode in ["train", "val"]:
            item["target"] = torch.from_numpy(self.targets[idx])
            item["error"] = torch.from_numpy(self.errors[idx])

        return item


def get_dataset(mode="train", load_cached_data=True):
    """Loads data from Parquet or Cache."""
    cache_file = os.path.join(CACHE_DIR, f"{mode}_data.pt")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache...")
        return torch.load(cache_file)

    print(f"Processing {mode} data from scratch...")
    if mode == "train":
        df = pd.read_parquet(TRAIN_METADATA)
    elif mode == "val":
        df = pd.read_parquet(VAL_METADATA)
    else:
        df = pd.read_parquet(TEST_METADATA)

    dataset = RNADataset(df, mode=mode)

    # Save to cache
    torch.save(dataset, cache_file)
    return dataset


# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # We don't register buffer for fixed PE because inputs are variable values, not indices
        self.div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)

    def forward(self, x):
        # x: [Batch, Seq_Len] (float values of distance)
        # Output: [Batch, Seq_Len, d_model]
        batch_size, seq_len = x.shape
        x = x.unsqueeze(-1)  # [B, L, 1]
        div_term = self.div_term.to(x.device)

        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(x * div_term)
        pe[..., 1::2] = torch.cos(x * div_term)
        return pe


class BiGRUBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN
        residual = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return residual + out


class RNAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Embeddings
        self.seq_embed = nn.Embedding(VOCAB_SIZE_SEQ, EMBED_DIM)
        self.loop_embed = nn.Embedding(VOCAB_SIZE_LOOP, EMBED_DIM)
        self.dist_encoding = SinusoidalPositionalEncoding(EMBED_DIM)

        input_dim = EMBED_DIM * 3

        # Recurrent Stem
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Backbone
        self.blocks = nn.ModuleList(
            [BiGRUBlock(HIDDEN_DIM, DROPOUT) for _ in range(NUM_LAYERS)]
        )

        # Aggregation Weights (Stem + 6 Blocks = 7 outputs)
        self.mix_weights = nn.Parameter(torch.zeros(NUM_LAYERS + 1))

        # Heads
        self.value_head = nn.Linear(HIDDEN_DIM, 3)
        self.uncertainty_head = nn.Linear(HIDDEN_DIM, 3)

    def forward(self, seq, loop, pair_dist):
        # Embeddings
        x_seq = self.seq_embed(seq)
        x_loop = self.loop_embed(loop)
        x_dist = self.dist_encoding(pair_dist)

        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)

        # Stem
        x, _ = self.stem(x)  # [B, L, HIDDEN_DIM]

        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregation
        # Stack: [B, L, HIDDEN_DIM, N_LAYERS+1]
        stacked = torch.stack(outputs, dim=-1)
        weights = F.softmax(self.mix_weights, dim=0)
        aggregated = torch.sum(stacked * weights, dim=-1)

        # Heads
        values = self.value_head(aggregated)
        uncertainties = self.uncertainty_head(aggregated)

        return values, uncertainties


# ==================================================================================
# LOSS & TRAINING UTILS
# ==================================================================================


class HomoscedasticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = nn.Parameter(torch.zeros(1))
        self.s2 = nn.Parameter(torch.zeros(1))

    def forward(self, pred_val, true_val, pred_unc, true_unc):
        # Calculate MSEs
        loss_val = F.mse_loss(pred_val, true_val)
        loss_unc = F.mse_loss(pred_unc, true_unc)

        # Weighted Sum
        factor1 = torch.exp(-self.s1)
        factor2 = torch.exp(-self.s2)

        loss = (
            0.5 * factor1 * loss_val
            + 0.5 * factor2 * loss_unc
            + 0.5 * (self.s1 + self.s2)
        )
        return loss, loss_val.item(), loss_unc.item()


def mcrmse(y_true, y_pred):
    # y_true, y_pred: [N, 3] (flattened over batches and sequence length)
    colwise_mse = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.mean(np.sqrt(colwise_mse))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_mse_val = 0

    for batch in loader:
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["pair_dist"].to(device)
        target = batch["target"].to(device)
        error = batch["error"].to(device)

        optimizer.zero_grad()

        pred_val, pred_unc = model(seq, loop, dist)

        # Mask to scored positions (first 68)
        pred_val = pred_val[:, :SEQ_SCORED, :]
        pred_unc = pred_unc[:, :SEQ_SCORED, :]

        loss, mse_val, _ = criterion(pred_val, target, pred_unc, error)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse_val += mse_val

    return total_loss / len(loader), total_mse_val / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["pair_dist"].to(device)
            target = batch["target"].cpu().numpy()

            pred_val, _ = model(seq, loop, dist)
            pred_val = pred_val[:, :SEQ_SCORED, :].cpu().numpy()

            all_preds.append(pred_val)
            all_targets.append(target)

    all_preds = np.concatenate(all_preds, axis=0)  # [N_samples, 68, 3]
    all_targets = np.concatenate(all_targets, axis=0)

    # Reshape for MCRMSE calculation: [N_total_points, 3]
    flat_preds = all_preds.reshape(-1, 3)
    flat_targets = all_targets.reshape(-1, 3)

    score = mcrmse(flat_targets, flat_preds)
    return score


def predict_test(model, loader, device):
    model.eval()
    all_preds = []
    ids = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["pair_dist"].to(device)

            # Predict
            pred_val, _ = model(seq, loop, dist)  # [B, 107, 3]
            pred_val = pred_val.cpu().numpy()

            # We need to output all 107 positions, but only 3 columns are predicted.
            # The other 2 columns (deg_pH10, deg_50C) are 0.
            # Shape: [B, 107, 5]
            batch_size = pred_val.shape[0]
            full_preds = np.zeros((batch_size, SEQ_LEN, 5), dtype=np.float32)

            # Fill predicted columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
            # Map: 0->0, 1->1, 2->3
            full_preds[:, :, 0] = pred_val[:, :, 0]
            full_preds[:, :, 1] = pred_val[:, :, 1]
            full_preds[:, :, 3] = pred_val[:, :, 2]

            all_preds.append(full_preds)

            # Get IDs
            start_idx = i * loader.batch_size
            end_idx = start_idx + batch_size
            batch_ids = loader.dataset.ids[
                start_idx:end_idx
            ]  # This is a bit hacky, better to access from dataset
            # Actually, the dataset in loader is accessible
            # Correct way:
            # The Dataset class doesn't return ID in __getitem__, so we access via index range
            # Or we can just iterate the dataset ids since DataLoader is sequential (shuffle=False)
            pass

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds


# ==================================================================================
# MAIN PIPELINE
# ==================================================================================


def run_pipeline(train_model=True, generate_submission=True):
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_ds = get_dataset("train")
    val_ds = get_dataset("val")
    test_ds = get_dataset("test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Initialize Model & Loss
    model = RNAModel().to(device)
    criterion = HomoscedasticLoss().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = float("inf")

    if train_model:
        print("Starting training...")
        for epoch in range(EPOCHS):
            train_loss, train_mse = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_score = validate(model, val_loader, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
            )

            if val_score < best_score:
                best_score = val_score
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  New best model saved! Score: {best_score:.6f}")

    if generate_submission:
        print("Generating submission...")
        # Load best model
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        preds = predict_test(model, test_loader, device)  # [N_samples, 107, 5]

        # Format submission
        # id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
        submission_data = []
        test_ids = test_ds.ids

        for i, sample_id in enumerate(test_ids):
            sample_preds = preds[i]  # [107, 5]
            for j in range(SEQ_LEN):
                row_id = f"{sample_id}_{j}"
                row_vals = sample_preds[j].tolist()
                submission_data.append([row_id] + row_vals)

        sub_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + SUBMISSION_COLS)
        sub_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
