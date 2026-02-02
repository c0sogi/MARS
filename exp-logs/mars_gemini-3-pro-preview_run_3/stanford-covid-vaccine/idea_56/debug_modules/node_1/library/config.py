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
    # Hyperparameters
    HIDDEN_DIM = 384
    NUM_LAYERS = 3
    STEM_FILTERS = 256
    KERNEL_SIZE = 3
    LEARNING_RATE = 1e-3
    GRAD_CLIP = 1.0
    BATCH_SIZE = 32
    EPOCHS = 15

    # Data Dimensions
    SEQ_LEN = 107
    SEQ_SCORED = 68
    INPUT_CHANNELS = 14  # 4 bases + 3 struct + 7 loops
    OUTPUT_DIM = 5

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    WORKING_DIR = "./working/idea_56"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Reproducibility
    SEED = 42


# ==========================================
# Data Processing & Caching
# ==========================================
class RNADataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df
        self.mode = mode

        # Precompute features
        self.sequences = df["sequence"].values
        self.structures = df["structure"].values
        self.loops = df["predicted_loop_type"].values

        if mode != "test":
            # Load targets
            self.reactivity = np.vstack(df["reactivity"].values)
            self.deg_Mg_pH10 = np.vstack(df["deg_Mg_pH10"].values)
            self.deg_pH10 = np.vstack(df["deg_pH10"].values)
            self.deg_Mg_50C = np.vstack(df["deg_Mg_50C"].values)
            self.deg_50C = np.vstack(df["deg_50C"].values)

            # Stack targets: (N, 68, 5)
            self.targets = np.stack(
                [
                    self.reactivity,
                    self.deg_Mg_pH10,
                    self.deg_pH10,
                    self.deg_Mg_50C,
                    self.deg_50C,
                ],
                axis=2,
            ).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Feature Extraction
        seq = self.sequences[idx]
        struct = self.structures[idx]
        loop = self.loops[idx]

        # One-hot encoding
        # Sequence: A, G, C, U -> 0, 1, 2, 3
        seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
        seq_enc = np.zeros((Config.SEQ_LEN, 4), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in seq_map:
                seq_enc[i, seq_map[char]] = 1.0

        # Structure: ., (, ) -> 0, 1, 2
        struct_map = {".": 0, "(": 1, ")": 2}
        struct_enc = np.zeros((Config.SEQ_LEN, 3), dtype=np.float32)
        pair_index = np.full(Config.SEQ_LEN, -1, dtype=np.int64)
        stack = []

        for i, char in enumerate(struct):
            if char in struct_map:
                struct_enc[i, struct_map[char]] = 1.0

            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    start = stack.pop()
                    pair_index[start] = i
                    pair_index[i] = start

        # Loop: S, M, I, B, H, E, X -> 0..6
        loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
        loop_enc = np.zeros((Config.SEQ_LEN, 7), dtype=np.float32)
        for i, char in enumerate(loop):
            if char in loop_map:
                loop_enc[i, loop_map[char]] = 1.0

        # Concatenate features: (107, 14)
        features = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        item = {
            "features": torch.tensor(features, dtype=torch.float32),
            "pair_index": torch.tensor(pair_index, dtype=torch.long),
        }

        if self.mode != "test":
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_data_loader(df, mode, batch_size, shuffle=False):
    dataset = RNADataset(df, mode)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )


def load_and_cache_data(load_cached_data=True):
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_cache.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_cache.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

        # Save to cache
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


# ==========================================
# Model Architecture: SDBR-BiGRU
# ==========================================


class DecoupledInteractionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message generation: W_msg * h_j + b_msg
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Gate: MLP on [h_i; h_j]
        # z_raw = W_g1 [h_i; h_j]
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Norm
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_index):
        # h: (B, L, H)
        # pair_index: (B, L) containing indices of pairs. -1 if unpaired.

        B, L, H = h.shape

        # 1. Gather h_j
        # Handle -1 indices by clamping to 0, then masking result
        mask = (pair_index != -1).unsqueeze(-1)  # (B, L, 1)
        safe_indices = pair_index.clone()
        safe_indices[safe_indices == -1] = 0

        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, H)  # (B, L, H)
        h_j = torch.gather(h, 1, gather_indices)  # (B, L, H)

        # 2. Input Zero-Masking
        # If unpaired, h_j should be 0.
        h_j = h_j * mask  # (B, L, H)

        # 3. Decoupled Message (Bias-Refined)
        # m_ij = GELU(W_msg * h_j + b_msg)
        # Note: If h_j is 0 (unpaired), this becomes GELU(b_msg), a learnable loop embedding.
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Stabilized MLP Gate
        # z_raw = W_g1 [h_i; h_j]
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)
        z_raw = self.gate_proj1(cat_input)

        # Internal Normalization
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Logit Projection & Sigmoid (No Logit Norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class SDBR_BiGRU(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                Config.INPUT_CHANNELS,
                Config.STEM_FILTERS,
                kernel_size=Config.KERNEL_SIZE,
                padding=Config.KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # Backbone
        self.layers = nn.ModuleList()
        input_dim = Config.STEM_FILTERS

        for i in range(Config.NUM_LAYERS):
            # BiGRU
            gru = nn.GRU(
                input_dim, Config.HIDDEN_DIM // 2, batch_first=True, bidirectional=True
            )

            # Interaction
            interaction = DecoupledInteractionModule(Config.HIDDEN_DIM)

            self.layers.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))
            input_dim = Config.HIDDEN_DIM

        # Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.OUTPUT_DIM)

    def forward(self, features, pair_index):
        # features: (B, L, 14) -> Transpose for Conv1d (B, 14, L)
        x = features.transpose(1, 2)
        x = self.stem(x)
        x = x.transpose(1, 2)  # (B, L, C)

        for layer in self.layers:
            gru = layer["gru"]
            interaction = layer["interaction"]

            # GRU
            x, _ = gru(x)  # (B, L, H)

            # Interaction
            x = interaction(x, pair_index)

        out = self.head(x)  # (B, L, 5)
        return out


# ==========================================
# Training & Evaluation
# ==========================================


def mcrmse_loss(pred, target):
    # pred, target: (B, 68, 5)
    pred_flat = pred.view(-1, 5)
    target_flat = target.view(-1, 5)

    mse = torch.mean((pred_flat - target_flat) ** 2, dim=0)  # (5,)
    rmse = torch.sqrt(mse)
    return torch.mean(rmse)


def train_pipeline(epochs=Config.EPOCHS, debug=False):
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)

    # Load Data
    train_df, val_df, test_df = load_and_cache_data()

    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        epochs = 2

    train_loader = get_data_loader(train_df, "train", Config.BATCH_SIZE, shuffle=True)
    val_loader = get_data_loader(val_df, "val", Config.BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDBR_BiGRU().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = float("inf")
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)  # (B, 68, 5)

            optimizer.zero_grad()
            preds = model(features, pair_index)  # (B, 107, 5)

            # Slice to scored positions
            preds_scored = preds[:, : Config.SEQ_SCORED, :]

            loss = mcrmse_loss(preds_scored, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                pair_index = batch["pair_index"].to(device)
                targets = batch["targets"].to(device)

                preds = model(features, pair_index)
                preds_scored = preds[:, : Config.SEQ_SCORED, :]

                val_preds_list.append(preds_scored.cpu())
                val_targets_list.append(targets.cpu())

        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Calculate Metric on 3 scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        scored_indices = [0, 1, 3]
        val_preds_filtered = val_preds[:, :, scored_indices]
        val_targets_filtered = val_targets[:, :, scored_indices]

        val_score = mcrmse_loss(val_preds_filtered, val_targets_filtered).item()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        if val_score < best_score:
            best_score = val_score
            best_model_state = model.state_dict()
            torch.save(
                best_model_state, os.path.join(Config.WORKING_DIR, "best_model.pth")
            )

    print(f"Training finished. Best Val MCRMSE: {best_score:.10f}")
    return best_model_state


def generate_submission(model_state, debug=False):
    # Load Data
    _, _, test_df = load_and_cache_data()
    if debug:
        test_df = test_df.iloc[:20]

    test_loader = get_data_loader(test_df, "test", Config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDBR_BiGRU().to(device)
    model.load_state_dict(model_state)
    model.eval()

    all_preds = []
    ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)

            preds = model(features, pair_index)  # (B, 107, 5)
            all_preds.append(preds.cpu().numpy())

            batch_start = i * Config.BATCH_SIZE
            batch_end = batch_start + features.size(0)
            batch_ids = test_df["id"].iloc[batch_start:batch_end].values
            ids.extend(batch_ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 5)

    # Format Submission
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]
            row_dict = {"id_seqpos": row_id}
            for j, col in enumerate(target_cols):
                row_dict[col] = float(row_vals[j])
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
