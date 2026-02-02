import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache
    CACHE_VERSION = "v2_partner_identity"
    TRAIN_CACHE = os.path.join(WORKING_DIR, f"train_data_{CACHE_VERSION}.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, f"val_data_{CACHE_VERSION}.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, f"test_data_{CACHE_VERSION}.npz")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Data Dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Model Hyperparameters
    HIDDEN_DIM = 64
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    DILATION_RATES = [1, 2, 4, 8, 16, 32]

    # Training Hyperparameters
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 4
    SEED = 42

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Functions ---


def get_structure_indices(structure, seq_len):
    """
    Parses dot-bracket structure to find partner indices.
    Returns:
        partner_idx: Array of shape (L,), containing index of partner or L if unpaired.
        p_minus1_idx: Array of shape (L,), containing partner-1 or L.
        p_plus1_idx: Array of shape (L,), containing partner+1 or L.
    """
    partner = np.full(seq_len, seq_len, dtype=np.int64)  # Default to L (padding)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner[i] = j
                partner[j] = i

    # Compute neighbors of partners
    p_minus1 = np.full(seq_len, seq_len, dtype=np.int64)
    p_plus1 = np.full(seq_len, seq_len, dtype=np.int64)

    for i in range(seq_len):
        j = partner[i]
        if j != seq_len:  # If paired
            if j > 0:
                p_minus1[i] = j - 1
            if j < seq_len - 1:
                p_plus1[i] = j + 1

    return partner, p_minus1, p_plus1


def process_data(csv_path, is_test=False):
    df = pd.read_csv(csv_path)

    # Vocabularies
    seq_map = {c: i for i, c in enumerate("AGUC")}
    struct_map = {c: i for i, c in enumerate(".()")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    ids = df["id"].values
    seq_inputs = []
    struct_inputs = []
    loop_inputs = []
    partner_seq_inputs = []

    partner_indices = []
    p_minus1_indices = []
    p_plus1_indices = []

    targets = []

    for idx, row in df.iterrows():
        # Sequence
        seq_vec = [seq_map.get(c, 0) for c in row["sequence"]]
        seq_inputs.append(seq_vec)

        # Structure
        struct_vec = [struct_map.get(c, 0) for c in row["structure"]]
        struct_inputs.append(struct_vec)

        # Loop Type
        loop_vec = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loop_inputs.append(loop_vec)

        # Neighbor Indices
        p, pm1, pp1 = get_structure_indices(row["structure"], Config.SEQ_LEN)
        partner_indices.append(p)
        p_minus1_indices.append(pm1)
        p_plus1_indices.append(pp1)

        # Partner Sequence Identity (Cite Lesson 00052)
        # If paired (p[i] < SEQ_LEN), get seq_vec[p[i]]. Else -1.
        p_seq = []
        for i in range(Config.SEQ_LEN):
            idx_p = p[i]
            if idx_p < Config.SEQ_LEN:
                p_seq.append(seq_vec[idx_p])
            else:
                p_seq.append(-1)
        partner_seq_inputs.append(p_seq)

        # Targets
        if not is_test:
            # Parse stringified lists
            t_react = np.array(ast.literal_eval(row["reactivity"]), dtype=np.float32)
            t_mg_ph10 = np.array(ast.literal_eval(row["deg_Mg_pH10"]), dtype=np.float32)
            t_ph10 = np.array(ast.literal_eval(row["deg_pH10"]), dtype=np.float32)
            t_mg_50c = np.array(ast.literal_eval(row["deg_Mg_50C"]), dtype=np.float32)
            t_50c = np.array(ast.literal_eval(row["deg_50C"]), dtype=np.float32)

            # Stack: (5, 68) -> Transpose to (68, 5)
            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_stack = np.stack([t_react, t_mg_ph10, t_ph10, t_mg_50c, t_50c], axis=1)

            # Pad to 107
            pad_len = Config.SEQ_LEN - Config.PRED_LEN
            t_padded = np.pad(t_stack, ((0, pad_len), (0, 0)), constant_values=0.0)
            targets.append(t_padded)

    # Convert to arrays
    X_seq = np.array(seq_inputs, dtype=np.int64)  # (N, 107)
    X_struct = np.array(struct_inputs, dtype=np.int64)
    X_loop = np.array(loop_inputs, dtype=np.int64)

    Idx_p = np.array(partner_indices, dtype=np.int64)
    Idx_pm1 = np.array(p_minus1_indices, dtype=np.int64)
    Idx_pp1 = np.array(p_plus1_indices, dtype=np.int64)
    X_partner_seq = np.array(partner_seq_inputs, dtype=np.int64)

    if not is_test:
        Y = np.array(targets, dtype=np.float32)  # (N, 107, 5)
        return ids, X_seq, X_struct, X_loop, X_partner_seq, Idx_p, Idx_pm1, Idx_pp1, Y
    else:
        return ids, X_seq, X_struct, X_loop, X_partner_seq, Idx_p, Idx_pm1, Idx_pp1


def get_dataset(mode="train", load_cached_data=True):
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        cache_path = Config.TRAIN_CACHE
        is_test = False
    elif mode == "val":
        csv_path = Config.VAL_CSV
        cache_path = Config.VAL_CACHE
        is_test = False
    else:
        csv_path = Config.TEST_CSV
        cache_path = Config.TEST_CACHE
        is_test = True

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if is_test:
            return (
                data["ids"],
                data["X_seq"],
                data["X_struct"],
                data["X_loop"],
                data["X_partner_seq"],
                data["Idx_p"],
                data["Idx_pm1"],
                data["Idx_pp1"],
            )
        else:
            return (
                data["ids"],
                data["X_seq"],
                data["X_struct"],
                data["X_loop"],
                data["X_partner_seq"],
                data["Idx_p"],
                data["Idx_pm1"],
                data["Idx_pp1"],
                data["Y"],
            )

    print(f"Processing {mode} data from scratch...")
    result = process_data(csv_path, is_test)

    if is_test:
        ids, X_seq, X_struct, X_loop, X_partner_seq, Idx_p, Idx_pm1, Idx_pp1 = result
        np.savez(
            cache_path,
            ids=ids,
            X_seq=X_seq,
            X_struct=X_struct,
            X_loop=X_loop,
            X_partner_seq=X_partner_seq,
            Idx_p=Idx_p,
            Idx_pm1=Idx_pm1,
            Idx_pp1=Idx_pp1,
        )
    else:
        ids, X_seq, X_struct, X_loop, X_partner_seq, Idx_p, Idx_pm1, Idx_pp1, Y = result
        np.savez(
            cache_path,
            ids=ids,
            X_seq=X_seq,
            X_struct=X_struct,
            X_loop=X_loop,
            X_partner_seq=X_partner_seq,
            Idx_p=Idx_p,
            Idx_pm1=Idx_pm1,
            Idx_pp1=Idx_pp1,
            Y=Y,
        )

    return result


class RNADataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        data = get_dataset(mode)
        if mode == "test":
            (
                self.ids,
                self.X_seq,
                self.X_struct,
                self.X_loop,
                self.X_partner_seq,
                self.Idx_p,
                self.Idx_pm1,
                self.Idx_pp1,
            ) = data
            self.Y = None
        else:
            (
                self.ids,
                self.X_seq,
                self.X_struct,
                self.X_loop,
                self.X_partner_seq,
                self.Idx_p,
                self.Idx_pm1,
                self.Idx_pp1,
                self.Y,
            ) = data

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to One-Hot
        # Seq: 4, Struct: 3, Loop: 7
        seq_oh = np.eye(4)[self.X_seq[idx]]
        struct_oh = np.eye(3)[self.X_struct[idx]]
        loop_oh = np.eye(7)[self.X_loop[idx]]

        # Partner Seq: 4 (Cite Lesson 00052)
        p_seq = self.X_partner_seq[idx]
        partner_oh = np.zeros((len(p_seq), 4), dtype=np.float32)
        valid_mask = p_seq != -1
        if np.any(valid_mask):
            partner_oh[valid_mask, p_seq[valid_mask]] = 1.0

        x = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_oh], axis=1
        )  # (107, 18)
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0)  # (14, 107) for CNN

        indices = {
            "p": torch.tensor(self.Idx_p[idx], dtype=torch.long),
            "pm1": torch.tensor(self.Idx_pm1[idx], dtype=torch.long),
            "pp1": torch.tensor(self.Idx_pp1[idx], dtype=torch.long),
        }

        if self.Y is not None:
            y = torch.tensor(self.Y[idx], dtype=torch.float32)
            return x, indices, y
        else:
            return x, indices, self.ids[idx]


# --- Model ---


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(growth_rate)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(self.bn(out))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseNeighborModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 4 + 3 + 7  # 14
        growth_rate = Config.HIDDEN_DIM

        # 1. Dense Dilated Backbone
        self.blocks = nn.ModuleList()
        current_dim = input_dim

        for d in Config.DILATION_RATES:
            blk = DenseBlock(
                current_dim, growth_rate, Config.KERNEL_SIZE, d, Config.DROPOUT
            )
            self.blocks.append(blk)
            current_dim += growth_rate

        self.backbone_out_dim = current_dim

        # 2. Latent Neighbor Compression
        self.compress = nn.Conv1d(current_dim, growth_rate, 1)

        # 3. Fusion (Local + Partner + P-1 + P+1)
        # Input to GRU will be: Backbone(Local) + Compressed(Partner) + Compressed(P-1) + Compressed(P+1)
        # Dimensions: current_dim + growth_rate * 3
        self.gru_input_dim = current_dim + growth_rate * 3

        # 4. Global Aggregation
        self.gru = nn.GRU(
            self.gru_input_dim, growth_rate, batch_first=True, bidirectional=True
        )

        # 5. Head
        self.head = nn.Linear(growth_rate * 2, 5)

    def forward(self, x, indices):
        # x: (B, 14, L)

        # Backbone
        h = x
        for block in self.blocks:
            h = block(h)
        # h: (B, backbone_out_dim, L)

        # Compression for gathering
        h_small = self.compress(h)  # (B, 64, L)

        # Prepare for gathering: (B, L, 64)
        h_small_t = h_small.permute(0, 2, 1)
        B, L, C = h_small_t.shape

        # Padding for invalid indices (index L)
        # Append a zero vector at the end of sequence dimension
        padding = torch.zeros(B, 1, C, device=x.device)
        h_padded = torch.cat([h_small_t, padding], dim=1)  # (B, L+1, C)

        # Gather
        # Expand indices to (B, L, C) for gather? No, gather works on dim 1
        # Indices are (B, L). We need to gather vectors.
        # torch.gather is tricky with vectors. Use fancy indexing.

        # Flatten batch and seq for indexing? Or use batched indexing.
        # h_padded: (B, L+1, C)
        # idx: (B, L)

        def gather_features(features, idxs):
            # features: (B, L+1, C)
            # idxs: (B, L)
            # We want output (B, L, C)
            # Expand idxs to (B, L, C)
            flat_idx = idxs.unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(features, 1, flat_idx)

        h_p = gather_features(h_padded, indices["p"])
        h_pm1 = gather_features(h_padded, indices["pm1"])
        h_pp1 = gather_features(h_padded, indices["pp1"])

        # Concatenate
        # h (local) needs permute: (B, L, backbone_out_dim)
        h_local = h.permute(0, 2, 1)

        h_fused = torch.cat(
            [h_local, h_p, h_pm1, h_pp1], dim=2
        )  # (B, L, gru_input_dim)

        # GRU
        out, _ = self.gru(h_fused)  # (B, L, 2*hidden)

        # Head
        logits = self.head(out)  # (B, L, 5)

        return logits


# --- Training Logic ---


def mcrmse_loss(pred, target, mask=None):
    # pred, target: (B, L, 5)
    # Scored columns: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_cols = [0, 1, 3]

    loss = 0
    count = 0

    for col in scored_cols:
        p = pred[:, :, col]
        t = target[:, :, col]

        if mask is not None:
            # Mask is typically just seq_scored length
            # Here we assume mask is handled by slicing in the loop or pre-slicing
            pass

        # Calculate MSE per column
        mse = torch.mean((p - t) ** 2)
        loss += torch.sqrt(mse)
        count += 1

    return loss / count


def train_model():
    # Set seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Data
    train_dataset = RNADataset("train")
    val_dataset = RNADataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Model
    model = DenseNeighborModel().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for x, indices, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            indices = {k: v.to(Config.DEVICE) for k, v in indices.items()}

            # Slice to scored length for training?
            # Competition metric is on first 68.
            # We train on first 68 to match metric distribution.
            x_in = x  # Model takes full length
            y_target = y[:, : Config.PRED_LEN, :]

            optimizer.zero_grad()
            pred = model(x_in, indices)
            pred_scored = pred[:, : Config.PRED_LEN, :]

            loss = mcrmse_loss(pred_scored, y_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, indices, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                indices = {k: v.to(Config.DEVICE) for k, v in indices.items()}

                pred = model(x, indices)
                pred_scored = pred[:, : Config.PRED_LEN, :]
                y_target = y[:, : Config.PRED_LEN, :]

                loss = mcrmse_loss(pred_scored, y_target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Best Validation Loss: {best_val_loss}")
    return model


def predict(model):
    test_dataset = RNADataset("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    preds = []
    ids_list = []

    with torch.no_grad():
        for x, indices, ids in test_loader:
            x = x.to(Config.DEVICE)
            indices = {k: v.to(Config.DEVICE) for k, v in indices.items()}

            out = model(x, indices)  # (B, 107, 5)
            out = out.cpu().numpy()

            for i in range(len(ids)):
                preds.append(out[i])
                ids_list.append(ids[i])

    # Format Submission
    # Need to flatten: id_seqpos, values...
    # Columns: id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C

    submission_data = []
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for sample_idx, sample_id in enumerate(ids_list):
        sample_pred = preds[sample_idx]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_pred[seqpos]
            row_dict = {"id_seqpos": row_id}
            for i, col in enumerate(cols):
                row_dict[col] = row_vals[i]
            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    model = train_model()
    predict(model)
