import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Data Dimensions
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Input Features
    # Seq(4) + Struct(3) + Loop(7) + Partner(5) = 19
    INPUT_CHANNELS = 19

    # Model Hyperparameters
    HIDDEN_DIM = 64
    LATENT_DIM = 32  # Dimension for compressed local/global vectors
    DROPOUT = 0.1
    DILATIONS = [1, 2, 4, 8, 16, 32]

    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Caching
    CACHE_DIR = "./working/idea_21"
    CACHE_KEY = "decoupled_dense_v1"

    # Reproducibility
    SEED = 42


# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)


# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_adj(structure, seq_len):
    """
    Parses dot-bracket structure to get partner indices.
    Returns:
        partner_indices: (seq_len, ) array where arr[i] = j if i pairs with j, else -1
    """
    partner_indices = np.full(seq_len, -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i
    return partner_indices


def one_hot_encode(seq, vocab):
    """
    One-hot encodes a sequence based on a vocabulary.
    Returns: (seq_len, len(vocab))
    """
    mapping = {char: i for i, char in enumerate(vocab)}
    seq_len = len(seq)
    vocab_size = len(vocab)
    one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def process_data(csv_path, mode="train", load_cached_data=True):
    """
    Processes data with caching mechanism.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_{Config.CACHE_KEY}.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        try:
            data = np.load(cache_file)
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Vocabularies
    SEQ_VOCAB = "AGCU"
    STRUCT_VOCAB = ".()"
    LOOP_VOCAB = "SMIBHEX"
    PARTNER_VOCAB = "AGCUN"  # N for None

    # Containers
    inputs = []
    partner_indices_list = []
    targets = []
    ids = []

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Base Features
        ohe_seq = one_hot_encode(seq, SEQ_VOCAB)
        ohe_struct = one_hot_encode(struct, STRUCT_VOCAB)
        ohe_loop = one_hot_encode(loop, LOOP_VOCAB)

        # 2. Partner Identity
        p_indices = get_structure_adj(struct, Config.SEQ_LEN)
        partner_seq = []
        for i, p_idx in enumerate(p_indices):
            if p_idx == -1:
                partner_seq.append("N")
            else:
                partner_seq.append(seq[p_idx])
        partner_seq = "".join(partner_seq)
        ohe_partner = one_hot_encode(partner_seq, PARTNER_VOCAB)

        # Concatenate Input Features
        # Shape: (SeqLen, Channels)
        sample_input = np.concatenate(
            [ohe_seq, ohe_struct, ohe_loop, ohe_partner], axis=1
        )
        inputs.append(sample_input)

        # Store partner indices for model gathering
        partner_indices_list.append(p_indices)

        ids.append(row["id"])

        # 3. Targets
        if mode in ["train", "val"]:
            # Parse stringified lists
            t_reactivity = np.array(
                ast.literal_eval(row["reactivity"]), dtype=np.float32
            )
            t_deg_Mg_pH10 = np.array(
                ast.literal_eval(row["deg_Mg_pH10"]), dtype=np.float32
            )
            t_deg_pH10 = np.array(ast.literal_eval(row["deg_pH10"]), dtype=np.float32)
            t_deg_Mg_50C = np.array(
                ast.literal_eval(row["deg_Mg_50C"]), dtype=np.float32
            )
            t_deg_50C = np.array(ast.literal_eval(row["deg_50C"]), dtype=np.float32)

            # Stack: (SeqLen, 5) - Note: raw targets are length 68
            # We need to pad to 107 for batching, though loss is masked
            sample_target = np.zeros((Config.SEQ_LEN, 5), dtype=np.float32)
            sl = Config.SCORED_LEN

            sample_target[:sl, 0] = t_reactivity
            sample_target[:sl, 1] = t_deg_Mg_pH10
            sample_target[:sl, 2] = t_deg_pH10
            sample_target[:sl, 3] = t_deg_Mg_50C
            sample_target[:sl, 4] = t_deg_50C

            targets.append(sample_target)

    inputs = np.array(inputs, dtype=np.float32)
    partner_indices_list = np.array(partner_indices_list, dtype=np.int32)
    ids = np.array(ids)

    if mode in ["train", "val"]:
        targets = np.array(targets, dtype=np.float32)
    else:
        targets = np.zeros((len(inputs), Config.SEQ_LEN, 5), dtype=np.float32)

    # Save to cache
    np.savez(
        cache_file,
        inputs=inputs,
        partner_indices=partner_indices_list,
        targets=targets,
        ids=ids,
    )
    print(f"Saved processed data to {cache_file}")

    return {
        "inputs": inputs,
        "partner_indices": partner_indices_list,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.tensor(data_dict["inputs"], dtype=torch.float32)
        self.partner_indices = torch.tensor(
            data_dict["partner_indices"], dtype=torch.long
        )
        self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Permute inputs to (Channels, SeqLen) for Conv1d
        return (
            self.inputs[idx].permute(1, 0),
            self.partner_indices[idx],
            self.targets[idx],
        )


# ==================================================================================
# MODEL
# ==================================================================================


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ScaleDecoupledCompactDenseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone: Dense Dilated TCN
        self.blocks = nn.ModuleList()
        current_dim = Config.INPUT_CHANNELS

        # Dilations: 1, 2, 4 (Local) | 8, 16, 32 (Global)
        for d in Config.DILATIONS:
            block = DenseBlock(current_dim, Config.HIDDEN_DIM, d)
            self.blocks.append(block)
            current_dim += Config.HIDDEN_DIM

        # Compression Layers (1x1 Convs)
        # Local stream: 3 blocks * 64 = 192 channels (from dilations 1, 2, 4)
        self.local_compress = nn.Conv1d(
            3 * Config.HIDDEN_DIM, Config.LATENT_DIM, kernel_size=1
        )

        # Global stream: 3 blocks * 64 = 192 channels (from dilations 8, 16, 32)
        self.global_compress = nn.Conv1d(
            3 * Config.HIDDEN_DIM, Config.LATENT_DIM, kernel_size=1
        )

        # Aggregation (BiGRU)
        # Input: Local(32) + Global(32) + PartnerLocal(32) + PartnerGlobal(32) = 128
        gru_input_dim = 4 * Config.LATENT_DIM
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=Config.HIDDEN_DIM,  # 64
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        # BiGRU output: 64 * 2 = 128
        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

    def forward(self, x, partner_indices):
        # x: (B, C, L)
        # partner_indices: (B, L)

        block_outputs = []
        current_input = x

        # 1. Backbone Pass
        for block in self.blocks:
            out = block(current_input)
            block_outputs.append(out)
            # Dense connection: concat input and output for next layer
            current_input = torch.cat([current_input, out], dim=1)

        # 2. Scale Decoupling
        # Local: blocks 0, 1, 2
        local_feats = torch.cat(block_outputs[:3], dim=1)  # (B, 192, L)
        z_local = self.local_compress(local_feats)  # (B, 32, L)

        # Global: blocks 3, 4, 5
        global_feats = torch.cat(block_outputs[3:], dim=1)  # (B, 192, L)
        z_global = self.global_compress(global_feats)  # (B, 32, L)

        # 3. Interaction (Gather Partner Features)
        # Permute to (B, L, C) for easier gathering
        z_local = z_local.permute(0, 2, 1)  # (B, L, 32)
        z_global = z_global.permute(0, 2, 1)  # (B, L, 32)

        B, L, C = z_local.shape

        # Handle unpaired indices (-1).
        # We append a zero vector at index L for each batch.
        dummy = torch.zeros(B, 1, C, device=x.device)

        z_local_padded = torch.cat([z_local, dummy], dim=1)  # (B, L+1, 32)
        z_global_padded = torch.cat([z_global, dummy], dim=1)  # (B, L+1, 32)

        # Map -1 to L
        gather_indices = partner_indices.clone()
        gather_indices[gather_indices == -1] = L

        # Expand indices for gathering: (B, L, C)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, C)

        p_local = torch.gather(z_local_padded, 1, gather_indices_expanded)
        p_global = torch.gather(z_global_padded, 1, gather_indices_expanded)

        # Fusion
        # (B, L, 128)
        fused = torch.cat([z_local, z_global, p_local, p_global], dim=2)

        # 4. Aggregation
        gru_out, _ = self.gru(fused)  # (B, L, 128)

        # 5. Head
        logits = self.head(gru_out)  # (B, L, 5)

        return logits


# ==================================================================================
# TRAINING UTILS
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    """
    Calculates MCRMSE.
    pred, target: (B, L, 5)
    We only score columns 0, 1, 3 (reactivity, deg_Mg_pH10, deg_Mg_50C)
    """
    # Select scored columns: 0, 1, 3
    scored_indices = [0, 1, 3]
    pred_scored = pred[:, :, scored_indices]
    target_scored = target[:, :, scored_indices]

    # Calculate MSE per column
    mse = torch.mean(
        (pred_scored - target_scored) ** 2, dim=(0, 1)
    )  # Mean over Batch and Length
    rmse = torch.sqrt(mse)
    return torch.mean(rmse)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for inputs, p_indices, targets in loader:
        inputs, p_indices, targets = (
            inputs.to(DEVICE),
            p_indices.to(DEVICE),
            targets.to(DEVICE),
        )

        optimizer.zero_grad()
        preds = model(inputs, p_indices)

        # Masking: Only first SCORED_LEN positions are valid
        # Apply mask implicitly by slicing
        preds_valid = preds[:, : Config.SCORED_LEN, :]
        targets_valid = targets[:, : Config.SCORED_LEN, :]

        loss = mcrmse_loss(preds_valid, targets_valid)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, p_indices, targets in loader:
            inputs, p_indices, targets = (
                inputs.to(DEVICE),
                p_indices.to(DEVICE),
                targets.to(DEVICE),
            )
            preds = model(inputs, p_indices)

            preds_valid = preds[:, : Config.SCORED_LEN, :]
            targets_valid = targets[:, : Config.SCORED_LEN, :]

            loss = mcrmse_loss(preds_valid, targets_valid)
            total_loss += loss.item()

    return total_loss / len(loader)


def predict(model, loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, p_indices, targets in loader:  # targets unused
            inputs, p_indices = inputs.to(DEVICE), p_indices.to(DEVICE)
            preds = model(inputs, p_indices)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ==================================================================================
# MAIN
# ==================================================================================


def main():
    # 1. Load Data
    train_data = process_data(Config.TRAIN_CSV, "train")
    val_data = process_data(Config.VAL_CSV, "val")
    test_data = process_data(Config.TEST_CSV, "test")

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 2. Setup Model
    model = ScaleDecoupledCompactDenseNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {DEVICE}...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, mcrmse_loss)
        val_loss = validate(model, val_loader)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # 4. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(os.path.join(Config.CACHE_DIR, "best_model.pth")))

    preds = predict(model, test_loader)  # (N, 107, 5)

    # 5. Generate Submission
    print("Generating submission file...")

    ids = test_data["ids"]
    submission_rows = []

    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(cols):
                row_dict[col_name] = row_vals[col_idx]
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
