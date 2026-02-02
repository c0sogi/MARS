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
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68

    # Model Hyperparameters
    GROWTH_RATE = 64
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1
    LATENT_DIM = 128

    # Input Channels
    # Seq(4) + Struct(3) + Loop(7) + PartnerId(5) + Recycling(5)
    NUM_INPUT_CHANNELS = 4 + 3 + 7 + 5 + 5
    NUM_TARGETS = 5

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    PATIENCE = 5

    # Paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Scored: reactivity, deg_Mg_pH10, deg_Mg_50C
    SCORED_INDICES = [0, 1, 3]


# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_map(structure):
    partner_map = np.full(len(structure), -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i
    return partner_map


def one_hot(seq, vocab):
    res = np.zeros((len(seq), len(vocab)), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in vocab:
            res[i, vocab[char]] = 1.0
    return res


def process_data(df, mode="train"):
    seq_vocab = {"A": 0, "G": 1, "U": 2, "C": 3}
    struct_vocab = {"(": 0, ")": 1, ".": 2}
    loop_vocab = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = []
    inputs = []
    partner_indices = []
    targets = []

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        oh_seq = one_hot(seq, seq_vocab)
        oh_struct = one_hot(struct, struct_vocab)
        oh_loop = one_hot(loop, loop_vocab)

        pmap = get_structure_map(struct)
        partner_indices.append(pmap)

        partner_id_feat = np.zeros((len(seq), 5), dtype=np.float32)
        for i, p_idx in enumerate(pmap):
            if p_idx != -1:
                base = seq[p_idx]
                if base in seq_vocab:
                    partner_id_feat[i, seq_vocab[base]] = 1.0
            else:
                partner_id_feat[i, 4] = 1.0

        sample_input = np.concatenate(
            [oh_seq, oh_struct, oh_loop, partner_id_feat], axis=1
        )
        inputs.append(sample_input)
        ids.append(row["id"])

        if mode != "test":
            t_list = []
            for col in Config.TARGET_COLS:
                val = row[col]
                if isinstance(val, str):
                    val = ast.literal_eval(val)
                arr = np.array(val, dtype=np.float32)
                padded = np.zeros(Config.SEQ_LENGTH, dtype=np.float32)
                padded[: len(arr)] = arr
                t_list.append(padded)

            sample_target = np.stack(t_list, axis=1)
            targets.append(sample_target)

    inputs = np.array(inputs, dtype=np.float32)
    partner_indices = np.array(partner_indices, dtype=np.int64)

    if mode != "test":
        targets = np.array(targets, dtype=np.float32)
        return ids, inputs, partner_indices, targets
    else:
        return ids, inputs, partner_indices


def get_dataset(mode="train", load_cached_data=True):
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_data_v1.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            if mode == "test":
                return data["ids"], data["inputs"], data["partner_indices"]
            else:
                return (
                    data["ids"],
                    data["inputs"],
                    data["partner_indices"],
                    data["targets"],
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing {mode} data...")
    df = pd.read_csv(os.path.join(Config.METADATA_DIR, f"{mode}.csv"))

    if mode == "test":
        ids, inputs, p_indices = process_data(df, mode)
        np.savez(cache_file, ids=ids, inputs=inputs, partner_indices=p_indices)
        return ids, inputs, p_indices
    else:
        ids, inputs, p_indices, targets = process_data(df, mode)
        np.savez(
            cache_file,
            ids=ids,
            inputs=inputs,
            partner_indices=p_indices,
            targets=targets,
        )
        return ids, inputs, p_indices, targets


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        pmap = torch.tensor(self.partner_indices[idx], dtype=torch.long)
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, pmap, y
        return x, pmap


# ==================================================================================
# MODEL
# ==================================================================================


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        out = F.relu(self.bn(x))
        out = self.conv(out)
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class SRDN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 19 static + 5 recycling = 24
        self.in_channels = 19 + 5
        self.stem = nn.Conv1d(self.in_channels, Config.GROWTH_RATE, kernel_size=1)

        self.blocks = nn.ModuleList()
        curr_channels = Config.GROWTH_RATE
        for d in Config.DILATIONS:
            blk = DenseBlock(curr_channels, Config.GROWTH_RATE, d)
            self.blocks.append(blk)
            curr_channels += Config.GROWTH_RATE

        self.to_latent = nn.Conv1d(curr_channels, Config.LATENT_DIM, kernel_size=1)
        self.gru = nn.GRU(
            input_size=Config.LATENT_DIM * 2,
            hidden_size=Config.LATENT_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(Config.LATENT_DIM * 2, Config.NUM_TARGETS)

    def forward(self, x, partner_indices):
        x = x.permute(0, 2, 1)  # (B, C, L)
        feat = self.stem(x)
        for block in self.blocks:
            feat = block(feat)

        latent = self.to_latent(feat)  # (B, 128, L)
        latent = latent.permute(0, 2, 1)  # (B, L, 128)
        B, L, C = latent.shape

        p_idx_clamped = partner_indices.clone()
        p_idx_clamped[p_idx_clamped == -1] = 0
        gather_idx = p_idx_clamped.unsqueeze(-1).expand(-1, -1, C)
        partner_feat = torch.gather(latent, 1, gather_idx)

        mask = (partner_indices != -1).unsqueeze(-1).float()
        partner_feat = partner_feat * mask

        fused = torch.cat([latent, partner_feat], dim=2)
        gru_out, _ = self.gru(fused)
        out = self.head(gru_out)
        return out


# ==================================================================================
# METRICS & LOSS
# ==================================================================================


def mcrmse_loss(pred, target, mask):
    # pred, target: (B, L, 5)
    # mask: (B, L)
    pred_scored = pred[:, :, Config.SCORED_INDICES]
    target_scored = target[:, :, Config.SCORED_INDICES]
    mse = (pred_scored - target_scored) ** 2

    mask = mask.unsqueeze(-1)  # (B, L, 1)
    mse = mse * mask

    # Sum errors and count valid pixels
    loss_sum = torch.sum(mse, dim=(0, 1))  # (3,)
    valid_count = torch.sum(mask)  # Total valid positions in batch

    # RMSE = sqrt(sum / count)
    rmse = torch.sqrt(loss_sum / valid_count)
    return torch.mean(rmse)


def validate(model, loader):
    model.eval()
    total_sse = torch.zeros(3, device=device)
    total_count = 0

    mask = torch.zeros(Config.SEQ_LENGTH, device=device)
    mask[: Config.SCORED_LENGTH] = 1.0

    with torch.no_grad():
        for x, pmap, y in loader:
            x, pmap, y = x.to(device), pmap.to(device), y.to(device)
            B = x.shape[0]

            # Pass 1
            recycling = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
            x_in = torch.cat([x, recycling], dim=2)
            pred1 = model(x_in, pmap)

            # Pass 2
            recycling = pred1.detach()
            x_in = torch.cat([x, recycling], dim=2)
            pred2 = model(x_in, pmap)

            pred_scored = pred2[:, :, Config.SCORED_INDICES]
            target_scored = y[:, :, Config.SCORED_INDICES]

            sq_diff = (pred_scored - target_scored) ** 2
            sq_diff = sq_diff * mask.view(1, -1, 1)

            total_sse += torch.sum(sq_diff, dim=(0, 1))
            total_count += B * Config.SCORED_LENGTH

    rmse_per_col = torch.sqrt(total_sse / total_count)
    mcrmse = torch.mean(rmse_per_col).item()
    return mcrmse


# ==================================================================================
# EXECUTION
# ==================================================================================


def run_pipeline():
    # Load Data
    train_ids, train_inputs, train_pmaps, train_targets = get_dataset("train")
    val_ids, val_inputs, val_pmaps, val_targets = get_dataset("val")
    test_ids, test_inputs, test_pmaps = get_dataset("test")

    train_dataset = RNADataset(train_inputs, train_pmaps, train_targets)
    val_dataset = RNADataset(val_inputs, val_pmaps, val_targets)
    test_dataset = RNADataset(test_inputs, test_pmaps)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Model
    model = SRDN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=Config.PATIENCE
    )

    loss_mask = torch.zeros(Config.SEQ_LENGTH, device=device)
    loss_mask[: Config.SCORED_LENGTH] = 1.0

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0

        for x, pmap, y in train_loader:
            x, pmap, y = x.to(device), pmap.to(device), y.to(device)
            B = x.shape[0]

            # Pass 1
            recycling_zero = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
            x1 = torch.cat([x, recycling_zero], dim=2)
            pred1 = model(x1, pmap)

            # Pass 2
            recycling_detached = pred1.detach()
            x2 = torch.cat([x, recycling_detached], dim=2)
            pred2 = model(x2, pmap)

            loss2 = mcrmse_loss(pred2, y, loss_mask.unsqueeze(0).expand(B, -1))
            loss1 = mcrmse_loss(pred1, y, loss_mask.unsqueeze(0).expand(B, -1))
            loss = loss2 + 0.5 * loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * B

        avg_train_loss = train_loss_accum / len(train_dataset)
        val_mcrmse = validate(model, val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Val MCRMSE: {best_mcrmse}")

    # Inference
    print("Generating Submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds_list = []

    with torch.no_grad():
        for x, pmap in test_loader:
            x, pmap = x.to(device), pmap.to(device)
            B = x.shape[0]

            recycling = torch.zeros(B, Config.SEQ_LENGTH, 5, device=device)
            x_in = torch.cat([x, recycling], dim=2)
            pred1 = model(x_in, pmap)

            recycling = pred1.detach()
            x_in = torch.cat([x, recycling], dim=2)
            pred2 = model(x_in, pmap)

            preds_list.append(pred2.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0)

    sub_ids = []
    sub_preds = []

    for i, sample_id in enumerate(test_ids):
        for pos in range(Config.SEQ_LENGTH):
            sub_ids.append(f"{sample_id}_{pos}")
            sub_preds.append(all_preds[i, pos])

    sub_preds = np.array(sub_preds)
    submission_df = pd.DataFrame(sub_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", sub_ids)

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


run_pipeline()
