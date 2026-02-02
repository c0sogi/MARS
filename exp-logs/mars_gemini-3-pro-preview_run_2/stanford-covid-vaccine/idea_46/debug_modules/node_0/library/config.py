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
    # Paths
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_46"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data
    SEQ_LEN = 107
    PRED_LEN = 68
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    TARGET_INDICES = [0, 1, 3]  # Indices of scored targets in ALL_TARGETS

    # Model Architecture
    EMBED_DIM = 32
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_LAYERS = [1, 2, 4, 8, 16, 32]  # Dilation rates
    LATENT_DIM = 64

    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_LAYERS = [1, 2, 4]  # Smaller feedback loop
    FEEDBACK_DIM = 32

    RNN_HIDDEN = 64
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 50
    PATIENCE = 7  # Early stopping
    SEED = 42
    NUM_WORKERS = 4


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed(Config.SEED)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_adj(structure, seq_len):
    """Parses dot-bracket structure to get partner indices."""
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


def parse_list_col(x):
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except:
        return np.zeros(Config.PRED_LEN, dtype=np.float32)


def process_data(mode="train", load_cached_data=True):
    """
    Loads metadata, processes sequences/structures, and caches the result.
    mode: 'train', 'val', or 'test'
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data_sdf_rn_v1.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        return np.load(cache_file, allow_pickle=True)

    print(f"Processing {mode} data from scratch...")
    df = pd.read_csv(os.path.join(Config.METADATA_DIR, f"{mode}.csv"))

    # Mappings
    seq_map = {c: i for i, c in enumerate("AGUC")}
    struct_map = {c: i for i, c in enumerate(".()")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    # Containers
    ids = df["id"].values
    seq_encoded = []
    struct_encoded = []
    loop_encoded = []
    partner_indices = []
    partner_identities = []  # Identity of the paired base

    targets = []

    for idx, row in df.iterrows():
        # Sequence
        seq_ints = [seq_map.get(c, 0) for c in row["sequence"]]
        seq_encoded.append(seq_ints)

        # Structure
        struct_ints = [struct_map.get(c, 0) for c in row["structure"]]
        struct_encoded.append(struct_ints)

        # Loop
        loop_ints = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]
        loop_encoded.append(loop_ints)

        # Adjacency / Partner
        adj = get_structure_adj(row["structure"], Config.SEQ_LEN)
        partner_indices.append(adj)

        # Partner Identity
        # If i is paired with j, identity is seq[j]. If unpaired, use a special token (4) or just 0 (A) masked later.
        # We'll use 4 as "No Partner"
        p_ids = []
        for i, neighbor in enumerate(adj):
            if neighbor != -1:
                p_ids.append(seq_ints[neighbor])
            else:
                p_ids.append(4)  # 4 = None
        partner_identities.append(p_ids)

        # Targets
        if mode != "test":
            t_list = []
            for col in Config.ALL_TARGETS:
                val = parse_list_col(row[col])
                # Pad to SEQ_LEN (though we only score first 68, model outputs 107)
                padded = np.zeros(Config.SEQ_LEN, dtype=np.float32)
                padded[: len(val)] = val
                t_list.append(padded)
            targets.append(np.stack(t_list, axis=1))  # (L, 5)

    # Convert to numpy
    data = {
        "ids": ids,
        "seq": np.array(seq_encoded, dtype=np.int32),
        "struct": np.array(struct_encoded, dtype=np.int32),
        "loop": np.array(loop_encoded, dtype=np.int32),
        "partner_idx": np.array(partner_indices, dtype=np.int32),
        "partner_id": np.array(partner_identities, dtype=np.int32),
    }

    if mode != "test":
        data["targets"] = np.array(targets, dtype=np.float32)

    np.savez_compressed(cache_file, **data)
    return data


# ==================================================================================
# DATASET
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data["ids"])

    def __getitem__(self, idx):
        item = {
            "seq": torch.tensor(self.data["seq"][idx], dtype=torch.long),
            "struct": torch.tensor(self.data["struct"][idx], dtype=torch.long),
            "loop": torch.tensor(self.data["loop"][idx], dtype=torch.long),
            "partner_idx": torch.tensor(
                self.data["partner_idx"][idx], dtype=torch.long
            ),
            "partner_id": torch.tensor(self.data["partner_id"][idx], dtype=torch.long),
        }

        if self.mode != "test":
            item["targets"] = torch.tensor(
                self.data["targets"][idx], dtype=torch.float32
            )

        return item


# ==================================================================================
# MODEL: SDF-RN
# ==================================================================================


class DenseLayer(nn.Module):
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
            nn.LayerNorm(
                Config.SEQ_LEN
            ),  # Applied to last dim, so we need permute in forward
            nn.SiLU(),
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            nn.LayerNorm(Config.SEQ_LEN),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
        )

    def forward(self, x):
        # x: (B, C, L)
        out = self.net[0](x)
        out = out.permute(0, 2, 1)  # (B, L, C)
        out = self.net[1](out)
        out = self.net[2](out)
        out = out.permute(0, 2, 1)  # (B, C, L)

        out = self.net[3](out)
        out = out.permute(0, 2, 1)
        out = self.net[4](out)
        out = self.net[5](out)
        out = out.permute(0, 2, 1)

        out = self.net[6](out)
        return torch.cat([x, out], dim=1)


class DenseTCN(nn.Module):
    def __init__(self, in_channels, growth_rate, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        c_in = in_channels
        for d in dilations:
            self.layers.append(DenseLayer(c_in, growth_rate, d))
            c_in += growth_rate
        self.out_channels = c_in

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SDFRN(nn.Module):
    def __init__(self):
        super().__init__()

        # Embeddings
        self.seq_emb = nn.Embedding(4, Config.EMBED_DIM)
        self.struct_emb = nn.Embedding(3, Config.EMBED_DIM)
        self.loop_emb = nn.Embedding(7, Config.EMBED_DIM)
        self.partner_emb = nn.Embedding(5, Config.EMBED_DIM)  # 4 bases + 1 none

        # Backbone (Static)
        in_dim = Config.EMBED_DIM * 4
        self.backbone = DenseTCN(
            in_dim, Config.BACKBONE_GROWTH_RATE, Config.BACKBONE_LAYERS
        )
        self.to_latent = nn.Conv1d(self.backbone.out_channels, Config.LATENT_DIM, 1)

        # Feedback Module (Dynamic)
        # Input: 5 (targets) + 3 (struct) + 7 (loop) = 15
        # We embed struct/loop again or reuse embeddings? Let's use one-hots for raw topology as described.
        # Actually, using embeddings is better for dimensionality.
        fb_in_dim = 5 + Config.EMBED_DIM * 2
        self.feedback_net = DenseTCN(
            fb_in_dim, Config.FEEDBACK_GROWTH_RATE, Config.FEEDBACK_LAYERS
        )
        self.to_fb_emb = nn.Conv1d(
            self.feedback_net.out_channels, Config.FEEDBACK_DIM, 1
        )

        # Aggregation
        # Input to RNN: (Latent + FB) * 2 (Self + Partner)
        rnn_in_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2
        self.rnn = nn.GRU(
            rnn_in_dim, Config.RNN_HIDDEN, batch_first=True, bidirectional=True
        )

        # Head
        self.head = nn.Linear(Config.RNN_HIDDEN * 2, 5)

    def forward(self, seq, struct, loop, partner_id, partner_idx, prev_pred=None):
        # 1. Static Representation
        # (B, L, Emb)
        x_seq = self.seq_emb(seq)
        x_struct = self.struct_emb(struct)
        x_loop = self.loop_emb(loop)
        x_part = self.partner_emb(partner_id)

        # (B, C, L)
        x_static = torch.cat([x_seq, x_struct, x_loop, x_part], dim=-1).permute(0, 2, 1)

        # Run Backbone
        z = self.backbone(x_static)
        z = self.to_latent(z)  # (B, Latent, L)

        # 2. Feedback Representation
        if prev_pred is None:
            prev_pred = torch.zeros(seq.size(0), 5, Config.SEQ_LEN, device=seq.device)
        else:
            # Ensure shape (B, 5, L)
            if prev_pred.shape[-1] != Config.SEQ_LEN:
                prev_pred = prev_pred.permute(0, 2, 1)

        # Concat prev_pred with raw topology (embeddings)
        # (B, C, L)
        fb_in = torch.cat(
            [prev_pred, x_struct.permute(0, 2, 1), x_loop.permute(0, 2, 1)], dim=1
        )

        e_fb = self.feedback_net(fb_in)
        e_fb = self.to_fb_emb(e_fb)  # (B, FB_Dim, L)

        # 3. Interaction & Aggregation
        # Combine Z and E_fb -> (B, L, Z+FB)
        h_self = torch.cat([z, e_fb], dim=1).permute(0, 2, 1)

        # Gather partner features
        batch_size, seq_len, _ = h_self.shape
        # Create batch indices for gather
        batch_idx = (
            torch.arange(batch_size, device=seq.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Handle -1 in partner_idx (unpaired) by clamping to 0 then masking
        safe_partner_idx = partner_idx.clone()
        mask_unpaired = safe_partner_idx == -1
        safe_partner_idx[mask_unpaired] = 0

        # Gather: h_partner[b, i] = h_self[b, partner_idx[b, i]]
        # We need to flatten to use index selection or use gather
        # Easier: use advanced indexing
        h_partner = h_self[batch_idx, safe_partner_idx]  # (B, L, C)

        # Mask unpaired partners to 0
        h_partner[mask_unpaired] = 0

        # Fuse
        rnn_in = torch.cat([h_self, h_partner], dim=-1)  # (B, L, (Z+FB)*2)

        # RNN
        rnn_out, _ = self.rnn(rnn_in)

        # Prediction
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits


# ==================================================================================
# TRAINING UTILS
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    # pred, target: (B, L, 5)
    # We only care about columns [0, 1, 3] and positions 0..67

    scored_cols = Config.TARGET_INDICES

    loss = 0
    count = 0

    for col_idx in scored_cols:
        p = pred[:, : Config.PRED_LEN, col_idx]
        t = target[:, : Config.PRED_LEN, col_idx]

        mse = (p - t) ** 2
        root_mean_mse = torch.sqrt(mse.mean())
        loss += root_mean_mse
        count += 1

    return loss / count


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            pid = batch["partner_id"].to(device)
            pidx = batch["partner_idx"].to(device)
            targets = batch["targets"].to(device)

            # Pass 1
            pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)

            # Pass 2
            pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

            loss = mcrmse_loss(pred2, targets)
            total_loss += loss.item() * seq.size(0)
            count += seq.size(0)

    return total_loss / count


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0

    for batch in loader:
        seq = batch["seq"].to(device)
        struct = batch["struct"].to(device)
        loop = batch["loop"].to(device)
        pid = batch["partner_id"].to(device)
        pidx = batch["partner_idx"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Pass 1
        pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)

        # Pass 2 (Detach pred1)
        pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1.detach())

        # Loss
        loss1 = mcrmse_loss(pred1, targets)
        loss2 = mcrmse_loss(pred2, targets)
        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_data = process_data("train")
    val_data = process_data("val")
    test_data = process_data("test")

    train_dataset = RNADataset(train_data, "train")
    val_dataset = RNADataset(val_data, "val")
    test_dataset = RNADataset(test_data, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model Setup
    model = SDFRN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    early_stop_count = 0

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            torch.save(
                model.state_dict(), os.path.join(Config.CACHE_DIR, "best_model.pth")
            )
        else:
            early_stop_count += 1

        if early_stop_count >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 4. Inference
    print("Generating submission...")
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CACHE_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    preds = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            pid = batch["partner_id"].to(device)
            pidx = batch["partner_idx"].to(device)

            # Pass 1
            pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)
            # Pass 2
            pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

            preds.append(pred2.cpu().numpy())

    # Flatten predictions
    preds = np.concatenate(preds, axis=0)  # (N_test, 107, 5)
    test_ids = test_data["ids"]

    # Format submission
    # We need to output for each id_seqpos
    submission_rows = []
    for i, sample_id in enumerate(test_ids):
        sample_pred = preds[i]  # (107, 5)
        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Model output matches this order
            vals = sample_pred[j]
            # Clip negative values if necessary? Usually not strictly required but good practice if metric is RMSE
            # But here targets can be negative, so we don't clip.
            submission_rows.append([row_id] + vals.tolist())

    sub_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + Config.ALL_TARGETS)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
