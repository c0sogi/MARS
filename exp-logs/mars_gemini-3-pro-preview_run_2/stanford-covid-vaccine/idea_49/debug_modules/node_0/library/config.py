import os
import ast
import gc
import random
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
    # Data Dimensions
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Features
    # Sequence (4) + Structure (3) + Loop (7) + Partner Identity (4) = 18
    INPUT_CHANNELS = 18

    # Model Hyperparameters
    HIDDEN_DIM = 64
    FEEDBACK_DIM = 16
    FEEDBACK_EMBED_DIM = 32
    LATENT_DIM = 64
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Targets
    NUM_TARGETS = 5
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    TARGET_INDICES = [
        0,
        1,
        3,
    ]  # Indices corresponding to scored targets in the 5-col array

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_49"
    CACHE_DIR = "./working/idea_49"

    # Files
    TRAIN_FILE = "train.csv"
    VAL_FILE = "val.csv"
    TEST_FILE = "test.csv"
    SAMPLE_SUBMISSION = "sample_submission.csv"

    # Cache Keys
    CACHE_TRAIN = "train_data_ei_pfn_v1.npz"
    CACHE_VAL = "val_data_ei_pfn_v1.npz"
    CACHE_TEST = "test_data_ei_pfn_v1.npz"

    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)
Config.set_seed()


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a mapping array where arr[i] = j if i pairs with j, else -1.
    """
    pairs = np.full(len(structure), -1, dtype=int)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def process_data(mode="train", load_cached_data=True):
    """
    Loads metadata, generates features (One-Hot + Partner Identity), and targets.
    Handles caching.
    """
    cache_file = getattr(Config, f"CACHE_{mode.upper()}")
    cache_path = os.path.join(Config.CACHE_DIR, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        data = np.load(cache_path)
        return {
            "inputs": data["inputs"],
            "partner_indices": data["partner_indices"],
            "targets": data["targets"] if "targets" in data else None,
            "ids": data["ids"],
        }

    print(f"Processing {mode} data from scratch...")

    # Load Metadata
    meta_file = getattr(Config, f"{mode.upper()}_FILE")
    df = pd.read_csv(os.path.join(Config.METADATA_DIR, meta_file))

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Channels: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    partner_indices = np.full((num_samples, seq_len), -1, dtype=np.int32)
    targets = (
        np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
        if mode != "test"
        else None
    )
    ids = df["id"].values

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Base Features
        for i, char in enumerate(seq):
            inputs[idx, i, seq_map.get(char, 0)] = 1.0

        for i, char in enumerate(struct):
            inputs[idx, i, 4 + struct_map.get(char, 0)] = 1.0

        for i, char in enumerate(loop):
            inputs[idx, i, 7 + loop_map.get(char, 0)] = 1.0

        # 2. Partner Identity & Indices
        pairs = get_structure_pairs(struct)
        partner_indices[idx] = pairs

        for i, pair_idx in enumerate(pairs):
            if pair_idx != -1:
                partner_char = seq[pair_idx]
                # Partner channels start at 4+3+7 = 14
                inputs[idx, i, 14 + seq_map.get(partner_char, 0)] = 1.0

        # 3. Targets
        if mode != "test":
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Note: The order in metadata/train.csv might vary, but usually consistent.
            # We assume the order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # based on typical dataset structure.
            t_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
            for t_i, col in enumerate(t_cols):
                val_list = ast.literal_eval(row[col])
                # Pad or truncate to seq_len (targets are usually 68 long)
                length = len(val_list)
                targets[idx, :length, t_i] = val_list

    # Save to cache
    save_dict = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return {
        "inputs": inputs,
        "partner_indices": partner_indices,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.inputs = torch.tensor(data_dict["inputs"], dtype=torch.float32)
        self.partner_indices = torch.tensor(
            data_dict["partner_indices"], dtype=torch.long
        )
        self.mode = mode
        if mode != "test":
            self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            "inputs": self.inputs[idx],  # (L, 18)
            "partner_indices": self.partner_indices[idx],  # (L,)
        }
        if self.mode != "test":
            item["targets"] = self.targets[idx]  # (L, 5)
        return item


# ==================================================================================
# MODEL
# ==================================================================================


class PreActDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.ln2 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        # Input x: (B, C, L)
        # LayerNorm expects (B, L, C), so permute
        res = x

        out = x.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act(out)
        out = out.permute(0, 2, 1)

        out = self.conv1(out)

        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act(out)
        out = out.permute(0, 2, 1)

        out = self.conv2(out)
        out = self.dropout(out)

        return res + out


class EIPFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Input Embedding Stem
        # Projects sparse one-hot (18) to dense latent (64)
        self.input_stem = nn.Conv1d(
            config.INPUT_CHANNELS, config.HIDDEN_DIM, kernel_size=1
        )

        # 2. Backbone (Static)
        self.backbone_blocks = nn.ModuleList()
        for d in config.DILATIONS:
            self.backbone_blocks.append(
                PreActDilatedBlock(
                    config.HIDDEN_DIM,
                    config.HIDDEN_DIM,
                    dilation=d,
                    dropout=config.DROPOUT,
                )
            )
        self.backbone_out = nn.Conv1d(
            config.HIDDEN_DIM, config.LATENT_DIM, kernel_size=1
        )

        # 3. Pure-Feedback Module
        self.fb_embed = nn.Conv1d(
            config.NUM_TARGETS, config.FEEDBACK_DIM, kernel_size=1
        )
        self.fb_blocks = nn.ModuleList()
        for d in config.DILATIONS:  # Use same dilation schedule but smaller width
            self.fb_blocks.append(
                PreActDilatedBlock(
                    config.FEEDBACK_DIM,
                    config.FEEDBACK_DIM,
                    dilation=d,
                    dropout=config.DROPOUT,
                )
            )
        self.fb_out = nn.Conv1d(
            config.FEEDBACK_DIM, config.FEEDBACK_EMBED_DIM, kernel_size=1
        )

        # 4. Interaction & Aggregation
        # Input to GRU: (Latent_i + FB_i) + (Latent_j + FB_j) = (64+32) * 2 = 192
        gru_input_dim = (config.LATENT_DIM + config.FEEDBACK_EMBED_DIM) * 2
        self.gru = nn.GRU(
            gru_input_dim,
            config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(config.HIDDEN_DIM * 2, config.NUM_TARGETS)

    def forward_backbone(self, x):
        # x: (B, L, 18) -> (B, 18, L)
        x = x.permute(0, 2, 1)
        x = self.input_stem(x)

        for block in self.backbone_blocks:
            x = block(x)

        z = self.backbone_out(x)  # (B, Latent, L)
        return z.permute(0, 2, 1)  # (B, L, Latent)

    def forward_feedback(self, y_prev):
        # y_prev: (B, L, 5) -> (B, 5, L)
        x = y_prev.permute(0, 2, 1)
        x = self.fb_embed(x)

        for block in self.fb_blocks:
            x = block(x)

        e_fb = self.fb_out(x)  # (B, FB_Embed, L)
        return e_fb.permute(0, 2, 1)  # (B, L, FB_Embed)

    def forward_head(self, z, e_fb, partner_indices):
        # z: (B, L, Latent)
        # e_fb: (B, L, FB_Embed)
        # partner_indices: (B, L)

        B, L, _ = z.shape

        # Self vector
        self_vec = torch.cat([z, e_fb], dim=2)  # (B, L, 96)

        # Partner vector
        # Create a batch index grid
        batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, L)

        # Mask for unpaired bases (-1)
        mask_unpaired = partner_indices == -1
        # Replace -1 with 0 for gather (will be masked out later)
        safe_indices = partner_indices.clone()
        safe_indices[mask_unpaired] = 0

        # Gather partner features
        partner_vec = self_vec[batch_idx, safe_indices]  # (B, L, 96)

        # Zero out unpaired
        partner_vec[mask_unpaired] = 0.0

        # Concatenate
        combined = torch.cat([self_vec, partner_vec], dim=2)  # (B, L, 192)

        # RNN
        out, _ = self.gru(combined)

        # Linear
        preds = self.head(out)  # (B, L, 5)
        return preds

    def forward(self, inputs, partner_indices, y_prev=None):
        # 1. Static Backbone
        z = self.forward_backbone(inputs)

        # 2. Feedback Loop
        if y_prev is None:
            y_prev = torch.zeros(
                inputs.shape[0],
                inputs.shape[1],
                self.config.NUM_TARGETS,
                device=inputs.device,
            )

        # Mask feedback (only scored columns should drive feedback ideally, but here we use all predicted)
        # Strict masking: Zero out columns that are not scored in the metric?
        # Idea says: "Strict Masking: we strictly apply a binary mask to zero out unscored targets"
        # Scored are 0, 1, 3. Unscored 2, 4.
        mask = torch.zeros_like(y_prev)
        mask[:, :, [0, 1, 3]] = 1.0
        y_masked = y_prev * mask

        e_fb = self.forward_feedback(y_masked)

        # 3. Interaction
        preds = self.forward_head(z, e_fb, partner_indices)

        return preds


# ==================================================================================
# TRAINING UTILS
# ==================================================================================


def mcrmse_loss(pred, target, mask=None):
    """
    Mean Columnwise Root Mean Squared Error.
    pred, target: (B, L, 5)
    mask: (B, L) - valid positions
    """
    # Only evaluate on scored targets: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    losses = []
    for idx in scored_indices:
        p = pred[:, :, idx]
        t = target[:, :, idx]

        if mask is not None:
            diff = (p - t) ** 2
            # Mean over valid positions per sample, then sqrt, then mean over batch?
            # Metric definition: sqrt(mean((y-y_hat)^2)) per column.
            # We flatten valid positions.
            valid_diff = diff[mask.bool()]
            rmse = torch.sqrt(torch.mean(valid_diff))
        else:
            rmse = torch.sqrt(torch.mean((p - t) ** 2))
        losses.append(rmse)

    return torch.mean(torch.stack(losses))


def train_model(debug=False):
    # Load Data
    train_data = process_data("train")
    val_data = process_data("val")

    if debug:
        train_data["inputs"] = train_data["inputs"][:100]
        train_data["partner_indices"] = train_data["partner_indices"][:100]
        train_data["targets"] = train_data["targets"][:100]
        val_data["inputs"] = val_data["inputs"][:20]
        val_data["partner_indices"] = val_data["partner_indices"][:20]
        val_data["targets"] = val_data["targets"][:20]

    train_ds = RNADataset(train_data, "train")
    val_ds = RNADataset(val_data, "val")

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EIPFN(Config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            p_idx = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Mask for valid positions (0 to 67)
            mask = torch.zeros(inputs.shape[0], inputs.shape[1], device=device)
            mask[:, : Config.SEQ_SCORED] = 1.0

            optimizer.zero_grad()

            # Pass 1: Zero feedback
            preds_1 = model(inputs, p_idx, y_prev=None)
            loss_1 = mcrmse_loss(preds_1, targets, mask)

            # Pass 2: Feedback from Pass 1 (Detached)
            preds_1_detached = preds_1.detach()
            preds_2 = model(inputs, p_idx, y_prev=preds_1_detached)
            loss_2 = mcrmse_loss(preds_2, targets, mask)

            # Total Loss
            loss = loss_2 + 0.5 * loss_1

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                p_idx = batch["partner_indices"].to(device)
                targets = batch["targets"].to(device)

                mask = torch.zeros(inputs.shape[0], inputs.shape[1], device=device)
                mask[:, : Config.SEQ_SCORED] = 1.0

                # Inference Pass 1
                preds_1 = model(inputs, p_idx, y_prev=None)
                # Inference Pass 2
                preds_2 = model(inputs, p_idx, y_prev=preds_1)

                val_loss = mcrmse_loss(preds_2, targets, mask)
                val_loss_accum += val_loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


def predict_submission():
    print("Generating submission...")
    test_data = process_data("test")
    test_ds = RNADataset(test_data, "test")
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EIPFN(Config).to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            p_idx = batch["partner_indices"].to(device)

            # Pass 1
            preds_1 = model(inputs, p_idx, y_prev=None)
            # Pass 2
            preds_2 = model(inputs, p_idx, y_prev=preds_1)

            all_preds.append(preds_2.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 5)

    # Format submission
    # id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
    ids = test_data["ids"]

    submission_rows = []
    for i, sample_id in enumerate(ids):
        # We must predict for all positions, but only up to seq_scored is relevant for metric.
        # Submission format requires all positions (length of sequence).
        seq_len = all_preds.shape[1]
        for j in range(seq_len):
            row_id = f"{sample_id}_{j}"
            vals = all_preds[i, j]
            # Clip values to valid range if necessary (though not strictly required)
            # vals = np.clip(vals, 0, None)

            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Our model outputs in this order (based on targets construction)
            submission_rows.append([row_id] + vals.tolist())

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=columns)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    train_model(debug=False)
    predict_submission()
