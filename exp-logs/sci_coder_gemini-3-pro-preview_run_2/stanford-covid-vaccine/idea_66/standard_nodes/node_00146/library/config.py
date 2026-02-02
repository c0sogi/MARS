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
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# ==================================================================================
# CONFIGURATION
# ==================================================================================
class Config:
    # Data Dimensions
    SEQ_LENGTH = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Feature Dimensions
    # Sequence(4) + Structure(3) + LoopType(7) + PartnerIdentity(4)
    INPUT_CHANNELS = 4 + 3 + 7 + 4

    # Model Hyperparameters
    HIDDEN_DIM = 32  # Growth rate
    LATENT_DIM = 64
    FEEDBACK_DIM = 32
    FEEDBACK_GROWTH_RATE = 12
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 25  # Cap for runtime safety
    EARLY_STOPPING_PATIENCE = 5
    NUM_WORKERS = 2
    SEED = 42

    # Paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_66"
    SUBMISSION_PATH = "./submission/submission.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Cache Keys
    CACHE_TRAIN = "train_data_gc_sdn_v1.npz"
    CACHE_VAL = "val_data_gc_sdn_v1.npz"
    CACHE_TEST = "test_data_gc_sdn_v1.npz"


# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)


# Set Seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(Config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================================================
# DATA PROCESSING UTILS
# ==================================================================================


def get_structure_adj(structure, seq_length):
    """
    Parses dot-bracket structure to find partner indices.
    Returns:
        partner_indices: Array of shape (L,) where value is index of partner or -1.
    """
    stack = []
    partner_indices = np.full(seq_length, -1, dtype=np.int32)

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
    One-hot encodes a sequence string based on a vocab dictionary.
    """
    mapping = {char: i for i, char in enumerate(vocab)}
    seq_len = len(seq)
    vocab_size = len(vocab)
    one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def process_data(csv_path, is_test=False, load_cached_data=True, cache_name="data.npz"):
    """
    Loads metadata, processes features, and caches results.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data}

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Vocabs
    seq_vocab = "AGUC"
    struct_vocab = "()."
    loop_vocab = "SMIBHEX"

    # Containers
    ids = []
    features = []
    partner_indices_list = []
    targets = []

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Base One-Hot Features
        ohe_seq = one_hot_encode(seq, seq_vocab)  # (L, 4)
        ohe_struct = one_hot_encode(struct, struct_vocab)  # (L, 3)
        ohe_loop = one_hot_encode(loop, loop_vocab)  # (L, 7)

        # 2. Partner Indices
        p_idx = get_structure_adj(struct, Config.SEQ_LENGTH)

        # 3. Partner Identity (Explicit Injection)
        # If i is paired with j, get one-hot of base at j. Else 0.
        ohe_partner = np.zeros((Config.SEQ_LENGTH, 4), dtype=np.float32)
        for i, partner_i in enumerate(p_idx):
            if partner_i != -1:
                ohe_partner[i] = ohe_seq[partner_i]

        # Concatenate all features
        # Shape: (L, 4+3+7+4) = (L, 18)
        sample_features = np.concatenate(
            [ohe_seq, ohe_struct, ohe_loop, ohe_partner], axis=1
        )

        ids.append(row["id"])
        features.append(sample_features)
        partner_indices_list.append(p_idx)

        if not is_test:
            # Parse targets
            sample_targets = []
            for col in target_cols:
                # Targets are stringified lists in CSV
                val_list = ast.literal_eval(row[col])
                # Pad to SEQ_LENGTH with NaNs or zeros (we mask loss anyway)
                # The provided targets are length 68. We pad to 107.
                padded = np.zeros(Config.SEQ_LENGTH, dtype=np.float32)
                padded[: len(val_list)] = val_list
                sample_targets.append(padded)
            targets.append(np.stack(sample_targets, axis=1))  # (L, 5)

    features = np.array(features, dtype=np.float32)
    partner_indices_list = np.array(partner_indices_list, dtype=np.int32)
    ids = np.array(ids)

    if not is_test:
        targets = np.array(targets, dtype=np.float32)
        data_dict = {
            "ids": ids,
            "features": features,
            "partner_indices": partner_indices_list,
            "targets": targets,
        }
    else:
        data_dict = {
            "ids": ids,
            "features": features,
            "partner_indices": partner_indices_list,
        }

    np.savez_compressed(cache_path, **data_dict)
    print(f"Data processed and saved to {cache_path}")
    return data_dict


# ==================================================================================
# DATASET
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.features = data_dict["features"]
        self.partner_indices = data_dict["partner_indices"]
        self.ids = data_dict["ids"]
        self.is_test = is_test
        if not is_test:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (L, C) -> (C, L) for PyTorch Conv1d
        feat = torch.tensor(self.features[idx], dtype=torch.float32).permute(1, 0)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.is_test:
            return feat, p_idx, self.ids[idx]
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return feat, p_idx, target


# ==================================================================================
# MODEL ARCHITECTURE (GC-SDN)
# ==================================================================================


class SpatialInputStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, C_in, L)
        x = self.conv(x)
        # LN expects (B, L, C), so permute
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        return x


class DenseDilatedBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        self.conv_dilated = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )

        self.ln2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()
        self.conv_point = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # Post-activation structure
        # x: (B, C_in, L)

        # Branch
        out = x.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.conv_dilated(out)

        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        out = self.conv_point(out)
        out = self.dropout(out)

        return out


class FeedbackModule(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        # Spatial Stem
        self.stem = SpatialInputStem(in_channels, hidden_dim)

        # Lightweight Dense TCN
        self.blocks = nn.ModuleList()
        current_dim = hidden_dim
        # Using a smaller set of dilations for feedback
        dilations = [1, 2, 4, 8]
        for d in dilations:
            self.blocks.append(
                DenseDilatedBlock(current_dim, Config.FEEDBACK_GROWTH_RATE, d)
            )
            current_dim += Config.FEEDBACK_GROWTH_RATE

        self.proj = nn.Conv1d(current_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        features = [x]
        for block in self.blocks:
            new_feat = block(torch.cat(features, dim=1))
            features.append(new_feat)

        out = torch.cat(features, dim=1)
        out = self.proj(out)
        return out


class GCSDNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Static Encoder
        self.static_stem = SpatialInputStem(Config.INPUT_CHANNELS, Config.HIDDEN_DIM)

        self.static_blocks = nn.ModuleList()
        current_dim = Config.HIDDEN_DIM
        for d in Config.DILATIONS:
            self.static_blocks.append(
                DenseDilatedBlock(current_dim, Config.HIDDEN_DIM, d)
            )
            current_dim += Config.HIDDEN_DIM

        self.static_proj = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)

        # 2. Feedback Module
        self.feedback_module = FeedbackModule(
            Config.NUM_TARGETS, Config.HIDDEN_DIM, Config.FEEDBACK_DIM
        )

        # 3. Interaction & Aggregation
        # Input to RNN: Self(Latent + Feedback) + Partner(Latent + Feedback)
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2

        self.rnn = nn.GRU(
            rnn_input_dim,
            Config.LATENT_DIM,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Linear(Config.LATENT_DIM * 2, Config.NUM_TARGETS)

    def forward_backbone(self, x):
        x = self.static_stem(x)
        features = [x]
        for block in self.static_blocks:
            new_feat = block(torch.cat(features, dim=1))
            features.append(new_feat)
        out = torch.cat(features, dim=1)
        z = self.static_proj(out)  # (B, Latent, L)
        return z

    def forward_pass(self, z, y_prev, partner_indices):
        # z: (B, Latent, L)
        # y_prev: (B, Targets, L)
        # partner_indices: (B, L)

        # 1. Process Feedback
        # Mask specific channels in y_prev (deg_pH10 at idx 2, deg_50C at idx 4)
        # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Unscored: deg_pH10(2), deg_50C(4)
        mask = torch.tensor([1, 1, 0, 1, 0], device=z.device, dtype=torch.float32).view(
            1, -1, 1
        )
        y_masked = y_prev * mask

        e_fb = self.feedback_module(y_masked)  # (B, FeedbackDim, L)

        # 2. Interaction (Gather)
        # Combine Z and E_fb
        combined = torch.cat([z, e_fb], dim=1)  # (B, Latent+Feedback, L)
        combined_t = combined.permute(0, 2, 1)  # (B, L, C)

        batch_size, seq_len, channels = combined_t.shape

        # Gather partner features
        # partner_indices is (B, L). -1 indicates no partner.
        # We replace -1 with 0 for gather, then mask result.
        p_idx_safe = partner_indices.clone()
        mask_unpaired = p_idx_safe == -1
        p_idx_safe[mask_unpaired] = 0

        # Expand indices for gather: (B, L, C)
        p_idx_expanded = p_idx_safe.unsqueeze(-1).expand(-1, -1, channels)

        partner_vec = torch.gather(combined_t, 1, p_idx_expanded)
        partner_vec[mask_unpaired] = 0.0  # Zero out unpaired

        # Concatenate Self and Partner
        rnn_in = torch.cat([combined_t, partner_vec], dim=2)  # (B, L, C*2)

        # 3. Aggregation
        rnn_out, _ = self.rnn(rnn_in)
        pred = self.head(rnn_out)  # (B, L, Targets)

        return pred.permute(0, 2, 1)  # (B, Targets, L)

    def forward(self, x, partner_indices, y_init=None):
        # x: (B, C_in, L)
        # partner_indices: (B, L)

        # Compute Static Latent
        z = self.forward_backbone(x)

        # Pass 1: Zero Feedback
        if y_init is None:
            y_init = torch.zeros(
                (x.size(0), Config.NUM_TARGETS, x.size(2)), device=x.device
            )

        y1 = self.forward_pass(z, y_init, partner_indices)

        # Pass 2: Feedback from Pass 1 (Detached in training loop usually, but here we define flow)
        # In training loop, we will handle detach explicitly if needed, but for simple forward:
        y2 = self.forward_pass(z, y1, partner_indices)

        return y1, y2


# ==================================================================================
# TRAINING UTILS
# ==================================================================================


def mcrmse_loss(pred, target, scored_len=Config.SEQ_SCORED):
    # pred, target: (B, 5, L)
    # Only scored columns: 0, 1, 3 (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Only scored positions: 0 to scored_len

    scored_cols = [0, 1, 3]

    pred_scored = pred[:, scored_cols, :scored_len]
    target_scored = target[:, scored_cols, :scored_len]

    mse = (pred_scored - target_scored) ** 2
    # Mean over positions (dim 2) and batch (dim 0), then sqrt, then mean over columns (dim 1)
    rmse_per_col = torch.sqrt(torch.mean(mse, dim=(0, 2)))
    return torch.mean(rmse_per_col)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0

    # For global MCRMSE calculation (more accurate)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, p_idx, targets in loader:
            features = features.to(device)
            p_idx = p_idx.to(device)
            targets = targets.to(device)

            # Inference: 2 passes
            y1, y2 = model(features, p_idx)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 5, L)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric on scored region
    scored_len = Config.SEQ_SCORED
    scored_cols = [0, 1, 3]

    preds_s = all_preds[:, scored_cols, :scored_len]
    targs_s = all_targets[:, scored_cols, :scored_len]

    mse = np.mean((preds_s - targs_s) ** 2, axis=(0, 2))
    rmse = np.sqrt(mse)
    mcrmse = np.mean(rmse)

    return mcrmse


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================


def main():
    print("Starting Idea 66: GC-SDN Implementation")

    # 1. Load Data
    train_data = process_data(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        is_test=False,
        cache_name=Config.CACHE_TRAIN,
    )
    val_data = process_data(
        os.path.join(Config.METADATA_DIR, "val.csv"),
        is_test=False,
        cache_name=Config.CACHE_VAL,
    )
    test_data = process_data(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        is_test=True,
        cache_name=Config.CACHE_TEST,
    )

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data, is_test=True)

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

    # 2. Model & Optimizer
    model = GCSDNModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(
        f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples."
    )

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0

        for features, p_idx, targets in train_loader:
            features = features.to(device)
            p_idx = p_idx.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward
            # Pass 1: Zero init
            z = model.forward_backbone(features)
            y_init = torch.zeros_like(targets)
            y1 = model.forward_pass(z, y_init, p_idx)

            # Pass 2: Feedback from detached y1
            y2 = model.forward_pass(z, y1.detach(), p_idx)

            # Loss: MCRMSE on y2 + 0.5 * MCRMSE on y1
            loss1 = mcrmse_loss(y1, targets)
            loss2 = mcrmse_loss(y2, targets)
            loss = loss2 + 0.5 * loss1

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        val_mcrmse = validate(model, val_loader, device)
        scheduler.step(val_mcrmse)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_val_loss:
            best_val_loss = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # 4. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds_dict = {}

    with torch.no_grad():
        for features, p_idx, ids in test_loader:
            features = features.to(device)
            p_idx = p_idx.to(device)

            # 2-Pass Inference
            y1, y2 = model(features, p_idx)
            y_final = y2.cpu().numpy()  # (B, 5, L)

            for i, sample_id in enumerate(ids):
                # We need to output for each seqpos.
                # The sample submission format is: id_seqpos, reactivity, deg_Mg_pH10, ...
                # We must output all positions (0 to 106), though only 0-67 are scored.
                # However, the prompt says "Positions greater than seq_scored... still need a value".

                sample_pred = y_final[i]  # (5, 107)

                for seqpos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seqpos}"
                    preds = sample_pred[:, seqpos]
                    preds_dict[row_id] = preds

    # 5. Create Submission File
    print("Generating submission file...")
    # Load sample submission to get correct order/rows
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Columns in order: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Targets index map: 0:reactivity, 1:deg_Mg_pH10, 2:deg_pH10, 3:deg_Mg_50C, 4:deg_50C

    submission_data = []
    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # We iterate through sample submission to ensure exact match of rows
    # But for speed, we can construct DataFrame directly if we trust the IDs match logic.
    # The test set has 240 samples * 107 pos = 25680 rows.

    # Let's reconstruct the dataframe from our preds_dict
    # This is safer to match the exact IDs required.

    result_array = np.zeros((len(sample_sub), 5))

    for idx, row in sample_sub.iterrows():
        row_id = row["id_seqpos"]
        if row_id in preds_dict:
            result_array[idx] = preds_dict[row_id]
        else:
            # Should not happen if test set matches
            pass

    submission_df = pd.DataFrame(result_array, columns=cols)
    submission_df.insert(0, "id_seqpos", sample_sub["id_seqpos"])

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
