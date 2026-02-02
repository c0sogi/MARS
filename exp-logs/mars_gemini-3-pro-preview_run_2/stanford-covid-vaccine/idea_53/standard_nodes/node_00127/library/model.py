import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, parse_list_column, mcrmse_loss
from library.layers import SpatialInputStem, DenseTCN

# =========================================================================
# 1. Data Processing & Dataset
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, targets=None):
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features: [C, L]
        # partner_indices: [L]
        # targets: [L, 5] (if available)

        x = torch.tensor(self.features[idx], dtype=torch.float32)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_idx, y
        else:
            return x, p_idx


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find base pairs.
    Returns a mapping {index: partner_index}. Unpaired bases are not in the dict.
    """
    pairs = {}
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


def process_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Processes RNA data: generates one-hot encodings, partner features, and targets.
    Uses caching to avoid re-computation.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        if is_test:
            return data["features"], data["partner_indices"], data["ids"]
        else:
            return data["features"], data["partner_indices"], data["targets"]

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Debug subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    # Mappings
    seq_map = {c: i for i, c in enumerate("AGCU")}
    struct_map = {c: i for i, c in enumerate("().")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    features_list = []
    partner_indices_list = []
    targets_list = []
    ids_list = df["id"].values if "id" in df.columns else []

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        length = len(seq)

        # 1. Sequence One-Hot (4)
        seq_oh = np.zeros((4, length), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in seq_map:
                seq_oh[seq_map[char], i] = 1.0

        # 2. Structure One-Hot (3)
        struct_oh = np.zeros((3, length), dtype=np.float32)
        for i, char in enumerate(struct):
            if char in struct_map:
                struct_oh[struct_map[char], i] = 1.0

        # 3. Loop Type One-Hot (7)
        loop_oh = np.zeros((7, length), dtype=np.float32)
        for i, char in enumerate(loop):
            if char in loop_map:
                loop_oh[loop_map[char], i] = 1.0

        # 4. Partner Identity (4) & Indices
        pairs = get_structure_pairs(struct)
        partner_oh = np.zeros((4, length), dtype=np.float32)
        p_indices = np.full(length, -1, dtype=np.int32)

        for i in range(length):
            if i in pairs:
                j = pairs[i]
                p_indices[i] = j
                partner_char = seq[j]
                if partner_char in seq_map:
                    partner_oh[seq_map[partner_char], i] = 1.0

        # Concatenate all features: [18, Length]
        sample_features = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_oh], axis=0
        )
        features_list.append(sample_features)
        partner_indices_list.append(p_indices)

        # Targets
        if not is_test:
            # Parse stringified lists
            t_react = parse_list_column(row["reactivity"])
            t_mg_ph10 = parse_list_column(row["deg_Mg_pH10"])
            t_ph10 = parse_list_column(row["deg_pH10"])
            t_mg_50c = parse_list_column(row["deg_Mg_50C"])
            t_50c = parse_list_column(row["deg_50C"])

            # Stack: [Length, 5]
            # Note: Targets are length 68, need to pad to 107 for batching,
            # but loss only uses first 68.
            sample_targets = np.zeros((length, 5), dtype=np.float32)

            # Fill available data
            valid_len = len(t_react)
            sample_targets[:valid_len, 0] = t_react
            sample_targets[:valid_len, 1] = t_mg_ph10
            sample_targets[:valid_len, 2] = t_ph10
            sample_targets[:valid_len, 3] = t_mg_50c
            sample_targets[:valid_len, 4] = t_50c

            targets_list.append(sample_targets)

    # Convert to numpy arrays
    features = np.array(features_list)
    partner_indices = np.array(partner_indices_list)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if is_test:
        np.savez(
            cache_path, features=features, partner_indices=partner_indices, ids=ids_list
        )
        return features, partner_indices, ids_list
    else:
        targets = np.array(targets_list)
        np.savez(
            cache_path,
            features=features,
            partner_indices=partner_indices,
            targets=targets,
        )
        return features, partner_indices, targets


# =========================================================================
# 2. Model Architecture (SS-DFRN)
# =========================================================================


class SS_DFRN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Spatial Input Stem
        self.stem = SpatialInputStem(
            in_channels=Config.NUM_NODE_FEATURES,
            out_channels=Config.BACKBONE_GROWTH_RATE,
            kernel_size=Config.STEM_KERNEL_SIZE,
        )

        # 2. Main Backbone (Dense Dilated TCN)
        # Calculate output channels of DenseNet: In + Num_Blocks * Growth
        self.backbone = DenseTCN(
            in_channels=Config.BACKBONE_GROWTH_RATE,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            dilations=Config.BACKBONE_DILATIONS,
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )
        backbone_out_channels = self.backbone.out_channels

        # Latent Projection
        self.latent_proj = nn.Conv1d(
            backbone_out_channels, Config.LATENT_DIM, kernel_size=1
        )

        # 3. Lightweight Dense Feedback Module
        # Embedding
        self.fb_embedding = nn.Conv1d(
            Config.FEEDBACK_INPUT_CHANNELS, Config.FEEDBACK_GROWTH_RATE, kernel_size=1
        )

        # Feedback Backbone
        self.fb_backbone = DenseTCN(
            in_channels=Config.FEEDBACK_GROWTH_RATE,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=Config.BACKBONE_DILATIONS,  # Same dilation pattern
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )
        fb_out_channels = self.fb_backbone.out_channels

        # Feedback Projection
        self.fb_proj = nn.Conv1d(
            fb_out_channels, Config.FEEDBACK_OUTPUT_CHANNELS, kernel_size=1
        )

        # 4. Interaction & Aggregation
        # Input to RNN is [Z_i, E_fb_i, Z_j, E_fb_j] -> 2 * (Latent + Feedback_Out)
        rnn_input_dim = 2 * (Config.LATENT_DIM + Config.FEEDBACK_OUTPUT_CHANNELS)

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            bidirectional=Config.RNN_BIDIRECTIONAL,
            batch_first=True,
        )

        rnn_output_dim = (
            Config.RNN_HIDDEN_SIZE * 2
            if Config.RNN_BIDIRECTIONAL
            else Config.RNN_HIDDEN_SIZE
        )

        # Output Head
        self.head = nn.Linear(rnn_output_dim, 5)

    def forward(self, x, partner_indices, feedback=None):
        # x: [Batch, 18, Seq_Len]
        # partner_indices: [Batch, Seq_Len]
        # feedback: [Batch, Seq_Len, 5] or None

        B, C, L = x.shape

        # --- Step 1 & 2: Main Backbone ---
        stem_out = self.stem(x)
        backbone_out = self.backbone(stem_out)

        # --- Step 3: Latent Projection ---
        # Z: [Batch, Latent_Dim, Seq_Len]
        z = self.latent_proj(backbone_out)

        # --- Step 4: Feedback Processing ---
        if feedback is None:
            # First pass: zero feedback
            fb_emb = torch.zeros(B, Config.FEEDBACK_OUTPUT_CHANNELS, L, device=x.device)
        else:
            # feedback: [Batch, Seq_Len, 5] -> Permute to [Batch, 5, Seq_Len]
            fb_in = feedback.permute(0, 2, 1)

            # Mask unscored columns (indices 2 and 4) to zero to prevent noise
            # Scored: 0, 1, 3. Unscored: 2, 4.
            # Create a mask or just manually zero them out?
            # Strict masking as per Idea:
            mask = torch.tensor(
                [1, 1, 0, 1, 0], device=x.device, dtype=torch.float32
            ).view(1, 5, 1)
            fb_in = fb_in * mask

            fb_feat = self.fb_embedding(fb_in)
            fb_out = self.fb_backbone(fb_feat)
            fb_emb = self.fb_proj(fb_out)

        # --- Step 5: Interaction (Gather/Fusion) ---
        # Concatenate Z and E_fb: [Batch, Latent+FB_Out, Seq_Len]
        node_features = torch.cat([z, fb_emb], dim=1)

        # Permute to [Batch, Seq_Len, Channels] for gathering
        node_features_perm = node_features.permute(0, 2, 1)
        nf_dim = node_features_perm.shape[2]

        # Prepare indices for gather
        # partner_indices is [B, L]. -1 indicates no partner.
        # Replace -1 with 0 for valid gathering, then mask result.
        valid_mask = (partner_indices != -1).unsqueeze(-1).float()  # [B, L, 1]
        safe_indices = partner_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices: [B, L, nf_dim]
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, nf_dim)

        # Gather partner features
        partner_features = torch.gather(node_features_perm, 1, gather_indices)

        # Apply mask (zero out features for unpaired bases)
        partner_features = partner_features * valid_mask

        # Concatenate Self and Partner: [Batch, Seq_Len, 2 * nf_dim]
        rnn_input = torch.cat([node_features_perm, partner_features], dim=2)

        # --- Step 6: RNN Aggregation ---
        rnn_out, _ = self.rnn(rnn_input)

        # --- Step 7: Output Head ---
        # [Batch, Seq_Len, 5]
        logits = self.head(rnn_out)

        return logits


# =========================================================================
# 3. Training & Inference Logic
# =========================================================================


def train_model():
    set_seed(Config.SEED)

    # Load Data
    train_x, train_p, train_y = process_data(
        Config.TRAIN_CSV, Config.TRAIN_CACHE, load_cached_data=True
    )
    val_x, val_p, val_y = process_data(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data=True
    )

    # Datasets & Loaders
    train_dataset = RNADataset(train_x, train_p, train_y)
    val_dataset = RNADataset(val_x, val_p, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model Setup
    model = SS_DFRN().to(Config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for x, p_idx, y in train_loader:
            x, p_idx, y = (
                x.to(Config.DEVICE),
                p_idx.to(Config.DEVICE),
                y.to(Config.DEVICE),
            )

            optimizer.zero_grad()

            # Pass 1: No Feedback
            pred1 = model(x, p_idx, feedback=None)
            loss1 = mcrmse_loss(pred1, y)

            # Pass 2: Feedback from detached Pass 1
            # Detach to stop gradients flowing back through feedback generation
            feedback_in = pred1.detach()
            pred2 = model(x, p_idx, feedback=feedback_in)
            loss2 = mcrmse_loss(pred2, y)

            # Combined Loss
            loss = Config.LOSS_PASS2_WEIGHT * loss2 + Config.LOSS_PASS1_WEIGHT * loss1

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for x, p_idx, y in val_loader:
                x, p_idx, y = (
                    x.to(Config.DEVICE),
                    p_idx.to(Config.DEVICE),
                    y.to(Config.DEVICE),
                )

                # Pass 1
                pred1 = model(x, p_idx, feedback=None)

                # Pass 2
                pred2 = model(x, p_idx, feedback=pred1)

                # Metric based on final prediction
                val_loss = mcrmse_loss(pred2, y)
                val_loss_accum += val_loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
            print(f"  New Best Model Saved! Loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_loss:.6f}")


def predict_and_submit():
    set_seed(Config.SEED)

    # Load Test Data
    test_x, test_p, test_ids = process_data(
        Config.TEST_CSV, Config.TEST_CACHE, load_cached_data=True, is_test=True
    )
    test_dataset = RNADataset(test_x, test_p)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Load Model
    model = SS_DFRN().to(Config.DEVICE)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found, using initialized weights.")

    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for x, p_idx in test_loader:
            x, p_idx = x.to(Config.DEVICE), p_idx.to(Config.DEVICE)

            # Pass 1
            pred1 = model(x, p_idx, feedback=None)
            # Pass 2
            pred2 = model(x, p_idx, feedback=pred1)

            # Move to CPU
            predictions.append(pred2.cpu().numpy())

    # Concatenate: [Total_Samples, Seq_Len, 5]
    all_preds = np.concatenate(predictions, axis=0)

    # Prepare Submission
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    for i, sample_id in enumerate(test_ids):
        # Determine sequence length from predictions (should be 107)
        seq_len = all_preds.shape[1]

        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            vals = all_preds[i, pos, :]

            # Clip values to prevent extreme outliers if necessary, though not strictly required
            # vals = np.clip(vals, -10, 10)

            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Orchestrates the full pipeline: Training -> Inference -> Submission.
    """
    train_model()
    predict_and_submit()
