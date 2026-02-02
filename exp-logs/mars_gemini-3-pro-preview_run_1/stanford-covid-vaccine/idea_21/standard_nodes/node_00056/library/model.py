import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import load_data


# =========================================================================
# 1. Reproducibility
# =========================================================================
def set_seed(seed=Config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================================
# 2. Model Architecture
# =========================================================================


class ResidualBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm and BiGRU.
    Maintains the residual stream width (hidden_dim) throughout.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        # BiGRU hidden_size is hidden_dim // 2 so that output (cat fwd/bwd) is hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len, Hidden_Dim)
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of layer outputs.
    Cite solution_lesson_node_00055: Prefer Scalar Mixture Aggregation over Dense Concatenation.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, tensors):
        # tensors: list of (B, L, D)
        # stack: (N, B, L, D)
        stacked = torch.stack(tensors, dim=0)
        norm_weights = torch.softmax(self.weights, dim=0)
        # weighted sum
        weighted_sum = torch.sum(stacked * norm_weights[:, None, None, None], dim=0)
        return self.gamma * weighted_sum


class ScalarAggregatedBiGRU(nn.Module):
    """
    Scalar-Aggregated Wide-Stream Residual BiGRU.
    Uses a learnable scalar mixture to combine outputs from all layers.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.nuc_emb = nn.Embedding(config.VOCAB_SIZE, config.NUC_EMBED_DIM)
        self.loop_emb = nn.Embedding(config.LOOP_VOCAB_SIZE, config.LOOP_EMBED_DIM)
        # Positional embedding (Sinusoidal Pair Dist) is passed as float input, no lookup needed.

        input_dim = config.NUC_EMBED_DIM + config.LOOP_EMBED_DIM + config.POS_EMBED_DIM

        # --- Stem ---
        # Projects input features to the residual stream width
        self.stem_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.stem_dropout = nn.Dropout(config.DROPOUT)

        # --- Backbone (Residual Blocks) ---
        self.blocks = nn.ModuleList(
            [
                ResidualBiGRUBlock(config.HIDDEN_DIM, config.DROPOUT)
                for _ in range(config.N_LAYERS)
            ]
        )

        # --- Scalar Mixture ---
        # Combines Stem + N_LAYERS Blocks
        self.mixture = ScalarMixture(1 + config.N_LAYERS)

        # --- Head ---
        # Input channels = HIDDEN_DIM (preserved by mixture)
        # Output channels = 3 (reactivity, deg_Mg_pH10, deg_Mg_50C)
        self.head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, sequence, loop_type, pair_dist):
        # sequence: (B, L)
        # loop_type: (B, L)
        # pair_dist: (B, L, Pos_Dim)

        # 1. Embeddings
        emb_nuc = self.nuc_emb(sequence)  # (B, L, Nuc_Dim)
        emb_loop = self.loop_emb(loop_type)  # (B, L, Loop_Dim)

        # 2. Concatenate Inputs
        x = torch.cat([emb_nuc, emb_loop, pair_dist], dim=-1)  # (B, L, Input_Dim)

        # 3. Stem
        x, _ = self.stem_gru(x)
        x = self.stem_dropout(x)  # (B, L, Hidden_Dim)

        # Store outputs for aggregation
        outputs = [x]

        # 4. Residual Blocks
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # 5. Scalar Mixture Aggregation
        agg = self.mixture(outputs)  # (B, L, Hidden_Dim)

        # 6. Output Head
        logits = self.head(agg)  # (B, L, 3)

        return logits


# =========================================================================
# 3. Metrics and Loss
# =========================================================================


def masked_mse_loss(preds, targets, mask):
    """
    Computes MSE loss only on valid (masked) positions.
    """
    # preds: (B, L, 3)
    # targets: (B, L, 3)
    # mask: (B, L) - True where valid

    mask_expanded = mask.unsqueeze(-1).expand_as(preds)  # (B, L, 3)

    loss = (preds - targets) ** 2
    loss = loss * mask_expanded.float()

    # Avoid division by zero
    sum_loss = loss.sum()
    count = mask_expanded.sum() + 1e-8

    return sum_loss / count


def compute_mcrmse(preds, targets, mask):
    """
    Computes Mean Columnwise Root Mean Squared Error.
    Averages the RMSE of each target column independently.
    """
    # preds: (B, L, 3)
    # targets: (B, L, 3)
    # mask: (B, L)

    col_rmses = []
    # Iterate over the 3 target columns
    for i in range(3):
        p_col = preds[:, :, i]
        t_col = targets[:, :, i]

        # Select only valid positions
        # Using boolean indexing flattens the array to 1D containing only valid elements
        diff_sq = (p_col[mask] - t_col[mask]) ** 2

        mse = torch.mean(diff_sq)
        rmse = torch.sqrt(mse)
        col_rmses.append(rmse)

    # Average the RMSEs
    return torch.mean(torch.stack(col_rmses))


# =========================================================================
# 4. Training Engine
# =========================================================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move to device
        sequences = batch["sequence"].to(device)
        loop_types = batch["loop_type"].to(device)
        pair_dists = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        outputs = model(sequences, loop_types, pair_dists)
        loss = masked_mse_loss(outputs, targets, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(sequences, loop_types, pair_dists)

            all_preds.append(outputs)
            all_targets.append(targets)
            all_masks.append(mask)

    # Concatenate all batches
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    score = compute_mcrmse(preds, targets, masks)
    return score.item()


def generate_submission(model, test_loader, device):
    model.eval()
    results = []

    # Target columns in model output: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Submission columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    print("Generating predictions for submission...")

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            ids = batch["id"]

            # (B, L, 3)
            outputs = model(sequences, loop_types, pair_dists)
            outputs = outputs.cpu().numpy()

            # Process each sample in batch
            for i, sample_id in enumerate(ids):
                # Get sequence length (fixed 107)
                seq_len = Config.SEQ_LEN

                # Model predicts 3 columns
                pred_reactivity = outputs[i, :, 0]
                pred_deg_Mg_pH10 = outputs[i, :, 1]
                pred_deg_Mg_50C = outputs[i, :, 2]

                # Zero fill for unscored columns
                zeros = np.zeros(seq_len)

                # Create rows for each position
                for pos in range(seq_len):
                    row_id = f"{sample_id}_{pos}"
                    results.append(
                        {
                            "id_seqpos": row_id,
                            "reactivity": pred_reactivity[pos],
                            "deg_Mg_pH10": pred_deg_Mg_pH10[pos],
                            "deg_pH10": 0.0,
                            "deg_Mg_50C": pred_deg_Mg_50C[pos],
                            "deg_50C": 0.0,
                        }
                    )

    df_sub = pd.DataFrame(results)
    # Ensure column order
    df_sub = df_sub[
        Config.SUBMISSION_COLS.insert(0, "id_seqpos")
        or ["id_seqpos"] + Config.SUBMISSION_COLS
    ]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# =========================================================================
# 5. Main Execution
# =========================================================================


def run_experiment(debug=Config.DEBUG):
    set_seed()
    Config.setup()

    print(f"Starting experiment (Debug={debug})")
    print(f"Device: {Config.DEVICE}")

    # 1. Load Data
    train_dataset = load_data("train", debug=debug)
    val_dataset = load_data("val", debug=debug)
    test_dataset = load_data("test", debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Setup
    model = DenseAggregatedBiGRU(Config).to(Config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 3. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_score = evaluate(model, val_loader, Config.DEVICE)

        # Scheduler Step
        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! Score: {best_score:.6f}")

    print(f"Training complete. Best Val MCRMSE: {best_score:.6f}")

    # 4. Inference
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    generate_submission(model, test_loader, Config.DEVICE)
