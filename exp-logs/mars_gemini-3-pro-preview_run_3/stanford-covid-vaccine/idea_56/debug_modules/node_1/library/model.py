import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, calculate_metric
from library.data import get_loaders, load_data


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module (Lesson 80, 85).
    Implements:
    - Point-to-Point Gather of neighbor state h_j
    - Input Zero-Masking for unpaired bases
    - Bias-Refined Message: m_ij = GELU(W_msg * h_j + b_msg)
    - Stabilized MLP Gate: Sigmoid(W_g2 * GELU(LayerNorm(W_g1 * [h_i, h_j])))
    - Residual Injection and Post-Normalization
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message generation: W_msg * h_j + b_msg
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Gate: MLP on [h_i; h_j]
        # z_raw = W_g1 [h_i; h_j]
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
        # Internal Normalization (Lesson 75)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Norm (Lesson 68)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_index):
        # h: (B, L, H)
        # pair_index: (B, L) containing indices of pairs. -1 if unpaired.

        B, L, H = h.shape

        # 1. Gather h_j
        # Handle -1 indices by clamping to 0, then masking result
        mask = (pair_index != -1).unsqueeze(-1).float()  # (B, L, 1)
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
    """
    Stabilized Decoupled Bias-Refined BiGRU.
    Architecture:
    - 1D Conv Stem
    - 3 Layers of (BiGRU + StructuralInteractionModule)
    - Linear Head
    """

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
            interaction = StructuralInteractionModule(Config.HIDDEN_DIM)

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


def train_pipeline(epochs=Config.EPOCHS, debug=False):
    """
    Executes the training pipeline.
    """
    # Set seeds
    seed_everything(Config.SEED)

    # Load Data
    train_loader, val_loader, _ = get_loaders(debug=debug, load_cached_data=True)

    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDBR_BiGRU().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MCRMSELoss()

    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)  # (B, 68, 5)

            optimizer.zero_grad()
            preds = model(features, pair_index)  # (B, 107, 5)

            # Slice to scored positions for loss calculation
            preds_scored = preds[:, : Config.SEQ_SCORED, :]

            loss = criterion(preds_scored, targets)
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

                # Store full predictions and targets; calculate_metric handles slicing/filtering
                val_preds_list.append(preds.cpu())
                val_targets_list.append(targets.cpu())

        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Calculate Metric (MCRMSE on specific columns and positions)
        val_score = calculate_metric(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.10f} | Val MCRMSE: {val_score:.10f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Training finished. Best Val MCRMSE: {best_score:.10f}")
    return best_model_path


def generate_submission(model_state_path, debug=False):
    """
    Generates submission file using the trained model.
    """
    # Load Data (Need dataframe for IDs)
    _, _, test_df = load_data(debug=debug, load_cached_data=True)
    _, _, test_loader = get_loaders(debug=debug, load_cached_data=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDBR_BiGRU().to(device)

    # Load model state
    state_dict = torch.load(model_state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)

            preds = model(features, pair_index)  # (B, 107, 5)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 5)

    # Format Submission
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Ensure IDs align with predictions
    ids = test_df["id"].values

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
