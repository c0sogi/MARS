import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import copy
from library.config import Config
from library.data import get_loaders
from library.loss import MCRMSELoss

# =========================================================================
# Model Components
# =========================================================================


class HybridStem(nn.Module):
    """
    Hybrid Input Stem that preserves raw features while generating local context.
    Branch A: Identity (Raw Features)
    Branch B: Conv1d(k=3) -> LN -> SiLU
    """

    def __init__(self, in_channels, context_channels):
        super().__init__()
        self.branch_context = nn.Sequential(
            nn.Conv1d(in_channels, context_channels, kernel_size=3, padding=1),
            # LayerNorm in PyTorch expects (N, L, C) for normalized_shape=C
            # We handle permutation in forward
            nn.LayerNorm(context_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (B, L, C_in)

        # Branch A: Identity
        branch_a = x

        # Branch B: Context
        # Conv1d expects (B, C, L)
        x_perm = x.permute(0, 2, 1)
        out_b = self.branch_context[0](x_perm)  # Conv
        out_b = out_b.permute(0, 2, 1)  # (B, L, C)
        out_b = self.branch_context[1](out_b)  # LN
        out_b = self.branch_context[2](out_b)  # SiLU

        # Concatenate: (B, L, C_in + C_context)
        return torch.cat([branch_a, out_b], dim=2)


class DenseDilatedBlock(nn.Module):
    """
    Single block for the Dense Dilated TCN backbone.
    Structure: Conv(3x3, d) -> LN -> SiLU -> Conv(1x1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=0.1):
        super().__init__()
        self.net_conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        self.pointwise = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C_in)
        x_in = x.permute(0, 2, 1)

        out = self.net_conv(x_in)
        out = out.permute(0, 2, 1)
        out = self.norm1(out)
        out = self.act1(out)

        out = out.permute(0, 2, 1)
        out = self.pointwise(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.dropout(out)
        return out


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Processes recycled predictions with channel masking for unscored targets.
    """

    def __init__(self, in_channels=5, growth_rate=16, out_channels=32):
        super().__init__()
        # Lightweight Dense TCN
        self.layers = nn.ModuleList()
        dilations = [1, 2, 4, 8]
        current_dim = in_channels

        for d in dilations:
            self.layers.append(DenseDilatedBlock(current_dim, growth_rate, dilation=d))
            current_dim += growth_rate

        self.projection = nn.Linear(current_dim, out_channels)

    def forward(self, x):
        # x: (B, L, 5)
        # Mask unscored targets: deg_pH10 (idx 2) and deg_50C (idx 4)
        # ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Mask: [1, 1, 0, 1, 0]
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], device=x.device).view(1, 1, 5)
        x_masked = x * mask

        features = [x_masked]
        curr = x_masked

        for layer in self.layers:
            out = layer(curr)  # Returns (B, L, growth_rate)
            features.append(out)
            # Dense connection: concat all previous
            curr = torch.cat(features, dim=2)

        # Project
        return self.projection(curr)


class AHIRN(nn.Module):
    """
    Anchored Hybrid-Input Recurrent Network (AHI-RN).
    """

    def __init__(self):
        super().__init__()

        # 1. Hybrid Stem
        # Input channels: 4(Seq)+3(Struct)+7(Loop)+4(Partner) = 18
        self.stem = HybridStem(in_channels=18, context_channels=Config.HIDDEN_DIM)
        stem_out_dim = 18 + Config.HIDDEN_DIM

        # 2. Backbone (Dense Dilated TCN)
        self.backbone_blocks = nn.ModuleList()
        current_dim = stem_out_dim

        for d in Config.DILATIONS:
            # Growth rate is set to HIDDEN_DIM
            blk = DenseDilatedBlock(
                current_dim, Config.HIDDEN_DIM, dilation=d, dropout=Config.DROPOUT
            )
            self.backbone_blocks.append(blk)
            # Dense connection: Input grows by growth_rate at each step
            current_dim += Config.HIDDEN_DIM

        self.latent_proj = nn.Linear(current_dim, Config.LATENT_DIM)

        # 3. Feedback Module
        self.feedback = FeedbackModule(
            in_channels=5, growth_rate=16, out_channels=Config.FEEDBACK_EMBED_DIM
        )

        # 4. Aggregation (Bi-GRU)
        # Input: (Latent(64) + Feedback(32)) * 2 (Self+Partner) = 192
        gru_in_dim = (Config.LATENT_DIM + Config.FEEDBACK_EMBED_DIM) * 2
        self.gru = nn.GRU(
            gru_in_dim, Config.HIDDEN_DIM, batch_first=True, bidirectional=True
        )

        # 5. Head
        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

    def forward(self, inputs, partner_indices):
        # inputs: (B, L, 18)
        # partner_indices: (B, L)

        # --- Backbone (Static) ---
        x = self.stem(inputs)
        features = [x]
        curr = x

        for blk in self.backbone_blocks:
            out = blk(curr)
            features.append(out)
            curr = torch.cat(features, dim=2)

        z = self.latent_proj(curr)  # (B, L, 64)

        # --- Iterative Refinement ---
        # Pass 1: Zero feedback
        y_prev = torch.zeros((inputs.size(0), inputs.size(1), 5), device=inputs.device)
        y1 = self.refinement_step(z, y_prev, partner_indices)

        # Pass 2: Detached feedback from Pass 1
        y_prev_2 = y1.detach()
        y2 = self.refinement_step(z, y_prev_2, partner_indices)

        return y1, y2

    def refinement_step(self, z, y_prev, partner_indices):
        # z: (B, L, 64)
        # y_prev: (B, L, 5)

        # Feedback Embedding
        e_fb = self.feedback(y_prev)  # (B, L, 32)

        # Concatenate Self Vector
        self_vec = torch.cat([z, e_fb], dim=2)  # (B, L, 96)

        # Augmented Gather (Partner Vector)
        batch_size, seq_len, dim = self_vec.shape

        # Handle -1 indices in partner_indices
        safe_indices = partner_indices.clone()
        mask_unpaired = safe_indices == -1
        safe_indices[mask_unpaired] = 0  # Clamp to valid index for gather

        # Expand indices for gather: (B, L, D)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, dim)

        partner_vec = torch.gather(self_vec, 1, gather_indices)  # (B, L, 96)

        # Zero out vectors where partner was -1
        partner_vec[mask_unpaired.unsqueeze(-1).expand(-1, -1, dim)] = 0.0

        # Fusion
        combined = torch.cat([self_vec, partner_vec], dim=2)  # (B, L, 192)

        # Global Aggregation
        gru_out, _ = self.gru(combined)  # (B, L, 128)

        # Head
        logits = self.head(gru_out)

        return logits


# =========================================================================
# Training & Inference Logic
# =========================================================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partners = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        y1, y2 = model(inputs, partners)

        # Loss on both passes
        loss1 = criterion(y1, targets)
        loss2 = criterion(y2, targets)

        # Weighted sum
        loss = loss2 + Config.AUX_LOSS_WEIGHT * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partners = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            _, y2 = model(inputs, partners)

            # Validation metric uses final prediction only
            loss = criterion(y2, targets)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def predict_test(model, loader, device):
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partners = batch["partner_indices"].to(device)
            batch_ids = batch["id"]

            _, y2 = model(inputs, partners)

            preds.append(y2.cpu().numpy())
            ids.extend(batch_ids)

    return np.concatenate(preds, axis=0), ids


def run_training_and_inference():
    # Setup
    device = Config.DEVICE
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Init Model
    model = AHIRN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = MCRMSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"  New best model saved! Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    test_preds, test_ids = predict_test(model, test_loader, device)

    # Format Submission
    print("Generating submission file...")
    submission_data = []
    targets = Config.ALL_TARGETS

    for i, sample_id in enumerate(test_ids):
        pred_matrix = test_preds[i]  # (107, 5)

        for pos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{pos}"
            row_vals = pred_matrix[pos].tolist()
            submission_data.append([row_id] + row_vals)

    columns = ["id_seqpos"] + targets
    sub_df = pd.DataFrame(submission_data, columns=columns)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
