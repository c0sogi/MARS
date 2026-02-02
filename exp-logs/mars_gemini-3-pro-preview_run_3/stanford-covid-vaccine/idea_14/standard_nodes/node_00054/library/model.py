import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.config import Config
from library.utils import mcrmse_loss, global_mcrmse_metric


class GatedSpatialInjection(nn.Module):
    """
    Gated Latent Spatial Injection Layer.
    Injects structural context into local features using a learnable gating mechanism.

    Mechanism:
    1. Gather paired features h_j for each position h_i based on pair_indices.
    2. Compute a trust gate g_ij = Sigmoid(W_gate * [h_i; h_j]).
    3. Update h_i = h_i + g_ij * (W_proj * h_j).
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # Gate computation: takes concatenated [h_i; h_j] -> gate value
        self.gate_net = nn.Linear(feature_dim * 2, feature_dim)

        # Projection for the paired feature before injection
        self.proj_net = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Feature_Dim)
            pair_indices: Tensor of shape (Batch, Seq_Len) containing indices of paired bases.
                          -1 indicates unpaired bases.
        """
        B, L, C = x.shape

        # 1. Create a mask for valid pairs (where index != -1)
        valid_mask = pair_indices != -1  # (B, L)

        # 2. Prepare indices for gathering.
        # Replace -1 with 0 to ensure valid indices for gather (masked later).
        safe_indices = pair_indices.clone()
        safe_indices[~valid_mask] = 0

        # Expand indices for gathering across the feature dimension
        # shape: (B, L, C)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, C)

        # 3. Gather paired features h_j
        # h_paired[b, i, :] = x[b, safe_indices[b, i], :]
        h_paired = torch.gather(x, 1, gather_indices)

        # 4. Compute Gate
        # Concatenate h_i (x) and h_j (h_paired)
        combined = torch.cat([x, h_paired], dim=-1)  # (B, L, 2*C)
        gate = torch.sigmoid(self.gate_net(combined))  # (B, L, C)

        # 5. Project Paired Feature
        h_paired_proj = self.proj_net(h_paired)  # (B, L, C)

        # 6. Apply Update with Mask
        # Only update if the position is actually paired
        mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L, 1)

        # h_new = h_i + mask * (gate * h_j_proj)
        update = mask_expanded * (gate * h_paired_proj)
        x_new = x + update

        return x_new


class RNAModel(nn.Module):
    """
    Latent Structure-Gated BiGRU Architecture.

    Components:
    1. Conv1d Stem: Projects sparse one-hot inputs to dense local features.
    2. Gated Spatial Injection: Incorporates long-range structural dependencies.
    3. BiGRU Backbone: Captures sequential context.
    4. Linear Head: Predicts targets.
    """

    def __init__(self):
        super().__init__()

        # 1. Convolutional Stem
        # Projects (B, 14, L) -> (B, 256, L)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_DIM,
                out_channels=Config.CONV_FILTERS,
                kernel_size=Config.CONV_KERNEL,
                padding=Config.CONV_KERNEL // 2,
            ),
            nn.GELU(),
        )

        # 2. Gated Spatial Injection
        self.spatial_injection = GatedSpatialInjection(Config.CONV_FILTERS)

        # 3. Backbone (BiGRU)
        self.gru = nn.GRU(
            input_size=Config.CONV_FILTERS,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0.0,
        )

        # 4. Output Head
        # BiGRU outputs (B, L, 2*Hidden) -> (B, L, 768)
        self.head = nn.Linear(Config.HIDDEN_DIM * 2, Config.OUTPUT_DIM)

    def forward(self, inputs, pair_indices):
        """
        Args:
            inputs: (Batch, Seq_Len, 14)
            pair_indices: (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.transpose(1, 2)

        # Conv Stem
        x = self.conv_stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Spatial Injection
        x = self.spatial_injection(x, pair_indices)

        # BiGRU Backbone
        # output: (B, L, 2*Hidden)
        x, _ = self.gru(x)

        # Output Head
        logits = self.head(x)

        return logits


def train_model(train_loader, val_loader):
    """
    Executes the training pipeline with the specified strategy:
    - AdamW Optimizer
    - Cosine Annealing Scheduler
    - Gradient Clipping
    - MCRMSE Loss
    - Early Stopping based on Global MCRMSE
    """
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    model = RNAModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_val_score = float("inf")
    best_epoch = 0

    # Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            inputs = batch["sequence"].to(device)
            pair_indices = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            preds = model(inputs, pair_indices)

            loss = mcrmse_loss(preds, targets)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["sequence"].to(device)
                pair_indices = batch["pair_index"].to(device)
                targets = batch["targets"].to(device)

                preds = model(inputs, pair_indices)

                val_preds.append(preds.cpu())
                val_targets.append(targets.cpu())

        # Calculate Global Metric
        val_score = global_mcrmse_metric(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(
        f"Training complete. Best Val MCRMSE: {best_val_score:.6f} at Epoch {best_epoch+1}"
    )
    return best_val_score


def predict(test_loader):
    """
    Generates predictions for the test set using the best saved model.
    Returns:
        ids (list): List of sample IDs.
        preds (np.ndarray): Array of predictions sliced to seq_scored.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    model = RNAModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Warning: No model checkpoint found. Using initialized model.")

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["sequence"].to(device)
            pair_indices = batch["pair_index"].to(device)
            ids = batch["id"]

            preds = model(inputs, pair_indices)

            # Move to CPU
            preds = preds.cpu().numpy()

            # Slice to scored sequence length
            preds = preds[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate
    final_preds = np.concatenate(all_preds, axis=0)
    return all_ids, final_preds
