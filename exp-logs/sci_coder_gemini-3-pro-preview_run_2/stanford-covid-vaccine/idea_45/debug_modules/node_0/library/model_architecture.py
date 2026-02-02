import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from library.loss_metric import MCRMSELoss, GlobalMetricsTracker
from library.data_processor import get_dataloaders

# ==================================================================================
# CONFIGURATION & REPRODUCIBILITY
# ==================================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class RobustDilatedBlock(nn.Module):
    """
    Standard Full-Rank Convolution Block with Layer Normalization.
    Structure: LN -> ReLU -> Conv(k=3, d=d) -> LN -> ReLU -> Conv(k=1) -> Dropout
    Uses GroupNorm(1, C) as a channel-first implementation of LayerNorm.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super(RobustDilatedBlock, self).__init__()

        # LayerNorm 1 (Pre-activation)
        self.ln1 = nn.GroupNorm(1, in_channels)
        self.act1 = nn.ReLU()

        # Standard Dilated Convolution (k=3)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )

        # LayerNorm 2
        self.ln2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.ReLU()

        # Pointwise Convolution (k=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.drop(out)
        return out


class DenseBackbone(nn.Module):
    """
    Stack of RobustDilatedBlocks with Dense Connectivity.
    """

    def __init__(self, in_channels, growth_rate=64, num_layers=6):
        super(DenseBackbone, self).__init__()
        self.layers = nn.ModuleList()

        # Dilations: 1, 2, 4, 8, 16, 32
        dilations = [2**i for i in range(num_layers)]

        current_channels = in_channels
        for d in dilations:
            block = RobustDilatedBlock(
                in_channels=current_channels, out_channels=growth_rate, dilation=d
            )
            self.layers.append(block)
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features for input
            in_feat = torch.cat(features, dim=1)
            out_feat = layer(in_feat)
            features.append(out_feat)

        # Return concatenation of all block outputs (excluding original input)
        # Or typically in DenseNet, we pass the full stack.
        # Here we return the full concatenated stack.
        return torch.cat(features, dim=1)


class DenseFeedbackModule(nn.Module):
    """
    Lightweight Dense TCN for processing recycled predictions.
    """

    def __init__(self, in_channels, growth_rate=16, num_layers=4, out_dim=32):
        super(DenseFeedbackModule, self).__init__()
        self.layers = nn.ModuleList()

        # Dilations: 1, 2, 4, 8
        dilations = [2**i for i in range(num_layers)]

        current_channels = in_channels
        for d in dilations:
            block = RobustDilatedBlock(
                in_channels=current_channels, out_channels=growth_rate, dilation=d
            )
            self.layers.append(block)
            current_channels += growth_rate

        # Final projection to fixed embedding size
        self.project = nn.Conv1d(current_channels, out_dim, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            in_feat = torch.cat(features, dim=1)
            out_feat = layer(in_feat)
            features.append(out_feat)

        full_stack = torch.cat(features, dim=1)
        return self.project(full_stack)


class RDFRN(nn.Module):
    def __init__(self, seq_len=107, pred_len=68):
        super(RDFRN, self).__init__()
        self.pred_len = pred_len

        # Input Dimensions
        # Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
        self.input_dim = 18

        # 1. Main Backbone
        self.backbone = DenseBackbone(
            in_channels=self.input_dim, growth_rate=64, num_layers=6
        )

        # Projection of backbone output to Latent Z
        self.backbone_proj = nn.Conv1d(self.backbone.out_channels, 64, kernel_size=1)

        # 2. Feedback Module
        # Preds(5) + Struct(3) + Loop(7) = 15
        self.feedback_input_dim = 15
        self.feedback_module = DenseFeedbackModule(
            in_channels=self.feedback_input_dim, growth_rate=16, out_dim=32
        )

        # 3. Aggregation & Head
        # Fusion Input: (Z_self + E_fb_self) + (Z_partner + E_fb_partner)
        # (64 + 32) * 2 = 192
        self.rnn_input_dim = 192
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.head = nn.Linear(256 * 2, 5)

    def forward_pass(self, z, feedback_emb, partner_indices):
        """
        Executes Interaction, Aggregation, and Projection.
        z: (B, 64, L)
        feedback_emb: (B, 32, L)
        partner_indices: (B, L)
        """
        B, _, L = z.shape

        # Concatenate Self Vectors: [Z, E_fb] -> (B, 96, L)
        self_feat = torch.cat([z, feedback_emb], dim=1)

        # Augmented Gather for Partner Vectors
        # Prepare indices for gather: (B, 1, L) -> expand to (B, 96, L)
        # Mask -1 indices to 0 temporarily
        p_idx = partner_indices.clone()
        mask_unpaired = p_idx == -1
        p_idx[mask_unpaired] = 0

        # Expand for gather
        gather_idx = p_idx.unsqueeze(1).expand(-1, self_feat.size(1), -1)  # (B, 96, L)

        # Gather
        partner_feat = torch.gather(self_feat, 2, gather_idx)  # (B, 96, L)

        # Apply Zero-Mask to unpaired positions
        # mask_unpaired is (B, L), expand to (B, 96, L)
        mask_expanded = mask_unpaired.unsqueeze(1).expand_as(partner_feat)
        partner_feat[mask_expanded] = 0.0

        # Fusion: Concatenate Self + Partner -> (B, 192, L)
        fused = torch.cat([self_feat, partner_feat], dim=1)

        # RNN Aggregation
        # Permute to (B, L, C) for RNN
        fused_perm = fused.permute(0, 2, 1)
        gru_out, _ = self.gru(fused_perm)

        # Projection
        preds = self.head(gru_out)  # (B, L, 5)

        return preds

    def forward(self, inputs, partner_indices):
        """
        inputs: (B, L, 18) -> Permute to (B, 18, L) inside
        partner_indices: (B, L)
        """
        # Permute inputs to (B, C, L)
        x = inputs.permute(0, 2, 1)

        # 1. Static Backbone
        backbone_out = self.backbone(x)
        z = self.backbone_proj(backbone_out)  # (B, 64, L)

        # Extract Raw Topology for Feedback (Struct(3) + Loop(7))
        # Channels 4 to 13 (indices 4:14) in original input
        # Input map: Seq(0-3), Struct(4-6), Loop(7-13), PartnerID(14-17)
        raw_topology = x[:, 4:14, :]  # (B, 10, L)

        # ==========================
        # PASS 1: Zero Feedback
        # ==========================
        B, _, L = z.shape
        # Initialize zero feedback predictions (B, 5, L)
        y0 = torch.zeros(B, 5, L, device=z.device)

        # Construct Feedback Input
        fb_in_1 = torch.cat([y0, raw_topology], dim=1)  # (B, 15, L)
        e_fb_1 = self.feedback_module(fb_in_1)  # (B, 32, L)

        # Run Interaction + Head
        y1 = self.forward_pass(z, e_fb_1, partner_indices)  # (B, L, 5)

        # ==========================
        # PASS 2: Recycled Feedback
        # ==========================
        # Detach gradients
        r = y1.detach()  # (B, L, 5)

        # Mask unscored columns (indices 2 and 4: deg_pH10, deg_50C)
        # Keep 0, 1, 3. Zero out 2, 4.
        mask_cols = torch.tensor([1, 1, 0, 1, 0], device=z.device, dtype=torch.float32)
        r_masked = r * mask_cols.view(1, 1, 5)

        # Permute to (B, 5, L)
        r_perm = r_masked.permute(0, 2, 1)

        # Construct Feedback Input
        fb_in_2 = torch.cat([r_perm, raw_topology], dim=1)
        e_fb_2 = self.feedback_module(fb_in_2)

        # Run Interaction + Head
        y2 = self.forward_pass(z, e_fb_2, partner_indices)

        return y1, y2


# ==================================================================================
# TRAINING UTILITIES
# ==================================================================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        y1, y2 = model(inputs, partner_indices)

        # Loss Calculation: MCRMSE(y2) + 0.5 * MCRMSE(y1)
        loss2 = criterion(y2, targets)
        loss1 = criterion(y1, targets)
        loss = loss2 + 0.5 * loss1

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    tracker = GlobalMetricsTracker()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            _, y2 = model(inputs, partner_indices)

            tracker.update(y2, targets)

    return tracker.compute()


def predict_test(model, loader, device):
    model.eval()
    preds_list = []
    ids_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            ids = batch["id"]

            _, y2 = model(inputs, partner_indices)

            preds_list.append(y2.cpu().numpy())
            ids_list.extend(ids)

    return np.concatenate(preds_list, axis=0), ids_list


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================


def run_training(epochs=15, batch_size=32):
    print(f"Initializing RDF-RN Model on {DEVICE}...")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cached_data=True
    )

    model = RDFRN().to(DEVICE)
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_mcrmse = float("inf")
    patience_counter = 0
    early_stopping_patience = 6

    print("Starting Training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE)
        val_mcrmse = val_metrics["mcrmse"]

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), "./working/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Best Validation MCRMSE: {best_mcrmse:.6f}")

    # Load best model for inference
    model.load_state_dict(torch.load("./working/best_model.pth", map_location=DEVICE))

    # Generate Submission
    print("Generating Submission...")
    test_preds, test_ids = predict_test(model, test_loader, DEVICE)

    # Format submission
    # Need to flatten predictions: id_seqpos
    submission_rows = []
    seq_scored = 68

    for i, sample_id in enumerate(test_ids):
        # preds shape (107, 5)
        # We need to output all 107 positions, but only first 68 are scored.
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        sample_preds = test_preds[i]  # (107, 5)

        for pos in range(sample_preds.shape[0]):
            row_id = f"{sample_id}_{pos}"
            vals = sample_preds[pos]
            # Clip values to valid range if necessary (though not strictly required by metric)
            # vals = np.clip(vals, 0, None)

            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    os.makedirs("./submission", exist_ok=True)
    sub_df.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv with {len(sub_df)} rows.")


# To run the training:
# run_training(epochs=20, batch_size=32)
