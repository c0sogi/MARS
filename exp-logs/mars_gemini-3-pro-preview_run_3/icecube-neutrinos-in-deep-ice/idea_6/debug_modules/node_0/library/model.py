import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from library.config import Config
from library import utils, data

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class DynEdgeConv(nn.Module):
    """
    Dynamic Edge Convolution Layer.
    Constructs a dynamic graph based on nearest neighbors in the feature space (x,y,z,t)
    and performs message passing.
    """

    def __init__(self, in_channels, out_channels, k=8):
        super(DynEdgeConv, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        # MLP for processing edge features: [x_i, x_j - x_i]
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, N, C) Input node features.
            mask: (Batch, N) Boolean mask (True for valid nodes, False for padding).
        Returns:
            out: (Batch, N, out_channels) Updated node features.
        """
        B, N, C = x.shape

        # 1. KNN Graph Construction
        # We use the first 4 channels (x, y, z, t) for distance calculation
        # to ensure the graph topology is based on spatiotemporal proximity.
        coords = x[:, :, :4]
        dist = torch.cdist(coords, coords)  # (B, N, N)

        # Handle masking: Set distances to/from padded nodes to infinity
        if mask is not None:
            # mask is (B, N). We want (B, N, N) where dist[b, i, j] is inf if mask[b, j] is False
            mask_expanded = mask.unsqueeze(1).expand(B, N, N)
            dist = dist.masked_fill(~mask_expanded, float("inf"))

            # Also mask rows corresponding to padded nodes to prevent them from messing up gradients
            # (though they will be masked out later in pooling)
            # We don't strictly need to mask rows for topk, as they will just find some neighbors.

        # Get k nearest neighbors
        k = min(self.k, N)
        _, indices = torch.topk(dist, k=k, dim=-1, largest=False)  # (B, N, k)

        # 2. Gather neighbor features
        # Create batch index offset
        batch_indices = torch.arange(B, device=x.device).view(-1, 1, 1).expand(B, N, k)

        # Gather neighbors: (B, N, k, C)
        neighbors = x[batch_indices, indices, :]

        # 3. Construct Edge Features
        # Central nodes: (B, N, k, C)
        central = x.unsqueeze(2).expand(B, N, k, C)

        # Edge features: Concatenate central features and relative differences
        edge_features = torch.cat(
            [central, neighbors - central], dim=-1
        )  # (B, N, k, 2*C)

        # 4. Message Passing
        messages = self.mlp(edge_features)  # (B, N, k, out_channels)

        # 5. Aggregation (Max Pooling over neighbors)
        out = torch.max(messages, dim=2)[0]  # (B, N, out_channels)

        return out


class CFDGN(nn.Module):
    """
    Canonical-Frame Dynamic Graph Network.
    """

    def __init__(self):
        super(CFDGN, self).__init__()

        # Data produces 6 features: x, y, z, time, charge, aux
        self.in_channels = 6
        self.hidden_dim = Config.HIDDEN_DIM
        self.k = Config.K_NEIGHBORS

        # Input Embedding
        self.embedding = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim), nn.ReLU()
        )

        # Dynamic Graph Blocks
        self.block1 = DynEdgeConv(self.hidden_dim, self.hidden_dim, k=self.k)
        self.block2 = DynEdgeConv(self.hidden_dim, self.hidden_dim, k=self.k)
        self.block3 = DynEdgeConv(self.hidden_dim, self.hidden_dim, k=self.k)

        # Output Head
        # Global pooling results in 2 * hidden_dim (Mean + Max)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, Config.OUTPUT_DIM),  # Output: 3 (x, y, z vector)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, N, 6)
            mask: (Batch, N)
        """
        # Embedding
        h = self.embedding(x)  # (B, N, H)

        # DynEdge Blocks with Residual Connections
        h1 = self.block1(h, mask)
        h = h + h1

        h2 = self.block2(h, mask)
        h = h + h2

        h3 = self.block3(h, mask)
        h = h + h3

        # Global Pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)

            # Mean Pooling (ignoring padding)
            sum_pool = torch.sum(h * mask_expanded, dim=1)
            count = torch.sum(mask_expanded, dim=1)
            count = torch.clamp(count, min=1.0)
            mean_pool = sum_pool / count

            # Max Pooling (ignoring padding)
            h_masked = h.clone()
            h_masked[~mask] = -1e9  # Set padding to very small value
            max_pool = torch.max(h_masked, dim=1)[0]
        else:
            mean_pool = torch.mean(h, dim=1)
            max_pool = torch.max(h, dim=1)[0]

        # Concatenate Global Features
        global_feat = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2*H)

        # Prediction Head
        out = self.head(global_feat)  # (B, 3)

        # Normalize to unit vector
        out = F.normalize(out, p=2, dim=1)

        return out


# -----------------------------------------------------------------------------
# Training & Evaluation Logic
# -----------------------------------------------------------------------------


def criterion(pred, target):
    """
    Cosine Similarity Loss: 1 - cos(theta)
    """
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    return 1.0 - cos_sim.mean()


def compute_mae(pred, target):
    """
    Mean Angular Error in radians.
    """
    cos_sim = F.cosine_similarity(pred, target, dim=1)
    # Clamp to avoid numerical issues with acos
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cos_sim)
    return angles.mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target"].to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()

        pred = model(x, mask)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_mae += compute_mae(pred, target) * batch_size
        total_samples += batch_size

    return running_loss / total_samples, running_mae / total_samples


def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            batch_size = x.size(0)

            pred = model(x, mask)
            loss = criterion(pred, target)

            running_loss += loss.item() * batch_size
            running_mae += compute_mae(pred, target) * batch_size
            total_samples += batch_size

    return running_loss / total_samples, running_mae / total_samples


def train_model(train_loader, val_loader):
    """
    Main training loop with Early Stopping.
    """
    Config.setup_directories()
    device = torch.device(Config.DEVICE)
    print(f"Initializing CF-DGN on {device}...")

    model = CFDGN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_val_mae = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, device)

        scheduler.step()

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1:02d}/{Config.EPOCHS} | "
            f"Time: {duration:.1f}s | "
            f"Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("  -> Model Saved (New Best)")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model for return
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    return model


# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves to CSV.
    Handles the inverse rotation from Canonical Frame to Global Frame.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    results = []
    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            rotation = batch["rotation"].numpy()  # (B, 3, 3)
            event_ids = batch["event_id"].numpy()

            # Predict direction in Canonical Frame
            # pred_local: (B, 3)
            pred_local = model(x, mask).cpu().numpy()

            # Inverse Rotation: Transform back to Global Frame
            # Global vector v_global = R^T @ v_local
            # Since our rotation matrices are orthogonal, R^T is the inverse.
            # We perform batch matrix multiplication.
            # einsum: 'bij, bj -> bi' where i is global dim, j is local dim.
            # We want: v_global[i] = sum_j (R[j, i] * v_local[j]) -- wait, R is (3,3).
            # Let's be precise:
            # v_local = R @ v_global  => v_global = R^T @ v_local
            # For a single item: v_g = R.T . v_l
            # Batch wise: v_g[b, i] = sum_j (R[b, j, i] * v_l[b, j])

            pred_global = np.einsum("bji,bj->bi", rotation, pred_local)

            # Convert Cartesian to Spherical (Azimuth, Zenith)
            az, ze = utils.cartesian_to_spherical(
                pred_global[:, 0], pred_global[:, 1], pred_global[:, 2]
            )

            # Collect results
            for eid, a, z in zip(event_ids, az, ze):
                results.append({"event_id": int(eid), "azimuth": a, "zenith": z})

    # Create DataFrame and Save
    df_sub = pd.DataFrame(results)

    # Ensure column order
    df_sub = df_sub[["event_id", "azimuth", "zenith"]]

    # Save
    Config.setup_directories()
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(
        f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} predictions."
    )


def run_pipeline():
    """
    End-to-end execution: Data Loading -> Training -> Inference.
    """
    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_dataloaders()

    # 2. Train
    model = train_model(train_loader, val_loader)

    # 3. Predict
    generate_submission(model, test_loader)
