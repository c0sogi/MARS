import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import from provided libraries
from library.config import Config, direction_loss
from library.data import IceCubeDataset, process_batch
from library.geometry import load_sensor_geometry
from library.utils import seed_everything, direction_to_angles, angular_error

# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------


class DynEdgeConv(nn.Module):
    """
    Dynamic Edge Convolution Block.
    Constructs a k-NN graph based on spatial coordinates and aggregates features.
    """

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        # MLP applied to edge features: [x_i, x_j - x_i] -> 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, C].
               Assumes first 3 channels are (x, y, z) for distance computation.
        Returns:
            Tensor of shape [B, N, out_channels].
        """
        B, N, C = x.shape

        # 1. k-NN Graph Construction
        # Use spatial coordinates (first 3 features) for distance
        pos = x[:, :, :3]

        # Compute pairwise distance matrix [B, N, N]
        dist = torch.cdist(pos, pos)

        # Mask self-loops by adding a large value to the diagonal
        eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        dist = dist + eye * 1e9

        # Get indices of k nearest neighbors [B, N, k]
        _, idx = dist.topk(self.k, largest=False, dim=-1)

        # 2. Gather Neighbor Features
        # Create batch index to gather correctly
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, self.k)

        # Gather neighbors: [B, N, k, C]
        neighbors = x[batch_idx, idx, :]

        # 3. Construct Edge Features
        # Central node: [B, N, 1, C] -> expand to [B, N, k, C]
        x_central = x.unsqueeze(2).expand(-1, -1, self.k, -1)

        # Edge feature: [x_i, x_j - x_i]
        edge_feat = torch.cat([x_central, neighbors - x_central], dim=-1)

        # 4. Message Passing (MLP)
        out = self.mlp(edge_feat)  # [B, N, k, out_channels]

        # 5. Aggregation (Max Pooling)
        out = out.max(dim=2)[0]  # [B, N, out_channels]

        return out


class DV_AGN(nn.Module):
    """
    Dual-View Attentive Graph Network.
    Processes Raw and Canonical views in parallel and fuses them via attention.
    """

    def __init__(self):
        super().__init__()

        input_dim = 6  # x, y, z, t, q, aux
        embed_dim = Config.EMBED_DIM
        hidden_dim = Config.HIDDEN_DIM
        k = Config.K_NEIGHBORS
        dropout = Config.DROPOUT

        # --- Encoders ---
        self.embed_raw = nn.Linear(input_dim, embed_dim)
        self.embed_canon = nn.Linear(input_dim, embed_dim)

        # Raw Stream GNN
        self.gnn_raw = nn.ModuleList(
            [
                DynEdgeConv(embed_dim, embed_dim, k=k),
                DynEdgeConv(embed_dim, embed_dim, k=k),
            ]
        )

        # Canonical Stream GNN
        self.gnn_canon = nn.ModuleList(
            [
                DynEdgeConv(embed_dim, embed_dim, k=k),
                DynEdgeConv(embed_dim, embed_dim, k=k),
            ]
        )

        # --- Cross Attention Fusion ---
        # Query = Canonical, Key/Value = Raw
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

        # --- Global Pooling ---
        # Attention-based pooling
        self.pool_attn = nn.Sequential(nn.Linear(embed_dim, 1), nn.Tanh())

        # --- Prediction Head ---
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # nx, ny, nz
        )

    def forward(self, x_raw, x_canon):
        # x_raw, x_canon: [B, N, 6]

        # 1. Embedding
        h_raw = self.embed_raw(x_raw)  # [B, N, D]
        h_canon = self.embed_canon(x_canon)  # [B, N, D]

        # 2. GNN Streams (with Residuals)
        for layer in self.gnn_raw:
            h_raw = h_raw + layer(h_raw)

        for layer in self.gnn_canon:
            h_canon = h_canon + layer(h_canon)

        # 3. Cross Attention Fusion
        # We query the Raw stream (context) using the Canonical stream (focus)
        Q = self.query(h_canon)  # [B, N, D]
        K = self.key(h_raw)  # [B, N, D]
        V = self.value(h_raw)  # [B, N, D]

        # Attention Scores: [B, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Context: [B, N, D]
        context = torch.matmul(attn_weights, V)

        # Fuse: Add context to canonical features
        h_fused = h_canon + context

        # 4. Global Pooling
        # Compute importance weights for each node
        pool_logits = self.pool_attn(h_fused)  # [B, N, 1]
        pool_weights = torch.softmax(pool_logits, dim=1)  # [B, N, 1]

        # Weighted sum
        h_global = (h_fused * pool_weights).sum(dim=1)  # [B, D]

        # 5. Prediction
        out = self.head(h_global)  # [B, 3]

        return out


# -----------------------------------------------------------------------------
# Training & Inference Logic
# -----------------------------------------------------------------------------


def train_model(load_cached_data=True):
    """
    Executes the training pipeline.
    """
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Loading sensor geometry from {Config.SENSOR_GEO_PATH}...")
    sensor_map = load_sensor_geometry(Config.SENSOR_GEO_PATH)

    # Load Metadata
    print("Loading metadata...")
    train_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "train_metadata.parquet")
    )
    val_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "val_metadata.parquet")
    )

    # Get Batch IDs
    train_batches = train_meta["batch_id"].unique()
    val_batches = val_meta["batch_id"].unique()

    # Initialize Model
    model = DV_AGN().to(Config.DEVICE)
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler placeholder
    scheduler = None

    best_val_mae = float("inf")
    patience = 3
    no_improve = 0

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0
        train_batches_count = 0

        # Shuffle train batches
        np.random.shuffle(train_batches)

        for batch_id in train_batches:
            # Process Batch
            X_raw, X_canon, y = process_batch(
                batch_id,
                train_meta,
                sensor_map,
                mode="train",
                load_cached_data=load_cached_data,
            )

            dataset = IceCubeDataset(X_raw, X_canon, y)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize scheduler once we know the number of steps
            if scheduler is None:
                total_steps = len(train_batches) * len(loader) * Config.EPOCHS
                scheduler = OneCycleLR(
                    optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps
                )

            for xr, xc, targets in loader:
                xr = xr.to(Config.DEVICE)
                xc = xc.to(Config.DEVICE)
                az = targets[:, 0].to(Config.DEVICE)
                ze = targets[:, 1].to(Config.DEVICE)

                optimizer.zero_grad()
                pred = model(xr, xc)
                loss = direction_loss(pred, az, ze)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                train_loss_sum += loss.item()

            train_batches_count += 1

            # Memory cleanup
            del X_raw, X_canon, y, dataset, loader
            gc.collect()

        avg_train_loss = train_loss_sum / max(1, train_batches_count)

        # Validation
        model.eval()
        val_errors = []

        with torch.no_grad():
            for batch_id in val_batches:
                X_raw, X_canon, y = process_batch(
                    batch_id,
                    val_meta,
                    sensor_map,
                    mode="val",
                    load_cached_data=load_cached_data,
                )

                dataset = IceCubeDataset(X_raw, X_canon, y)
                loader = DataLoader(
                    dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                )

                for xr, xc, targets in loader:
                    xr = xr.to(Config.DEVICE)
                    xc = xc.to(Config.DEVICE)
                    az = targets[:, 0].numpy()
                    ze = targets[:, 1].numpy()

                    pred = model(xr, xc)
                    errors = angular_error(pred, az, ze)
                    val_errors.extend(errors)

                del X_raw, X_canon, y, dataset, loader
                gc.collect()

        avg_val_mae = np.mean(val_errors)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.8f}, Val MAE={avg_val_mae:.8f}"
        )

        # Checkpointing
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "model.pth")
            )
            print(f"New best model saved with MAE: {best_val_mae:.8f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break


def generate_submission(load_cached_data=True):
    """
    Generates submission file for the test set.
    """
    seed_everything(Config.SEED)
    Config.setup()

    model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    if not os.path.exists(model_path):
        print("No trained model found. Cannot generate submission.")
        return

    print(f"Loading sensor geometry...")
    sensor_map = load_sensor_geometry(Config.SENSOR_GEO_PATH)

    print("Loading test metadata...")
    test_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "test_metadata.parquet")
    )
    test_batches = test_meta["batch_id"].unique()

    # Load Model
    model = DV_AGN().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    all_ids = []
    all_azimuth = []
    all_zenith = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch_id in test_batches:
            X_raw, X_canon, ids = process_batch(
                batch_id,
                test_meta,
                sensor_map,
                mode="test",
                load_cached_data=load_cached_data,
            )

            dataset = IceCubeDataset(
                X_raw, X_canon, ids
            )  # Pass ids as y for dataset compatibility
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            for xr, xc, batch_ids in loader:
                xr = xr.to(Config.DEVICE)
                xc = xc.to(Config.DEVICE)

                pred = model(xr, xc)

                # Convert vector to angles
                az, ze = direction_to_angles(pred)

                all_azimuth.extend(az.cpu().numpy())
                all_zenith.extend(ze.cpu().numpy())
                all_ids.extend(batch_ids.numpy())

            del X_raw, X_canon, ids, dataset, loader
            gc.collect()

    # Create Submission DataFrame
    print("Saving submission...")
    submission = pd.DataFrame(
        {"event_id": all_ids, "azimuth": all_azimuth, "zenith": all_zenith}
    )

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
