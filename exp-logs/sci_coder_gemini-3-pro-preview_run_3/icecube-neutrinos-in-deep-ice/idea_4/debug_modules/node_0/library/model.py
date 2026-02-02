import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool, knn_graph
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

import library.config as config
import library.utils as utils

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class NeutrinoGNN(nn.Module):
    def __init__(self):
        super(NeutrinoGNN, self).__init__()

        # Hyperparameters
        self.k = config.K_NEIGHBORS
        node_feat_dim = config.NODE_FEAT_DIM
        hidden_dim = config.HIDDEN_DIM

        # Correcting the Eigen dimension mismatch: utils.py produces 12 features (3 evals + 9 evecs)
        # config.py might say 9, but the data source dictates 12.
        self.eigen_dim = 12

        # 1. Dynamic Edge Convolution Layers
        # EdgeConv(nn) expects an MLP. The MLP input is 2 * node_feat_dim because
        # EdgeConv implicitly concatenates [x_i, x_j - x_i].
        # This captures the relative differences (dx, dy, dz, dt, dq) as requested.

        # Layer 1
        self.conv1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ),
            aggr="max",
        )

        # Layer 2
        self.conv2 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ),
            aggr="max",
        )

        # Layer 3
        self.conv3 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ),
            aggr="max",
        )

        # 2. Fusion Head
        # We concat Global Mean Pool + Global Max Pool + Global Eigen Features
        fusion_input_dim = (hidden_dim * 2) + self.eigen_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.GLOBAL_POOL_DIM),
            nn.BatchNorm1d(config.GLOBAL_POOL_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.GLOBAL_POOL_DIM, config.GLOBAL_POOL_DIM // 2),
            nn.ReLU(),
            nn.Linear(
                config.GLOBAL_POOL_DIM // 2, config.OUTPUT_DIM
            ),  # Output: (nx, ny, nz)
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        global_features = data.global_features  # Shape (Batch, 12)

        # 1. Graph Construction & Convolution
        # Layer 1
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.conv1(x, edge_index)

        # Layer 2
        # We can re-compute graph or reuse. Dynamic graph updates are usually better for particle tracks.
        # However, keeping it static or updating based on features is a choice.
        # Updating based on pos (static geometry) is safer for stability.
        # We reuse edge_index for efficiency as spatial locality dominates.
        x = self.conv2(x, edge_index)

        # Layer 3
        x = self.conv3(x, edge_index)

        # 2. Global Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        # 3. Fusion
        # global_features might need reshaping if it's not (Batch, Dim)
        if global_features.dim() == 3:
            # Sometimes PyG collate might add a dimension, e.g. (Batch, 1, 12)
            global_features = global_features.squeeze(1)

        combined = torch.cat([x_mean, x_max, global_features], dim=1)

        # 4. Prediction
        out = self.head(combined)

        return out


# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, pred, target):
        # Pred: (Batch, 3) unnormalized
        # Target: (Batch, 3) normalized unit vector

        # Cosine Similarity: (A . B) / (|A| |B|)
        # This automatically handles the normalization of 'pred'.
        # Target is assumed to be unit length, but cosine_similarity handles it regardless.
        cos_sim = F.cosine_similarity(pred, target, dim=1)

        # We want to maximize similarity (1.0), so we minimize (1 - similarity)
        loss = 1.0 - cos_sim.mean()
        return loss


# -----------------------------------------------------------------------------
# Training & Inference Utilities
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            # For metric calculation
            # Normalize predictions to unit vectors
            pred_norm = F.normalize(out, p=2, dim=1).cpu().numpy()
            target_np = data.y.cpu().numpy()

            # Convert to spherical for angular distance score
            # utils.angular_dist_score expects (azimuth, zenith)
            # We need to convert Cartesian -> Spherical

            # Process batch
            for i in range(len(pred_norm)):
                px, py, pz = pred_norm[i]
                tx, ty, tz = target_np[i]

                p_az, p_zen = utils.cartesian_to_spherical(px, py, pz)
                t_az, t_zen = utils.cartesian_to_spherical(tx, ty, tz)

                all_preds.append([float(p_az), float(p_zen)])
                all_targets.append([float(t_az), float(t_zen)])

    avg_loss = total_loss / len(loader.dataset)

    # Calculate Metric
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    metric = utils.angular_dist_score(all_targets, all_preds)

    return avg_loss, metric


def train_model(train_loader, val_loader):
    device = config.DEVICE
    print(f"Using device: {device}")

    model = NeutrinoGNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    criterion = CosineLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric (Mean Angular Error): {val_metric:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  -> Model saved to {config.MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # Load best model
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    return model


def generate_submission(model, test_loader):
    device = config.DEVICE
    model.eval()

    event_ids = []
    azimuths = []
    zeniths = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for data in tqdm(test_loader, disable=True):  # Disable tqdm to keep logs clean
            data = data.to(device)
            out = model(data)

            # Normalize
            pred_norm = F.normalize(out, p=2, dim=1).cpu().numpy()
            batch_event_ids = data.event_id.cpu().numpy()

            for i in range(len(pred_norm)):
                px, py, pz = pred_norm[i]
                az, zen = utils.cartesian_to_spherical(px, py, pz)

                event_ids.append(batch_event_ids[i])
                azimuths.append(az)
                zeniths.append(zen)

    # Create DataFrame
    df_sub = pd.DataFrame(
        {"event_id": event_ids, "azimuth": azimuths, "zenith": zeniths}
    )

    # Ensure event_id is int
    df_sub["event_id"] = df_sub["event_id"].astype(int)

    # Sort by event_id just in case, though sample submission order is preferred
    # We will load sample submission to ensure correct order
    try:
        sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
        # Merge to enforce order
        # Rename columns in sample to avoid conflict if needed, but here we just want the index
        df_final = sample_sub[["event_id"]].merge(df_sub, on="event_id", how="left")

        # Fill NaNs if any (shouldn't be)
        if df_final.isnull().any().any():
            print("Warning: Missing predictions for some events. Filling with default.")
            df_final = df_final.fillna(1.0)  # Dummy value

    except Exception as e:
        print(f"Could not align with sample submission: {e}. Saving raw predictions.")
        df_final = df_sub

    df_final.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
