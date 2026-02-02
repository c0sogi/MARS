import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from library.config import Config
from library.utils import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
)
from library.data_loader import get_dataloaders, get_test_loader

# ==========================================
# Model Architecture
# ==========================================


class DynEdgeLayer(nn.Module):
    """
    Dynamic Edge Convolution Layer.
    Constructs a dynamic graph based on k-nearest neighbors in feature space
    and performs message passing.
    """

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        # MLP applied to edge features: [x_i, x_j - x_i]
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape

        # 1. k-NN Graph Construction
        # Compute pairwise distances
        # For N=196, brute force cdist is efficient enough
        x_flat = x.view(B, N, C)
        dist = torch.cdist(x_flat, x_flat)  # (B, N, N)

        # Get indices of k nearest neighbors (excluding self if possible, but topk includes self)
        # We take top k. Since distance is 0 for self, it will be included.
        # This is standard for DynEdge.
        _, idx = torch.topk(-dist, k=self.k, dim=-1)  # (B, N, k)

        # 2. Gather Neighbor Features
        # Vectorized gathering
        # Offset indices by batch * N to flatten
        batch_offsets = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
        idx_flat = (idx + batch_offsets).view(-1)

        x_reshaped = x.view(B * N, C)
        neighbors = x_reshaped[idx_flat].view(B, N, self.k, C)

        # 3. Compute Edge Features
        # center: (B, N, 1, C)
        center = x.unsqueeze(2).expand(-1, -1, self.k, -1)

        # Edge feature: concatenation of center and relative difference
        edge_feature = torch.cat([center, neighbors - center], dim=-1)  # (B, N, k, 2*C)

        # 4. Message Passing (MLP)
        # Flatten for MLP: (B*N*k, 2*C)
        edge_feature = edge_feature.view(B * N * self.k, -1)

        # Apply MLP. Note: BatchNorm1d expects (N, C) which matches here.
        out = self.mlp(edge_feature)  # (B*N*k, out_channels)

        # Reshape back
        out = out.view(B, N, self.k, -1)

        # 5. Aggregation (Max Pooling over neighbors)
        out = out.max(dim=2)[0]  # (B, N, out_channels)

        return out


class ADGN_Model(nn.Module):
    """
    Attentive Dynamic Graph Network with Geometric Priors.
    Combines DynEdge layers, Transformer aggregation, and explicit geometric features.
    """

    def __init__(self):
        super().__init__()

        k = Config.K_NEIGHBORS
        dim = Config.EMBED_DIM
        input_dim = 7  # x, y, z, time, charge, aux, mask

        # --- Feature Extractor (DynEdge) ---
        self.conv1 = DynEdgeLayer(input_dim, dim, k=k)
        self.conv2 = DynEdgeLayer(dim, dim, k=k)
        self.conv3 = DynEdgeLayer(dim, dim, k=k)

        # --- Global Aggregator (Transformer) ---
        # We use a Transformer Encoder to aggregate the sequence of pulses
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=dim * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Learnable [CLS] token to capture global context
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # --- Geometric Priors Fusion ---
        priors_dim = 19
        self.priors_bn = nn.BatchNorm1d(priors_dim)

        # --- Prediction Head ---
        # Input: Transformer embedding (dim) + Priors (priors_dim)
        self.head = nn.Sequential(
            nn.Linear(dim + priors_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Output: vector (nx, ny, nz)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        # Initialize Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, priors):
        """
        Args:
            x: Pulse features (B, N, 7)
            priors: Geometric priors (B, 19)
        """
        B, N, _ = x.shape

        # 1. Local Feature Extraction (DynEdge with Residuals)
        h1 = self.conv1(x)
        h2 = self.conv2(h1) + h1
        h3 = self.conv3(h2) + h2  # (B, N, dim)

        # 2. Global Aggregation (Transformer)
        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        h_trans_in = torch.cat((cls_tokens, h3), dim=1)  # (B, N+1, dim)

        # Create Padding Mask
        # x[:, :, 6] is 1 for valid, 0 for padding.
        # Transformer mask expects True for padded positions.
        mask_vals = x[:, :, 6]  # (B, N)
        cls_mask = torch.ones(B, 1, device=x.device)
        full_mask = torch.cat((cls_mask, mask_vals), dim=1)  # (B, N+1)
        key_padding_mask = full_mask < 0.5

        # Pass through Transformer
        h_trans = self.transformer(h_trans_in, src_key_padding_mask=key_padding_mask)

        # Extract [CLS] token output
        cls_out = h_trans[:, 0, :]  # (B, dim)

        # 3. Geometric Fusion
        priors_norm = self.priors_bn(priors)
        combined = torch.cat((cls_out, priors_norm), dim=1)  # (B, dim + priors_dim)

        # 4. Prediction
        out = self.head(combined)

        # Normalize to unit vector
        out = F.normalize(out, p=2, dim=1)

        return out


# ==========================================
# Training & Inference Logic
# ==========================================


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target_spherical):
        """
        pred: (B, 3) Cartesian unit vectors
        target_spherical: (B, 2) Azimuth, Zenith
        """
        # Convert target to Cartesian
        az, zen = target_spherical[:, 0], target_spherical[:, 1]
        tx, ty, tz = spherical_to_cartesian(az, zen)
        target_cart = torch.stack([tx, ty, tz], dim=1)

        # Cosine Similarity = dot product of unit vectors
        # pred is already normalized in model
        # target is normalized by definition
        cos_sim = torch.sum(pred * target_cart, dim=1)

        # Loss = 1 - mean(cos_sim)
        return 1.0 - torch.mean(cos_sim)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, priors, y, _ in tqdm(loader, desc="Training", leave=False):
        X = X.to(device, non_blocking=True)
        priors = priors.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(X, priors)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, priors, y, _ in tqdm(loader, desc="Validating", leave=False):
            X = X.to(device, non_blocking=True)
            priors = priors.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = model(X, priors)
            loss = criterion(preds, y)

            total_loss += loss.item() * X.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    # Compute Competition Metric
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Convert predictions to spherical for metric calculation
    pred_az, pred_zen = cartesian_to_spherical(
        all_preds[:, 0], all_preds[:, 1], all_preds[:, 2]
    )
    y_pred_spherical = np.stack([pred_az, pred_zen], axis=1)

    metric = angular_dist_score(all_targets, y_pred_spherical)

    return avg_loss, metric


def predict_test(model, loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for X, priors, _, event_ids in tqdm(loader, desc="Inference"):
            X = X.to(device, non_blocking=True)
            priors = priors.to(device, non_blocking=True)

            # Forward
            preds_cart = model(X, priors)  # (B, 3)

            # Convert to Spherical
            px = preds_cart[:, 0].cpu().numpy()
            py = preds_cart[:, 1].cpu().numpy()
            pz = preds_cart[:, 2].cpu().numpy()

            az, zen = cartesian_to_spherical(px, py, pz)

            # Store
            batch_res = pd.DataFrame(
                {"event_id": event_ids, "azimuth": az, "zenith": zen}
            )
            results.append(batch_res)

    return pd.concat(results, ignore_index=True)


def run():
    """
    Main execution pipeline.
    """
    print("Initializing ADGN Solution...")
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Setup
    train_loader, val_loader = get_dataloaders()

    # 2. Model Setup
    model = ADGN_Model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = CosineSimilarityLoss()

    # 3. Training Loop
    best_metric = float("inf")
    best_epoch = -1
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric (Mean Angular Error): {val_metric:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"  -> New Best Model Saved!")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}.")
            break

    print(f"Training complete. Best Metric: {best_metric:.6f} at Epoch {best_epoch+1}")

    # 4. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    test_loader = get_test_loader()
    print(f"Predicting on Test Set ({len(test_loader.dataset)} events)...")

    submission_df = predict_test(model, test_loader, device)

    # 5. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    # Ensure columns are in correct order
    submission_df = submission_df[["event_id", "azimuth", "zenith"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")
