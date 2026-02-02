import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
    get_cosine_schedule_with_warmup,
)

# =============================================================================
# Loss Function
# =============================================================================


class CosineSimilarityLoss(nn.Module):
    """
    Computes 1 - CosineSimilarity(pred, target).
    Target is expected to be a normalized 3D vector.
    """

    def __init__(self):
        super().__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, pred, target):
        """
        Args:
            pred: (Batch, 3) unnormalized vector.
            target: (Batch, 3) normalized vector.
        """
        # We don't strictly need to normalize pred for CosineSimilarity as it does it internally,
        # but explicit normalization can sometimes help stability.
        # The nn.CosineSimilarity function computes: (a . b) / (|a| * |b|)
        loss = 1.0 - self.cosine_sim(pred, target)
        return loss.mean()


# =============================================================================
# Neural Network Components
# =============================================================================


class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TemporalResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial projection
        # Input: (Batch, 6, 192)
        base_filters = Config.RESNET_BASE_FILTERS  # 64

        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                Config.N_CHANNELS,
                base_filters,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        # ResNet Blocks
        # Sequence length logic: 192 -> (s2) -> 96 -> (s2) -> 48 -> (s2) -> 24

        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2, stride=1)
        self.layer2 = self._make_layer(
            base_filters, base_filters * 2, blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            base_filters * 2, base_filters * 4, blocks=2, stride=2
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = base_filters * 4  # 256

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, 6, 192)
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        return x.flatten(1)  # (Batch, 256)


class GeometricMLP(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = Config.NUM_GEOM_FEATURES
        hidden_dim = Config.MLP_HIDDEN_DIM

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class DualStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = TemporalResNet1D()
        self.geometric = GeometricMLP()

        fusion_input_dim = self.temporal.out_dim + self.geometric.out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 3),  # Output (x, y, z)
        )

    def forward(self, seq_x, geom_x):
        t_out = self.temporal(seq_x)
        g_out = self.geometric(geom_x)

        fused = torch.cat([t_out, g_out], dim=1)
        out = self.head(fused)
        return out


# =============================================================================
# Training & Inference Logic
# =============================================================================


def train_model(config, train_loader, val_loader):
    """
    Executes the training loop with Early Stopping and Cosine Annealing.
    """
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = DualStreamNetwork().to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = CosineSimilarityLoss()

    # Scheduler
    num_training_steps = len(train_loader) * config.NUM_EPOCHS
    num_warmup_steps = len(train_loader) * config.WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Tracking
    best_val_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(config.NUM_EPOCHS):
        # --- Training ---
        model.train()
        train_losses = []

        start_time = time.time()

        for batch_idx, (seq_x, geom_x, target_angles, _) in enumerate(train_loader):
            seq_x = seq_x.to(device)
            geom_x = geom_x.to(device)

            # Convert target angles (az, zen) to vectors (x, y, z) for loss
            azimuth = target_angles[:, 0].to(device)
            zenith = target_angles[:, 1].to(device)
            target_vecs_x, target_vecs_y, target_vecs_z = spherical_to_cartesian(
                azimuth, zenith
            )
            target_vecs = torch.stack(
                [target_vecs_x, target_vecs_y, target_vecs_z], dim=1
            )

            optimizer.zero_grad()
            preds = model(seq_x, geom_x)

            loss = criterion(preds, target_vecs)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_preds_az = []
        val_preds_zen = []
        val_true_az = []
        val_true_zen = []

        with torch.no_grad():
            for seq_x, geom_x, target_angles, _ in val_loader:
                seq_x = seq_x.to(device)
                geom_x = geom_x.to(device)

                # Forward
                preds_vec = model(seq_x, geom_x)

                # Convert preds to spherical
                p_x = preds_vec[:, 0]
                p_y = preds_vec[:, 1]
                p_z = preds_vec[:, 2]
                pred_az, pred_zen = cartesian_to_spherical(p_x, p_y, p_z)

                # Store (move to CPU numpy)
                val_preds_az.append(pred_az.cpu().numpy())
                val_preds_zen.append(pred_zen.cpu().numpy())
                val_true_az.append(target_angles[:, 0].numpy())
                val_true_zen.append(target_angles[:, 1].numpy())

        # Concatenate
        val_preds_az = np.concatenate(val_preds_az)
        val_preds_zen = np.concatenate(val_preds_zen)
        val_true_az = np.concatenate(val_true_az)
        val_true_zen = np.concatenate(val_true_zen)

        # Compute Metric
        val_score = angular_dist_score(
            val_true_az, val_true_zen, val_preds_az, val_preds_zen
        )

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MAE: {val_score:.8f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! Score: {best_val_score:.8f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_val_score:.8f}")
    return best_model_path


def predict_submission(config, test_loader, model_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device(config.DEVICE)
    print(f"Loading model from {model_path} for inference...")

    model = DualStreamNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_event_ids = []
    all_azimuth = []
    all_zenith = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, (seq_x, geom_x, _, event_ids) in enumerate(test_loader):
            seq_x = seq_x.to(device)
            geom_x = geom_x.to(device)

            # Forward
            preds_vec = model(seq_x, geom_x)

            # Convert to spherical
            p_x = preds_vec[:, 0]
            p_y = preds_vec[:, 1]
            p_z = preds_vec[:, 2]
            pred_az, pred_zen = cartesian_to_spherical(p_x, p_y, p_z)

            # Store
            all_event_ids.append(event_ids.numpy())
            all_azimuth.append(pred_az.cpu().numpy())
            all_zenith.append(pred_zen.cpu().numpy())

            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} batches...")

    # Concatenate
    all_event_ids = np.concatenate(all_event_ids)
    all_azimuth = np.concatenate(all_azimuth)
    all_zenith = np.concatenate(all_zenith)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"event_id": all_event_ids, "azimuth": all_azimuth, "zenith": all_zenith}
    )

    # Save
    out_path = config.SUBMISSION_PATH
    print(f"Saving submission to {out_path}...")
    submission_df.to_csv(out_path, index=False)
    print("Submission saved successfully.")
