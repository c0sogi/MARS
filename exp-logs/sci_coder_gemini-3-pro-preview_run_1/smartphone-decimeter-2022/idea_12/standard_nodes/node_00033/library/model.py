import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.utils import set_seed, Meters_to_WGS84
from library.data_preprocessing import prepare_training_data, prepare_test_data
from library.dataset import GNSSSequenceDataset, gnss_collate_fn

# ==========================================
# Model Architecture
# ==========================================


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
        self.branches = nn.ModuleList()

        # 1x1 Conv branch
        self.branches.append(
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Conv branches
        for d in dilations:
            padding = (3 - 1) // 2 * d
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        3,
                        padding=padding,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Global Pooling branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]

        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2], mode="nearest")
        res.append(global_feat)

        return self.project(torch.cat(res, dim=1))


class AtrousResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_dim=64):
        super().__init__()

        # --- Encoder ---
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ResidualBlock1D(base_dim, base_dim)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResidualBlock1D(base_dim, base_dim * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ResidualBlock1D(base_dim * 2, base_dim * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = ResidualBlock1D(base_dim * 4, base_dim * 8)
        self.pool4 = nn.MaxPool1d(2)

        # --- Bottleneck ---
        self.aspp = ASPP(base_dim * 8, base_dim * 8, Config.ASPP_DILATION_RATES)

        # --- Decoder ---
        # Dec4: Input (512 from ASPP + 512 from Enc4) -> Output 256
        self.dec4_up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec4_conv = nn.Conv1d(
            base_dim * 16, base_dim * 4, 1
        )  # Reduce channels after concat
        self.dec4_block = ResidualBlock1D(base_dim * 4, base_dim * 4)

        # Dec3: Input (256 + 256 from Enc3) -> Output 128
        self.dec3_up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec3_conv = nn.Conv1d(base_dim * 8, base_dim * 2, 1)
        self.dec3_block = ResidualBlock1D(base_dim * 2, base_dim * 2)

        # Dec2: Input (128 + 128 from Enc2) -> Output 64
        self.dec2_up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec2_conv = nn.Conv1d(base_dim * 4, base_dim, 1)
        self.dec2_block = ResidualBlock1D(base_dim, base_dim)

        # Dec1: Input (64 + 64 from Enc1) -> Output 64
        self.dec1_up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec1_conv = nn.Conv1d(base_dim * 2, base_dim, 1)
        self.dec1_block = ResidualBlock1D(base_dim, base_dim)

        # --- Heads ---
        self.head_final = nn.Conv1d(base_dim, out_channels, 1)
        self.head_aux1 = nn.Conv1d(
            base_dim * 2, out_channels, 1
        )  # Attached to Dec3 output
        self.head_aux2 = nn.Conv1d(
            base_dim * 4, out_channels, 1
        )  # Attached to Dec4 output

    def forward(self, x):
        # Encoder
        x0 = self.init_conv(x)

        x1 = self.enc1(x0)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        # Bottleneck
        b = self.aspp(p4)

        # Decoder
        # Stage 4
        d4 = self.dec4_up(b)
        # Handle potential shape mismatch due to odd input lengths
        if d4.shape[2] != x4.shape[2]:
            d4 = F.interpolate(d4, size=x4.shape[2], mode="linear", align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4_conv(d4)
        d4 = self.dec4_block(d4)
        out_aux2 = self.head_aux2(d4)

        # Stage 3
        d3 = self.dec3_up(d4)
        if d3.shape[2] != x3.shape[2]:
            d3 = F.interpolate(d3, size=x3.shape[2], mode="linear", align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3_conv(d3)
        d3 = self.dec3_block(d3)
        out_aux1 = self.head_aux1(d3)

        # Stage 2
        d2 = self.dec2_up(d3)
        if d2.shape[2] != x2.shape[2]:
            d2 = F.interpolate(d2, size=x2.shape[2], mode="linear", align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2_conv(d2)
        d2 = self.dec2_block(d2)

        # Stage 1
        d1 = self.dec1_up(d2)
        if d1.shape[2] != x1.shape[2]:
            d1 = F.interpolate(d1, size=x1.shape[2], mode="linear", align_corners=True)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1_conv(d1)
        d1 = self.dec1_block(d1)

        out_final = self.head_final(d1)

        return [out_final, out_aux1, out_aux2]


# ==========================================
# Training & Inference Pipeline
# ==========================================


def masked_mae_loss(pred, target, mask):
    """
    Calculates Mean Absolute Error ignoring padded values.
    """
    # pred: (B, C, L), target: (B, C, L), mask: (B, L)
    # Expand mask to match channels
    mask_expanded = mask.unsqueeze(1).expand_as(pred)

    loss = F.l1_loss(pred, target, reduction="none")
    loss = loss * mask_expanded

    # Avoid division by zero
    sum_mask = mask_expanded.sum()
    if sum_mask > 0:
        return loss.sum() / sum_mask
    else:
        return loss.sum() * 0.0


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()

        outputs = model(features)
        final_out, aux1, aux2 = outputs

        # Interpolate aux outputs to match target length (if necessary)
        if aux1.shape[2] != targets.shape[2]:
            aux1 = F.interpolate(
                aux1, size=targets.shape[2], mode="linear", align_corners=True
            )
        if aux2.shape[2] != targets.shape[2]:
            aux2 = F.interpolate(
                aux2, size=targets.shape[2], mode="linear", align_corners=True
            )

        # Calculate Deep Supervision Loss
        loss_final = masked_mae_loss(final_out, targets, masks)
        loss_aux1 = masked_mae_loss(aux1, targets, masks)
        loss_aux2 = masked_mae_loss(aux2, targets, masks)

        w = Config.LOSS_WEIGHTS
        total_loss = w[0] * loss_final + w[1] * loss_aux1 + w[2] * loss_aux2

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    running_loss = 0.0

    # Metrics for final output only
    mae_north = 0.0
    mae_east = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)

            outputs = model(features)
            final_out = outputs[0]

            # Loss calculation (using only final output for validation metric)
            loss = masked_mae_loss(final_out, targets, masks)
            running_loss += loss.item()

            # Detailed metrics
            # Unpad for metrics
            for i in range(features.shape[0]):
                valid_len = int(masks[i].sum().item())
                pred_seq = final_out[i, :, :valid_len].cpu().numpy()
                target_seq = targets[i, :, :valid_len].cpu().numpy()

                # Rows: North, East
                diff = np.abs(pred_seq - target_seq)
                mae_north += diff[0].sum()
                mae_east += diff[1].sum()
                total_samples += valid_len

    avg_loss = running_loss / len(loader)
    avg_mae_north = mae_north / total_samples
    avg_mae_east = mae_east / total_samples

    # Combined metric (mean of 50th/95th is competition metric, but here we use MAE for simplicity)
    # Just return avg distance error
    return avg_loss, avg_mae_north, avg_mae_east


def run_pipeline():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_df, val_df = prepare_training_data(load_cached_data=True)

    train_dataset = GNSSSequenceDataset(train_df, mode="train")
    val_dataset = GNSSSequenceDataset(val_df, mode="train", scaler=train_dataset.scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=gnss_collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=2,
    )

    # 2. Initialize Model
    model = AtrousResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_dim=Config.HIDDEN_DIM,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_n, val_e = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE N: {val_n:.4f}m E: {val_e:.4f}m"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # 4. Inference
    print("\nStarting Inference...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_df = prepare_test_data(load_cached_data=True)
    test_dataset = GNSSSequenceDataset(
        test_df, mode="test", scaler=train_dataset.scaler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one trip at a time for simplicity in reconstruction
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=2,
    )

    results = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            masks = batch["masks"]

            # Predict
            outputs = model(features)
            final_out = outputs[0].cpu().numpy()  # (1, 2, L)

            # Reconstruct
            trip_id = batch["trip_ids"][0]
            wls_pos = batch["wls_pos"][0]  # (L, 2) [lat, lon]
            timestamps = batch["timestamps"][0]

            valid_len = int(masks[0].sum().item())

            # Slice valid predictions: (2, Valid_L) -> (Valid_L, 2)
            preds = final_out[0, :, :valid_len].T

            delta_north = preds[:, 0]
            delta_east = preds[:, 1]

            base_lat = wls_pos[:valid_len, 0]
            base_lon = wls_pos[:valid_len, 1]
            valid_timestamps = timestamps[:valid_len]

            # Convert meters back to WGS84
            pred_lat, pred_lon = Meters_to_WGS84(
                base_lat, base_lon, delta_north, delta_east
            )

            # Store results
            trip_df = pd.DataFrame(
                {
                    "tripId": [trip_id] * valid_len,
                    "UnixTimeMillis": valid_timestamps,
                    "LatitudeDegrees": pred_lat,
                    "LongitudeDegrees": pred_lon,
                }
            )
            results.append(trip_df)

    # 5. Save Submission
    submission_df = pd.concat(results, ignore_index=True)

    # Ensure submission format matches sample_submission.csv
    # The sample submission might have gaps or specific rows.
    # We predicted for all timestamps available in test_metadata (which came from sample_submission).
    # Just to be safe, we merge with the template.
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Create a key for merging
    submission_df["key"] = (
        submission_df["tripId"] + "_" + submission_df["UnixTimeMillis"].astype(str)
    )
    sample_sub["key"] = (
        sample_sub["tripId"] + "_" + sample_sub["UnixTimeMillis"].astype(str)
    )

    # Merge predictions into sample
    final_sub = sample_sub.drop(columns=["LatitudeDegrees", "LongitudeDegrees"]).merge(
        submission_df[["key", "LatitudeDegrees", "LongitudeDegrees"]],
        on="key",
        how="left",
    )

    # Fill missing (if any) with 0 or interpolation?
    # If our test_metadata logic was correct, there should be no missing predictions for valid inputs.
    # However, if GNSS data was missing for a timestamp in sample_submission, we might have NaNs.
    # We'll forward fill then backward fill as a fallback.
    final_sub["LatitudeDegrees"] = (
        final_sub["LatitudeDegrees"].fillna(method="ffill").fillna(method="bfill")
    )
    final_sub["LongitudeDegrees"] = (
        final_sub["LongitudeDegrees"].fillna(method="ffill").fillna(method="bfill")
    )

    final_sub = final_sub.drop(columns=["key"])

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    final_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    run_pipeline()
