import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from library.config import (
    Z_DIM,
    DEVICE,
    LEARNING_RATE,
    POS_WEIGHT,
    NUM_EPOCHS,
    SUBMISSION_FILE,
    CACHE_DIR,
    TEST_METADATA,
    PATCH_SIZE,
)
from library.utils import fbeta_score, rle_encoding, set_seed


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class DepthProjectionCNN(nn.Module):
    """
    A 2.5D ResNet-style network for 3D to 2D ink detection.

    Architecture:
    1. Depth Projection: 1x1 Conv to compress Z-slices.
    2. Normalization: InstanceNorm to handle intensity shifts.
    3. Spatial Encoder: Stack of Dilated Residual Blocks for context.
    4. Classification: 1x1 Conv.
    """

    def __init__(self, in_channels=Z_DIM):
        super(DepthProjectionCNN, self).__init__()

        # Stage 1: Learnable Surface Projection
        # Increased channels to 64 for better feature representation
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            # Use InstanceNorm to address sensitivity to input mean intensity
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        # Stage 2: Spatial Encoder with Dilated Residual Blocks
        # Dilation increases receptive field without downsampling
        # Using exponentially increasing dilation for structural continuity (Cite solution_lesson_node_00003)
        self.encoder = nn.Sequential(
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=4),
            ResidualBlock(64, dilation=8),
        )

        # Stage 3: Classification Head
        self.classifier = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: (Batch, Z_DIM, H, W)
        x = self.projection(x)
        x = self.encoder(x)
        logits = self.classifier(x)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            # Store probabilities and targets for metric calculation
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate for global metric calculation
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate F0.5 score with default threshold 0.5
    score = fbeta_score(all_preds, all_targets, beta=0.5, threshold=0.5)

    return running_loss / len(loader.dataset), score


def train_model(dataloaders, epochs=NUM_EPOCHS, patience=3):
    """
    Trains the DepthProjectionCNN model with Early Stopping.

    Args:
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        epochs (int): Maximum number of epochs.
        patience (int): Epochs to wait for improvement before stopping.

    Returns:
        model: The trained model with the best validation weights loaded.
    """
    set_seed(42)
    device = torch.device(DEVICE)

    model = DepthProjectionCNN().to(device)

    # Weighted BCE Loss to handle class imbalance
    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device
        )
        val_loss, val_score = validate(model, dataloaders["val"], criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F0.5: {val_score:.6f}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model locally for reference
    torch.save(model.state_dict(), os.path.join(CACHE_DIR, "best_model.pth"))

    return model


def optimize_threshold(model, loader):
    """
    Finds the probability threshold that maximizes the F0.5 score on the validation set.
    """
    device = torch.device(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    best_threshold = 0.5
    best_score = 0.0

    # Search range [0.1, 0.9]
    thresholds = np.arange(0.1, 0.95, 0.05)

    for thr in thresholds:
        score = fbeta_score(all_preds, all_targets, beta=0.5, threshold=thr)
        if score > best_score:
            best_score = score
            best_threshold = thr

    print(f"Optimized Threshold: {best_threshold:.2f} (Val F0.5: {best_score:.6f})")
    return best_threshold


def predict_and_submit(model, test_loader, threshold=0.5):
    """
    Generates predictions for the test set, reconstructs full fragment masks,
    performs RLE, and saves submission.csv.
    """
    device = torch.device(DEVICE)
    model.eval()

    # 1. Determine canvas sizes for each test fragment
    # We read the test metadata to find the extent of each fragment
    if not os.path.exists(TEST_METADATA):
        print("Test metadata not found. Skipping submission generation.")
        return

    df_test = pd.read_csv(TEST_METADATA)
    fragment_ids = df_test["fragment_id"].unique()

    # Dictionary to store reconstructed probability maps
    # Key: fragment_id, Value: numpy array
    fragment_maps = {}
    fragment_counts = (
        {}
    )  # To handle overlaps if any (though grid is usually non-overlapping)

    for fid in fragment_ids:
        fid_df = df_test[df_test["fragment_id"] == fid]
        max_h = (fid_df["y"] + fid_df["h"]).max()
        max_w = (fid_df["x"] + fid_df["w"]).max()

        fragment_maps[fid] = np.zeros((max_h, max_w), dtype=np.float32)

    print("Generating predictions...")

    with torch.no_grad():
        for inputs, sample_ids in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.sigmoid(logits).cpu().numpy()

            # Iterate through batch
            for i, sample_id in enumerate(sample_ids):
                # Parse sample_id: {fragment_id}_{y}_{x}
                parts = sample_id.split("_")
                # Handle cases where fragment_id might contain underscores (unlikely based on data but safe)
                # The schema is fragment_id is the first part, y is second to last, x is last
                x = int(parts[-1])
                y = int(parts[-2])
                fid = "_".join(parts[:-2])

                # Get prediction patch
                pred_patch = preds[i, 0, :, :]  # (H, W)

                # Determine valid crop size (handle edges)
                h, w = pred_patch.shape

                # Place in canvas
                # Note: If stride < patch_size, we would need averaging.
                # Assuming standard grid generation from metadata script which handles boundaries.
                # We simply overwrite or add. Since metadata generates patches, we overwrite.

                # Ensure we don't go out of bounds (though metadata should prevent this)
                cur_h, cur_w = fragment_maps[fid].shape
                h_end = min(y + h, cur_h)
                w_end = min(x + w, cur_w)

                h_len = h_end - y
                w_len = w_end - x

                fragment_maps[fid][y : y + h_len, x : x + w_len] = pred_patch[
                    :h_len, :w_len
                ]

    # 2. Threshold, RLE, and Write Submission
    print(f"Writing submission to {SUBMISSION_FILE}...")

    submission_data = []

    for fid in sorted(fragment_ids):
        prob_map = fragment_maps[fid]
        binary_mask = (prob_map > threshold).astype(np.uint8)

        rle = rle_encoding(binary_mask)
        submission_data.append({"Id": fid, "Predicted": rle})

    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print("Submission saved successfully.")
