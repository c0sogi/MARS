import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, dataset

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class ResidualBlock(nn.Module):
    """
    A Residual Block utilizing Dilated Convolutions and Batch Normalization.
    Maintains spatial resolution via padding = dilation.
    """

    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class HDNet(nn.Module):
    """
    Hyper-Dense Dilated Network (HD-Net).

    Architecture:
    1. Depth-Projection Stem: 1x1 Conv + InstanceNorm.
    2. Sequential Dilated Backbone: 4 Residual Blocks with increasing dilation (1, 2, 4, 8).
    3. Hyper-Dense Aggregation Head: Concatenates outputs of all blocks + 1x1 Conv Classifier.
    """

    def __init__(self):
        super(HDNet, self).__init__()

        # Dimensions from config
        in_channels = config.Z_DIM  # 65
        base_channels = config.MODEL_BASE_CHANNELS  # 32

        # 1. Depth-Projection Stem
        # Compresses 3D volume (treated as channels) to 2D feature map
        # Followed by Instance Normalization to handle global intensity shifts
        self.stem_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.stem_norm = nn.InstanceNorm2d(base_channels, affine=True)
        self.stem_act = nn.ReLU(inplace=True)

        # 2. Sequential Dilated Backbone
        # Block 1: Texture (Dilation 1)
        self.block1 = ResidualBlock(base_channels, dilation=1)

        # Block 2: Short-Range (Dilation 2)
        self.block2 = ResidualBlock(base_channels, dilation=2)

        # Block 3: Mid-Range (Dilation 4)
        self.block3 = ResidualBlock(base_channels, dilation=4)

        # Block 4: Context (Dilation 8)
        self.block4 = ResidualBlock(base_channels, dilation=8)

        # 3. Hyper-Dense Aggregation Head
        # Concatenates outputs from all 4 blocks (Dense Aggregation)
        # 4 blocks * base_channels
        agg_channels = base_channels * 4

        self.classifier = nn.Conv2d(agg_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Input: (Batch, 65, H, W)

        # Stem
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = self.stem_act(x)

        # Backbone with Dense Collection
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3)

        # Hyper-Dense Aggregation
        # Concatenate along channel dimension (dim=1)
        concat_features = torch.cat([feat1, feat2, feat3, feat4], dim=1)

        # Classifier
        logits = self.classifier(concat_features)

        return logits


# =============================================================================
# TRAINING LOGIC
# =============================================================================


def train_model(num_epochs=config.NUM_EPOCHS, patience=3):
    """
    Trains the HDNet model.
    """
    utils.set_seed(config.SEED)

    # DataLoaders
    train_loader, val_loader, _ = dataset.get_dataloaders()

    # Model Setup
    model = HDNet().to(config.DEVICE)

    # Loss & Optimizer
    # Weighted BCE to handle class imbalance
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on {config.DEVICE}...")

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for volumes, labels in train_loader:
            volumes = volumes.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * volumes.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes = volumes.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                outputs = model(volumes)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * volumes.size(0)

                # Store for metric calculation (apply sigmoid for probs)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Calculate F0.5 Score (using default 0.5 threshold for monitoring)
        y_true = np.concatenate(all_labels)
        y_pred_probs = np.concatenate(all_preds)
        y_pred_bin = (y_pred_probs >= 0.5).astype(np.uint8)
        val_f05 = utils.calculate_fbeta(y_true, y_pred_bin, beta=0.5)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F0.5: {val_f05:.10f}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print("  Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training complete.")


# =============================================================================
# INFERENCE & SUBMISSION LOGIC
# =============================================================================


def predict_and_submit():
    """
    1. Loads best model.
    2. Optimizes threshold on Validation set.
    3. Predicts on Test set.
    4. Stitches patches and generates RLE submission.
    """
    utils.set_seed(config.SEED)

    # Load Model
    model = HDNet().to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("No checkpoint found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    _, val_loader, test_loader = dataset.get_dataloaders()

    # --- Step 1: Threshold Optimization (Validation) ---
    print("Optimizing threshold on validation set...")
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(config.DEVICE)
            outputs = model(volumes)
            probs = torch.sigmoid(outputs)

            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    y_true_val = np.concatenate(all_val_labels)
    y_probs_val = np.concatenate(all_val_probs)

    best_threshold = utils.optimize_threshold(y_true_val, y_probs_val, beta=0.5)

    # --- Step 2: Test Prediction & Stitching ---
    print(
        f"Generating predictions for test set using threshold {best_threshold:.4f}..."
    )

    # Load test metadata to determine fragment dimensions
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Initialize canvases for each fragment
    # We need to find the max width and height for each fragment
    fragment_dims = {}
    for fid in df_test["fragment_id"].unique():
        frag_rows = df_test[df_test["fragment_id"] == fid]
        max_w = (frag_rows["x"] + frag_rows["w"]).max()
        max_h = (frag_rows["y"] + frag_rows["h"]).max()
        fragment_dims[fid] = (max_h, max_w)

    fragment_masks = {
        fid: np.zeros(dims, dtype=np.uint8) for fid, dims in fragment_dims.items()
    }

    with torch.no_grad():
        for volumes, sample_ids in test_loader:
            volumes = volumes.to(config.DEVICE)
            outputs = model(volumes)
            probs = torch.sigmoid(outputs)

            # Apply threshold immediately to save memory
            preds = (probs >= best_threshold).float().cpu().numpy()

            # Place patches onto canvases
            for i, sample_id in enumerate(sample_ids):
                # sample_id format: {fragment_id}_{y}_{x}
                parts = sample_id.split("_")
                # Handle fragment IDs that might contain underscores (though data description says '1','2','a' etc)
                # Assuming standard format from metadata script: last two are y and x
                x = int(parts[-1])
                y = int(parts[-2])
                fid = "_".join(parts[:-2])

                patch_pred = preds[i, 0, :, :]  # (H, W)

                # Determine valid width/height (handle edge cases where patch might be padded)
                # The model output is always 512x512.
                # We need to crop it to the actual w, h from metadata if it was padded?
                # The dataset pads input. We should crop output.
                # Let's look up the w, h for this sample
                row = df_test[df_test["sample_id"] == sample_id].iloc[0]
                w, h = row["w"], row["h"]

                # Crop the valid region from the top-left of the prediction
                valid_pred = patch_pred[:h, :w]

                # Place on canvas
                fragment_masks[fid][y : y + h, x : x + w] = valid_pred.astype(np.uint8)

    # --- Step 3: RLE Encoding & Submission ---
    predictions = {}
    for fid, mask in fragment_masks.items():
        rle = utils.rle_encode(mask)
        predictions[fid] = rle

    utils.write_submission(predictions)
