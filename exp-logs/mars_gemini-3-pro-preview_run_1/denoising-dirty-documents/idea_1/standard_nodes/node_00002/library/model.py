import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.utils import set_seed, calculate_rmse

# --- Architecture Components ---


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder: 32 -> 64 -> 128
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)  # Bottleneck input

        # Decoder: 256 -> 128 -> 64 -> 32
        factor = 2 if bilinear else 1
        self.up1 = Up(384 if bilinear else 256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# --- Training and Inference Logic ---


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device):
    model.eval()
    total_rmse = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate RMSE for this image
            # Note: inputs/targets are (1, 1, H, W) in validation
            rmse = calculate_rmse(targets, outputs)
            total_rmse += rmse
            count += 1

    return total_rmse / count if count > 0 else 0.0


def generate_submission(model, dataloader, device, output_path):
    print("Generating submission...")
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, img_ids in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Move to CPU and numpy
            # Shape: (1, 1, H, W) -> (H, W)
            pred_img = outputs.squeeze().cpu().numpy()
            img_id = img_ids[0]  # batch size is 1

            h, w = pred_img.shape

            # Generate IDs: {img_id}_{row}_{col} (1-based indexing)
            # Efficiently create grid
            rows, cols = np.indices((h, w))
            rows = rows + 1
            cols = cols + 1

            # Flatten arrays
            flat_vals = pred_img.flatten()
            flat_rows = rows.flatten()
            flat_cols = cols.flatten()

            # Create ID strings
            # Using list comprehension as it's straightforward
            flat_ids = [f"{img_id}_{r}_{c}" for r, c in zip(flat_rows, flat_cols)]

            # Append to results
            df_img = pd.DataFrame({"id": flat_ids, "value": flat_vals})
            results.append(df_img)

    # Concatenate all results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print("No test data found.")


def run(epochs=None, batch_size=None):
    """
    Main execution function to train the model and generate submission.
    """
    # Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Overrides
    if epochs is None:
        epochs = Config.NUM_EPOCHS
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model Initialization
    model = UNet(n_channels=Config.NUM_CHANNELS, n_classes=1, bilinear=True)
    model.to(device)

    # Optimization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training Loop
    best_val_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss (MSE): {train_loss:.6f} - Val RMSE: {val_rmse:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            # print(f"  New best model saved! RMSE: {best_val_rmse:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for inference
    print(f"Loading best model (RMSE: {best_val_rmse:.6f}) for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
