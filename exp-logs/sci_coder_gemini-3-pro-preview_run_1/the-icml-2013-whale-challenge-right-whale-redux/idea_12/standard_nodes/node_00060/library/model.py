import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import pandas as pd
import os
import time

from library.config import Config
from library.utils import set_seed, compute_auc
from library.dataset import get_dataloaders

# ==========================================
# Model Components
# ==========================================


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.
    Captures long-range dependencies with precise positional information.
    """

    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concat
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Attn maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer for temporal aggregation.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x: (Batch, Time, Features)
        weights = self.attention(x)
        out = torch.sum(x * weights, dim=1)
        return out


class MultiBandResNetCRNN(nn.Module):
    """
    Multi-Band Hierarchical Coordinate-Attention ResNet-18 CRNN.
    """

    def __init__(self):
        super(MultiBandResNetCRNN, self).__init__()

        # 1. Backbone: ResNet18
        # Load weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)

        # Modify first layer for 1 channel input
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        # Initialize with average of RGB weights
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        # 2. Time-Preserving Strides
        # Layer 2 keeps stride 2 (standard)
        # Layer 3: Stride (2, 1)
        self.resnet.layer3[0].conv1.stride = (2, 1)
        self.resnet.layer3[0].downsample[0].stride = (2, 1)
        # Layer 4: Stride (2, 1)
        self.resnet.layer4[0].conv1.stride = (2, 1)
        self.resnet.layer4[0].downsample[0].stride = (2, 1)

        # 3. Coordinate Attention
        # We wrap the layers with CA if configured
        if Config.USE_COORDINATE_ATTENTION:
            self.ca2 = CoordinateAttention(128, 128)
            self.ca3 = CoordinateAttention(256, 256)
            self.ca4 = CoordinateAttention(512, 512)

        # 4. RNN Head
        # Feature dimensions calculation
        # Layer 2: 128 ch, 2 bands -> 256
        # Layer 3: 256 ch, 2 bands -> 512
        # Layer 4: 512 ch, 1 band  -> 512
        # Total: 1280
        self.rnn_input_dim = 128 * 2 + 256 * 2 + 512
        self.rnn_hidden_dim = 128

        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.attn_pool = AttentionPooling(self.rnn_hidden_dim * 2)
        self.fc = nn.Linear(self.rnn_hidden_dim * 2, 1)

    def _multiband_pool(self, x, num_bands):
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        assert F % num_bands == 0, f"Freq dim {F} not divisible by {num_bands}"

        # Reshape to separate bands: (B, C, Bands, F_per_band, T)
        x_reshaped = x.view(B, C, num_bands, F // num_bands, T)

        # Average over frequency within each band: (B, C, Bands, T)
        x_pooled = x_reshaped.mean(dim=3)

        # Flatten bands into channels: (B, C * Bands, T)
        x_out = x_pooled.view(B, C * num_bands, T)
        return x_out

    def forward(self, x):
        # x: (B, 1, F, T)

        # Stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Layer 1
        x = self.resnet.layer1(x)

        # Layer 2
        l2 = self.resnet.layer2(x)
        if Config.USE_COORDINATE_ATTENTION:
            l2 = self.ca2(l2)
        # Pool Layer 2: 2 Bands
        f2 = self._multiband_pool(l2, num_bands=2)

        # Layer 3
        l3 = self.resnet.layer3(l2)
        if Config.USE_COORDINATE_ATTENTION:
            l3 = self.ca3(l3)
        # Pool Layer 3: 2 Bands
        f3 = self._multiband_pool(l3, num_bands=2)

        # Layer 4
        l4 = self.resnet.layer4(l3)
        if Config.USE_COORDINATE_ATTENTION:
            l4 = self.ca4(l4)
        # Pool Layer 4: 1 Band (Global)
        f4 = self._multiband_pool(l4, num_bands=1)

        # Concatenate: (B, Total_Channels, T)
        features = torch.cat([f2, f3, f4], dim=1)

        # Prepare for RNN: (B, T, C)
        features = features.permute(0, 2, 1)

        # RNN
        self.gru.flatten_parameters()
        rnn_out, _ = self.gru(features)

        # Pooling
        pool_out = self.attn_pool(rnn_out)

        # Classification
        logits = self.fc(pool_out)

        return logits


# ==========================================
# Training & Execution Logic
# ==========================================


def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # Apply Mixup
        data, target_a, target_b, lam = mixup_data(data, target, Config.MIXUP_ALPHA)

        optimizer.zero_grad()
        output = model(data).squeeze(1)

        loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data).squeeze(1)
            loss = criterion(output, target)

            total_loss += loss.item()

            probs = torch.sigmoid(output).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(target.cpu().numpy())

    auc = compute_auc(all_targets, all_preds)
    return total_loss / len(loader), auc


def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for data, clip_ids in loader:
            data = data.to(device)
            output = model(data).squeeze(1)
            probs = torch.sigmoid(output).cpu().numpy()

            all_preds.extend(probs)
            all_ids.extend(clip_ids)

    return all_ids, all_preds


def run():
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Model
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = MultiBandResNetCRNN().to(device)

    # 3. Optimization
    # Use BCEWithLogitsLoss with pos_weight for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val AUC: {val_auc:.8f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best AUC! Model saved.")

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions on Test set...")
    test_ids, test_probs = predict(model, test_loader, device)

    # 6. Submission
    submission_df = pd.DataFrame({"clip": test_ids, "probability": test_probs})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Final Best Val AUC: {best_auc:.8f}")
