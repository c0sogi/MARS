import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from library.dataset import get_dataloaders, get_test_ids
from library.utils import seed_everything, AverageMeter, calculate_roc_auc

# --- Configuration ---
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_SEEDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Components ---


class BlurPool(nn.Module):
    """
    Anti-Aliased Downsampling using a low-pass filter (Blur) before subsampling.
    """

    def __init__(self, channels, stride=2):
        super(BlurPool, self).__init__()
        self.stride = stride
        # Gaussian kernel [1, 2, 1]
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]  # 3x3
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        # Register as buffer so it's saved with model but not updated by optimizer
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1, 1))
        self.groups = channels

    def forward(self, x):
        if self.stride == 1:
            return x
        # Pad to keep size correct before subsampling
        return F.conv2d(
            x, self.kernel, stride=self.stride, padding=1, groups=self.groups
        )


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for efficient mobile network design.
    Factorizes attention into two 1D feature encoding processes.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class Res2NeXtBlock(nn.Module):
    """
    Integrates Res2Net hierarchical connections within a ResNeXt topology.
    Includes Coordinate Attention and Anti-Aliased Downsampling (BlurPool).
    """

    def __init__(self, in_planes, planes, stride=1, scale=4, groups=8):
        super(Res2NeXtBlock, self).__init__()
        self.stride = stride
        self.scale = scale
        self.width = planes  # Using expansion=1 for simplicity given "Wide" channels

        # 1x1 Expand/Project
        self.conv1 = nn.Conv2d(in_planes, self.width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width)

        # Res2Net Convs
        # Split width into 'scale' groups
        self.width_per_scale = self.width // scale
        self.convs = nn.ModuleList()
        for i in range(scale - 1):
            # Grouped convolution inside the split
            self.convs.append(
                nn.Conv2d(
                    self.width_per_scale,
                    self.width_per_scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    bias=False,
                )
            )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(self.width_per_scale) for _ in range(scale - 1)]
        )

        # 1x1 Project
        self.conv3 = nn.Conv2d(self.width, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Coordinate Attention
        self.ca = CoordinateAttention(planes)

        # BlurPool for downsampling (applied after convs)
        self.downsample = None
        if stride > 1 or in_planes != planes:
            layers = []
            if stride > 1:
                layers.append(BlurPool(in_planes, stride=stride))
            layers.append(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            self.downsample = nn.Sequential(*layers)

        self.blur = BlurPool(planes, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Res2Net Logic
        spx = torch.split(out, self.width_per_scale, 1)

        # The first group is identity (or processed if we wanted, but standard Res2Net leaves one identity-like path)
        # We accumulate features hierarchically
        yx = []
        yx.append(spx[0])
        y_prev = spx[0]

        for i in range(1, self.scale):
            # y[i] = Conv(x[i] + y[i-1])
            y = self.convs[i - 1](spx[i] + y_prev)
            y = self.bns[i - 1](y)
            y = self.relu(y)
            yx.append(y)
            y_prev = y

        out = torch.cat(yx, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out)

        # Apply BlurPool if stride > 1
        if self.stride > 1:
            out = self.blur(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class WideAntiAliasedRes2NeXt(nn.Module):
    """
    Custom Wide Anti-Aliased Coordinate-Res2NeXt with Multi-Scale Aggregation.
    """

    def __init__(self):
        super(WideAntiAliasedRes2NeXt, self).__init__()

        self.in_planes = 64

        # Stem: 32x32 image, keep resolution high
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        # Stage 1: 64 channels, stride 1 (32x32)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        # Stage 2: 128 channels, stride 2 (16x16)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        # Stage 3: 256 channels, stride 2 (8x8)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)

        # Head: Multi-Scale Aggregation
        # Concatenating GAP from Stage 2 (128) and Stage 3 (256) -> 384
        self.fc = nn.Linear(128 + 256, 1)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        # First block handles stride and channel change
        layers.append(Res2NeXtBlock(self.in_planes, planes, stride=stride))
        self.in_planes = planes
        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(Res2NeXtBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        l1 = self.layer1(x)  # 32x32, 64
        l2 = self.layer2(l1)  # 16x16, 128
        l3 = self.layer3(l2)  # 8x8, 256

        # Multi-Scale Aggregation
        out2 = F.adaptive_avg_pool2d(l2, 1).view(l2.size(0), -1)
        out3 = F.adaptive_avg_pool2d(l3, 1).view(l3.size(0), -1)

        out = torch.cat([out2, out3], dim=1)
        return self.fc(out)


# --- Training & Execution ---


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = calculate_roc_auc(targets, preds)
    return losses.avg, auc


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # TTA: Original
            out = torch.sigmoid(model(images))

            # TTA: Horizontal Flip
            out_h = torch.sigmoid(model(torch.flip(images, [3])))

            # TTA: Vertical Flip
            out_v = torch.sigmoid(model(torch.flip(images, [2])))

            # Average
            p = (out + out_h + out_v) / 3.0
            preds.extend(p.cpu().numpy().flatten())

    return np.array(preds)


def run_experiment():
    # Ensure directories exist
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working/idea_29", exist_ok=True)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
    )
    test_ids = get_test_ids(load_cached_data=True)

    final_preds = np.zeros(len(test_ids))

    # Homogeneous Seed Averaging
    for seed in range(N_SEEDS):
        print(f"Training Seed {seed}...")
        seed_everything(seed)

        model = WideAntiAliasedRes2NeXt().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0.0
        best_model_path = f"./working/idea_29/model_seed_{seed}.pth"

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
            )

        # Load best model for inference
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        # Predict with TTA
        seed_preds = predict(model, test_loader, DEVICE)
        final_preds += seed_preds

    # Average predictions across seeds
    final_preds /= N_SEEDS

    # Save Submission
    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    df_sub.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")


# Execute experiment
run_experiment()
