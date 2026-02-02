import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders

# =============================================================================
# 1. Model Components: RepNeXt, Res2Net, SEBlock
# =============================================================================


def get_valid_groups(channels, target_groups=32):
    """Calculates a valid group number that divides channels, closest to target."""
    if channels % target_groups == 0:
        return target_groups
    for g in range(target_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RepNeXtUnit(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, target_groups=32, deploy=False
    ):
        super(RepNeXtUnit, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = get_valid_groups(in_channels, target_groups)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=self.groups,
                bias=True,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=self.groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=self.groups,
            )

    def forward(self, inputs):
        if self.deploy:
            return self.rbr_reparam(inputs)

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)
        kernelid, biasid = self._get_identity_tensor()
        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _get_identity_tensor(self):
        if self.rbr_identity is None:
            return 0, 0

        input_dim = self.in_channels // self.groups
        kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, 1, 1] = 1

        id_tensor = torch.from_numpy(kernel_value).to(self.rbr_identity.weight.device)

        running_mean = self.rbr_identity.running_mean
        running_var = self.rbr_identity.running_var
        gamma = self.rbr_identity.weight
        beta = self.rbr_identity.bias
        eps = self.rbr_identity.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return id_tensor * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            self.stride,
            1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        self.deploy = True


class Res2NetBottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, scales=4, groups=32, deploy=False
    ):
        super(Res2NetBottleneck, self).__init__()
        self.scales = scales
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Internal width. For Ultra-Wide capacity, we maintain high internal width.
        self.width = out_channels // 2

        self.conv1 = conv_bn(in_channels, self.width, 1, 1, 0)

        self.width_per_scale = self.width // scales
        self.convs = nn.ModuleList()

        for i in range(scales - 1):
            self.convs.append(
                RepNeXtUnit(
                    self.width_per_scale,
                    self.width_per_scale,
                    stride=stride,
                    target_groups=groups,
                    deploy=deploy,
                )
            )

        self.conv3 = conv_bn(self.width, out_channels, 1, 1, 0)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if stride == 2:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(2, 2), conv_bn(in_channels, out_channels, 1, 1, 0)
                )
            else:
                self.shortcut = conv_bn(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)

        spx = torch.split(out, self.width_per_scale, 1)
        sp = self.convs[0](spx[1])

        y = []
        if self.stride > 1:
            y.append(F.avg_pool2d(spx[0], 3, stride=self.stride, padding=1))
        else:
            y.append(spx[0])

        y.append(sp)

        for i in range(1, self.scales - 1):
            if self.stride == 1:
                sp = sp + spx[i + 1]
            else:
                # Simplified Res2Net logic for stride > 1 (not used in this specific architecture plan but kept for robustness)
                pass
            sp = self.convs[i](sp + spx[i + 1] if self.stride == 1 else sp)
            y.append(sp)

        out = torch.cat(y, 1)
        out = self.conv3(out)
        out = self.se(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class RepDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepDownsample, self).__init__()
        self.deploy = deploy
        self.conv3x3 = conv_bn(in_channels, out_channels, 3, 2, 1)
        self.conv1x1 = conv_bn(in_channels, out_channels, 1, 2, 0)

        if deploy:
            self.fused_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True)

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        return self.conv3x3(x) + self.conv1x1(x)

    def switch_to_deploy(self):
        if self.deploy:
            return
        k3, b3 = self._fuse_bn(self.conv3x3)
        k1, b1 = self._fuse_bn(self.conv1x1)
        k1 = F.pad(k1, [1, 1, 1, 1])

        self.fused_conv = nn.Conv2d(
            self.conv3x3.conv.in_channels,
            self.conv3x3.conv.out_channels,
            3,
            2,
            1,
            bias=True,
        )
        self.fused_conv.weight.data = k3 + k1
        self.fused_conv.bias.data = b3 + b1

        self.__delattr__("conv3x3")
        self.__delattr__("conv1x1")
        self.deploy = True

    def _fuse_bn(self, branch):
        k = branch.conv.weight
        rm = branch.bn.running_mean
        rv = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (rv + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return k * t, beta - rm * gamma / std


class CactusResNet(nn.Module):
    def __init__(self, num_classes=1, deploy=False):
        super(CactusResNet, self).__init__()
        self.deploy = deploy

        widths = [96, 192, 384]

        # Stem: 3 -> 96
        self.stem = RepNeXtUnit(3, widths[0], stride=1, target_groups=1, deploy=deploy)

        # Stage 1: 96 -> 96 (32x32)
        self.stage1 = Res2NetBottleneck(widths[0], widths[0], stride=1, deploy=deploy)

        # Downsample: 96 -> 192 (32x32 -> 16x16)
        self.ds1 = RepDownsample(widths[0], widths[1], deploy=deploy)

        # Stage 2: 192 -> 192 (16x16)
        self.stage2 = Res2NetBottleneck(widths[1], widths[1], stride=1, deploy=deploy)

        # Downsample: 192 -> 384 (16x16 -> 8x8)
        self.ds2 = RepDownsample(widths[1], widths[2], deploy=deploy)

        # Stage 3: 384 -> 384 (8x8)
        self.stage3 = Res2NetBottleneck(widths[2], widths[2], stride=1, deploy=deploy)

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[1] + widths[2], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        x = self.ds1(x)  # 16x16
        x2 = self.stage2(x)

        x = self.ds2(x2)  # 8x8
        x3 = self.stage3(x)

        # Multi-Scale Aggregation
        f2 = self.gap(x2).flatten(1)
        f3 = self.gap(x3).flatten(1)

        out = torch.cat([f2, f3], dim=1)
        out = self.fc(out)
        return out

    def switch_to_deploy(self):
        if self.deploy:
            return
        for m in self.modules():
            if isinstance(m, RepNeXtUnit):
                m.switch_to_deploy()
            if isinstance(m, RepDownsample):
                m.switch_to_deploy()
        self.deploy = True


# =============================================================================
# 2. Training and Evaluation Logic
# =============================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5

    return epoch_loss, auc


def run_training(seeds=[0, 1, 2, 3, 4], epochs=20, batch_size=64, debug_size=None):
    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, debug_size=debug_size
    )

    best_models = []

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)

        model = CactusResNet(num_classes=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_auc = 0.0
        best_state = None
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best state
        model.load_state_dict(best_state)
        best_models.append(model)

    return best_models, test_loader


def generate_submission(models, test_loader):
    device = get_device()
    print("\nGenerating submission with TTA...")

    # Prepare TTA transforms
    # We do TTA manually

    results = {}

    for model in models:
        model.eval()

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            batch_preds = []

            # TTA 1: Original
            for model in models:
                out = torch.sigmoid(model(images))
                batch_preds.append(out.cpu().numpy())

            # TTA 2: Horizontal Flip
            images_h = torch.flip(images, [3])
            for model in models:
                out = torch.sigmoid(model(images_h))
                batch_preds.append(out.cpu().numpy())

            # TTA 3: Vertical Flip
            images_v = torch.flip(images, [2])
            for model in models:
                out = torch.sigmoid(model(images_v))
                batch_preds.append(out.cpu().numpy())

            # Average predictions (Seeds x TTA)
            batch_preds = np.mean(batch_preds, axis=0).flatten()

            for img_id, pred in zip(ids, batch_preds):
                results[img_id] = pred

    # Create DataFrame
    sub_df = pd.DataFrame(list(results.items()), columns=["id", "has_cactus"])

    # Ensure output directory exists
    os.makedirs("submission", exist_ok=True)
    sub_path = "submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def main(debug=False):
    # Hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 64
    SEEDS = [0, 1, 2, 3, 4]

    if debug:
        EPOCHS = 2
        SEEDS = [0]
        DEBUG_SIZE = 100
    else:
        DEBUG_SIZE = None

    models, test_loader = run_training(
        seeds=SEEDS, epochs=EPOCHS, batch_size=BATCH_SIZE, debug_size=DEBUG_SIZE
    )
    generate_submission(models, test_loader)


if __name__ == "__main__":
    # Check if we are in a debug environment or full run
    # Default to full run
    main(debug=False)
