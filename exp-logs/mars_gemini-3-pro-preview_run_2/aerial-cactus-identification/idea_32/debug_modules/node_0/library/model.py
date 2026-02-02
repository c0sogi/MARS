import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import pandas as pd
from library.utils import AverageMeter, calculate_roc_auc, save_checkpoint, set_seed
from library.dataset import get_dataloaders

# =============================================================================
# RepVGG Block with Structural Re-parameterization
# =============================================================================


class RepVGGBlock(nn.Module):
    """
    RepVGG Block:
    - Training: Parallel 3x3, 1x1, and Identity branches.
    - Inference: Fused 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Activation
        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Branch 1: 3x3 Conv + BN
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            # Branch 2: 1x1 Conv + BN
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=padding - 1,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            # Branch 3: Identity + BN (only if dims match)
            self.rbr_identity = (
                nn.BatchNorm2d(in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # Identity branch is just BN
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.rbr_dense[0].in_channels,
            self.rbr_dense[0].out_channels,
            self.rbr_dense[0].kernel_size,
            self.rbr_dense[0].stride,
            self.rbr_dense[0].padding,
            self.rbr_dense[0].dilation,
            self.rbr_dense[0].groups,
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
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


# =============================================================================
# Wide RepVGG with Multi-Scale Aggregation
# =============================================================================


class WideRepVGG(nn.Module):
    """
    Wide RepVGG Architecture:
    - Backbone: 3 Stages with [64, 128, 256] channels.
    - Downsampling: 32x32 -> 16x16 -> 8x8.
    - Head: Multi-Scale Aggregation (Stage 2 + Stage 3) -> Dense.
    """

    def __init__(self, num_classes=1, deploy=False):
        super(WideRepVGG, self).__init__()
        self.deploy = deploy

        # Stage 1: 3 -> 64 channels, Stride 1 (32x32)
        # We use 2 blocks: 1 for transition/stride, 1 for depth
        self.stage1 = self._make_stage(3, 64, stride=1, num_blocks=2, deploy=deploy)

        # Stage 2: 64 -> 128 channels, Stride 2 (16x16)
        self.stage2 = self._make_stage(64, 128, stride=2, num_blocks=2, deploy=deploy)

        # Stage 3: 128 -> 256 channels, Stride 2 (8x8)
        self.stage3 = self._make_stage(128, 256, stride=2, num_blocks=2, deploy=deploy)

        # Multi-Scale Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Input to linear is 128 (Stage 2) + 256 (Stage 3) = 384
        self.linear = nn.Linear(128 + 256, num_classes)

    def _make_stage(self, in_channels, out_channels, stride, num_blocks, deploy):
        layers = []
        # First block handles channel change and stride
        layers.append(
            RepVGGBlock(in_channels, out_channels, stride=stride, deploy=deploy)
        )
        # Subsequent blocks are identity (stride 1, same channels)
        for _ in range(1, num_blocks):
            layers.append(
                RepVGGBlock(out_channels, out_channels, stride=1, deploy=deploy)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 1
        x = self.stage1(x)

        # Stage 2
        x = self.stage2(x)
        feat2 = x  # Capture 16x16 features

        # Stage 3
        x = self.stage3(x)
        feat3 = x  # Capture 8x8 features

        # Multi-Scale Aggregation
        # GAP on Stage 2 features
        g2 = self.gap(feat2).view(feat2.size(0), -1)
        # GAP on Stage 3 features
        g3 = self.gap(feat3).view(feat3.size(0), -1)

        # Concatenate
        combined = torch.cat([g2, g3], dim=1)

        # Classification
        out = self.linear(combined)
        return out

    def switch_to_deploy(self):
        if self.deploy:
            return
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True


# =============================================================================
# Training and Inference Utilities
# =============================================================================


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    scores = AverageMeter()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Metrics
        probs = torch.sigmoid(outputs)
        auc = calculate_roc_auc(labels, probs)

        losses.update(loss.item(), images.size(0))
        scores.update(auc, images.size(0))

    return losses.avg, scores.avg


def validate(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            auc = calculate_roc_auc(labels, probs)

            losses.update(loss.item(), images.size(0))
            scores.update(auc, images.size(0))

    return losses.avg, scores.avg


def train_model(seed, epochs=20, batch_size=64, save_dir="./working/idea_32"):
    """
    Trains a single model instance.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=True
    )

    # Model
    model = WideRepVGG(num_classes=1, deploy=False).to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    model_save_path = os.path.join(save_dir, f"model_seed_{seed}.pth")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training for Seed {seed}...")
    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            train_loader, model, criterion, optimizer, device
        )
        val_loss, val_auc = validate(val_loader, model, criterion, device)

        scheduler.step()

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
        )

    print(f"Finished Seed {seed}. Best Val AUC: {best_auc:.6f}")
    return model_save_path


def predict_with_tta(model, test_loader, device):
    """
    Generates predictions using Test Time Augmentation (Original, HFlip, VFlip).
    """
    model.eval()
    preds_dict = {}

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # TTA 1: Original
            out1 = torch.sigmoid(model(images))

            # TTA 2: Horizontal Flip
            out2 = torch.sigmoid(model(torch.flip(images, [3])))

            # TTA 3: Vertical Flip
            out3 = torch.sigmoid(model(torch.flip(images, [2])))

            # Average
            avg_preds = (out1 + out2 + out3) / 3.0
            avg_preds = avg_preds.cpu().numpy().flatten()

            for i, img_id in enumerate(ids):
                preds_dict[img_id] = avg_preds[i]

    return preds_dict


def run_inference_ensemble(seeds, save_dir="./working/idea_32"):
    """
    Loads models for all seeds, converts to deploy mode, and ensembles predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=128, load_cached_data=True)

    ensemble_preds = {}

    for seed in seeds:
        model_path = os.path.join(save_dir, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found at {model_path}")
            continue

        # Load Model
        model = WideRepVGG(num_classes=1, deploy=False)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Switch to Deploy Mode (Reparameterization)
        model.switch_to_deploy()
        model.to(device)
        model.eval()

        # Predict
        preds = predict_with_tta(model, test_loader, device)

        # Accumulate
        for img_id, prob in preds.items():
            if img_id not in ensemble_preds:
                ensemble_preds[img_id] = []
            ensemble_preds[img_id].append(prob)

    # Average across seeds
    final_submission = []
    for img_id, probs in ensemble_preds.items():
        avg_prob = np.mean(probs)
        final_submission.append({"id": img_id, "has_cactus": avg_prob})

    df_sub = pd.DataFrame(final_submission)

    # Save submission
    os.makedirs("submission", exist_ok=True)
    sub_path = "submission/submission.csv"
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def main_pipeline():
    """
    Main execution pipeline: Train 5 seeds, then ensemble inference.
    """
    seeds = [0, 1, 2, 3, 4]

    # Train
    for seed in seeds:
        train_model(seed, epochs=20)

    # Inference
    run_inference_ensemble(seeds)
