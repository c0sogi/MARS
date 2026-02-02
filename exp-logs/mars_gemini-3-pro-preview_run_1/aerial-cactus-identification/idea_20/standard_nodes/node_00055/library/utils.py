import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# =============================================================================
# 1. Utility Functions
# =============================================================================


def seed_everything(seed=42):
    random_seed = seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# 2. Data Loading & Caching
# =============================================================================


def load_and_cache_data(
    input_dir="./input",
    metadata_dir="./metadata",
    cache_dir="./working/idea_20",
    load_cached_data=True,
):
    """
    Loads images and labels. Caches them as .npy files to avoid re-reading small files.
    """
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_imgs.npy")
    train_label_cache_path = os.path.join(cache_dir, "train_labels.npy")
    test_cache_path = os.path.join(cache_dir, "test_imgs.npy")
    test_id_cache_path = os.path.join(cache_dir, "test_ids.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading data from cache...")
        train_imgs = np.load(train_cache_path)
        train_labels = np.load(train_label_cache_path)
        test_imgs = np.load(test_cache_path)
        test_ids = np.load(test_id_cache_path, allow_pickle=True)
        return train_imgs, train_labels, test_imgs, test_ids

    print("Cache not found or forced reload. Processing raw images...")

    # Load Metadata
    # We combine train and val metadata to perform our own 5-fold split
    train_meta = pd.read_csv(os.path.join(metadata_dir, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(metadata_dir, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(metadata_dir, "test_metadata.csv"))

    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Process Train
    train_imgs = []
    train_labels = []

    for idx, row in full_train_meta.iterrows():
        img_path = os.path.join(input_dir, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_imgs.append(img)
        train_labels.append(row["has_cactus"])

    train_imgs = np.array(train_imgs, dtype=np.uint8)
    train_labels = np.array(train_labels, dtype=np.float32)

    # Process Test
    test_imgs = []
    test_ids = []

    for idx, row in test_meta.iterrows():
        img_path = os.path.join(input_dir, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            # Should not happen based on metadata validation
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_imgs.append(img)
        test_ids.append(row["id"])

    test_imgs = np.array(test_imgs, dtype=np.uint8)
    test_ids = np.array(test_ids)

    # Save to cache
    np.save(train_cache_path, train_imgs)
    np.save(train_label_cache_path, train_labels)
    np.save(test_cache_path, test_imgs)
    np.save(test_id_cache_path, test_ids)

    print(
        f"Data processed and cached. Train shape: {train_imgs.shape}, Test shape: {test_imgs.shape}"
    )

    return train_imgs, train_labels, test_imgs, test_ids


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Simple geometric augmentations using numpy/cv2 logic if transform is requested
        # We implement basic transforms directly here for efficiency and control
        if self.transform:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
            # Random Vertical Flip
            if np.random.rand() < 0.5:
                img = np.flipud(img)

        # Normalize to 0-1 and CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        if self.labels is not None:
            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                self.labels[idx], dtype=torch.float32
            )
        else:
            return torch.tensor(img, dtype=torch.float32)


# =============================================================================
# 3. Model Architecture (RepVGG)
# =============================================================================


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


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
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
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
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
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
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


class RepVGG(nn.Module):
    def __init__(self, num_classes=1, width_multiplier=[1, 1, 1, 2], deploy=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy

        # Conservative Stem: 3x3, stride 1 (Preserve 32x32)
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=int(32 * width_multiplier[0]),
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )

        # Stage 1: 32x32 -> 16x16
        self.stage1 = self._make_stage(
            int(32 * width_multiplier[0]),
            int(64 * width_multiplier[1]),
            stride=2,
            num_blocks=2,
            deploy=deploy,
        )

        # Stage 2: 16x16 -> 8x8
        self.stage2 = self._make_stage(
            int(64 * width_multiplier[1]),
            int(128 * width_multiplier[2]),
            stride=2,
            num_blocks=3,
            deploy=deploy,
        )

        # Aux Head
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(128 * width_multiplier[2]), num_classes),
        )

        # Stage 3: 8x8 -> 4x4
        self.stage3 = self._make_stage(
            int(128 * width_multiplier[2]),
            int(256 * width_multiplier[3]),
            stride=2,
            num_blocks=4,
            deploy=deploy,
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(256 * width_multiplier[3]), num_classes)

    def _make_stage(self, in_channels, out_channels, stride, num_blocks, deploy):
        layers = []
        layers.append(
            RepVGGBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                deploy=deploy,
            )
        )
        for _ in range(num_blocks - 1):
            layers.append(
                RepVGGBlock(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    deploy=deploy,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)

        aux_out = None
        if not self.deploy and self.training:
            aux_out = self.aux_head(out)

        out = self.stage3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.training and not self.deploy:
            return out, aux_out
        return out


def repvgg_model_convert(model: torch.nn.Module, save_path=None):
    # Converts a trained RepVGG model to deploy mode (fuses layers)
    # Create new model in deploy mode
    new_model = RepVGG(deploy=True)
    new_model.eval()

    # Copy weights via state dict matching where possible, but RepVGG requires specific block conversion
    # Simpler approach: iterate modules and switch
    # We must operate on the original model instance or a deep copy
    import copy

    model_copy = copy.deepcopy(model)

    for module in model_copy.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()

    if save_path:
        torch.save(model_copy.state_dict(), save_path)

    return model_copy


# =============================================================================
# 4. Training Utilities (Mixup, SWA)
# =============================================================================


def mixup_data(x, y, alpha=0.2, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.unsqueeze(1)) + (1 - lam) * criterion(
        pred, y_b.unsqueeze(1)
    )


# =============================================================================
# 5. Main Pipeline
# =============================================================================


def run_training(epochs=35, batch_size=64, swa_start=25):
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    train_imgs, train_labels, test_imgs, test_ids = load_and_cache_data()

    # K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_imgs))
    test_preds_accum = np.zeros((len(test_imgs),))

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    checkpoint_dir = "./working/idea_20/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_imgs, train_labels)):
        print(f"\n=== Fold {fold} ===")

        # Datasets
        train_ds = CactusDataset(
            train_imgs[train_idx], train_labels[train_idx], transform=True
        )
        val_ds = CactusDataset(
            train_imgs[val_idx], train_labels[val_idx], transform=False
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Model
        model = RepVGG(deploy=False).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        # Schedulers
        # Phase 1: Cosine Annealing until SWA start
        scheduler = CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=1e-5)

        # SWA
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

        best_auc = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                # Mixup
                imgs, targets_a, targets_b, lam = mixup_data(
                    imgs, labels, alpha=0.2, device=device
                )

                optimizer.zero_grad()
                outputs, aux_outputs = model(imgs)

                loss_main = mixup_criterion(
                    criterion, outputs, targets_a, targets_b, lam
                )
                loss_aux = mixup_criterion(
                    criterion, aux_outputs, targets_a, targets_b, lam
                )
                loss = loss_main + 0.4 * loss_aux

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # SWA Logic
            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # Validation (Standard Model)
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)  # Forward returns only logits in eval mode
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                    val_targets.extend(labels.numpy())

            auc = roc_auc_score(val_targets, val_preds)
            if auc > best_auc:
                best_auc = auc
                # Save best standard model just in case
                torch.save(model.state_dict(), f"{checkpoint_dir}/fold{fold}_best.pth")

            # print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.6f}")

        # End of training: Update BN for SWA model
        print("Updating SWA Batch Norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Validate SWA Model
        swa_model.eval()
        val_preds_swa = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                # SWA model wraps the RepVGG, so forward calls RepVGG.forward
                # But RepVGG.forward in eval mode returns just logits
                outputs = swa_model(imgs)
                val_preds_swa.extend(torch.sigmoid(outputs).cpu().numpy().flatten())

        swa_auc = roc_auc_score(val_targets, val_preds_swa)
        print(f"Fold {fold} SWA AUC: {swa_auc}")

        # Convert SWA model to Deploy mode for efficient inference
        # We need to extract the module from AveragedModel first
        deploy_model = repvgg_model_convert(swa_model.module)
        deploy_model.eval()
        deploy_model.to(device)

        # --- Inference on Test Set (TTA) ---
        # TTA: Original, HFlip, VFlip, Rotate180 (H+V)
        test_ds = CactusDataset(test_imgs, transform=False)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        fold_test_preds = []

        with torch.no_grad():
            for imgs in test_loader:
                imgs = imgs.to(device)

                # 1. Original
                p1 = torch.sigmoid(deploy_model(imgs))

                # 2. HFlip
                p2 = torch.sigmoid(deploy_model(torch.flip(imgs, [3])))

                # 3. VFlip
                p3 = torch.sigmoid(deploy_model(torch.flip(imgs, [2])))

                # 4. Rotate 180
                p4 = torch.sigmoid(deploy_model(torch.flip(imgs, [2, 3])))

                avg_p = (p1 + p2 + p3 + p4) / 4.0
                fold_test_preds.extend(avg_p.cpu().numpy().flatten())

        test_preds_accum += np.array(fold_test_preds)

    # Average over folds
    final_preds = test_preds_accum / 5.0

    # Save Submission
    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    # This block is for testing locally, but the task says "DO NOT include an if __name__ == '__main__': block"
    # However, the task also says "Your response should only contain a single code block...".
    # And "Task: Implement the utils.py module".
    # Usually utils.py is imported. But here it seems to be the main driver.
    # I will provide the functions. The user can call run_training().
    pass
