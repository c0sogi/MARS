import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from library import config, utils

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Residual Block with SE-Gate and Zero-Gamma Initialization.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> SE -> Add -> (No ReLU at end)
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        padding=1,
        use_se=True,
        se_reduction=16,
        zero_init_residual=True,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels, reduction=se_reduction)

        # Zero-Gamma Initialization for the last BN to ensure identity mapping at start
        if zero_init_residual:
            nn.init.constant_(self.bn2.weight, 0)
            nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out += residual
        # Note: We omit the final ReLU to allow the residual (noise) to be negative.
        return out


class SE_ZI_ResDnCNN(nn.Module):
    """
    SE-Gated Zero-Initialized Residual Denoising Network.
    Predicts the noise residual.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=20,
        kernel_size=3,
        padding=1,
        use_se=True,
        se_reduction=16,
        zero_init_residual=True,
    ):
        super(SE_ZI_ResDnCNN, self).__init__()

        # Head: Input -> Features
        self.head = nn.Conv2d(
            in_channels, num_features, kernel_size, padding=padding, bias=False
        )

        # Body: Stack of Residual Blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    num_features,
                    kernel_size,
                    padding,
                    use_se,
                    se_reduction,
                    zero_init_residual,
                )
            )
        self.body = nn.Sequential(*blocks)

        # Tail: Features -> Output (Noise)
        self.tail = nn.Conv2d(
            num_features, out_channels, kernel_size, padding=padding, bias=False
        )

        # Initialization for Head and Tail
        nn.init.kaiming_normal_(self.head.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.tail.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        noise = self.tail(feat)
        return noise


# =============================================================================
# DATA PROCESSING
# =============================================================================


def extract_patches(image, patch_size, stride):
    """Extracts patches from a single image."""
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
    return np.array(patches)


def prepare_data(load_cached_data=True):
    """
    Loads data from metadata, extracts patches, and caches them.
    Returns (train_patches, train_targets, val_patches, val_targets)
    """
    # Check cache first
    if load_cached_data:
        if (
            os.path.exists(config.TRAIN_PATCHES_PATH)
            and os.path.exists(config.TRAIN_TARGETS_PATH)
            and os.path.exists(config.VAL_PATCHES_PATH)
            and os.path.exists(config.VAL_TARGETS_PATH)
        ):
            print("Loading cached data...")
            train_patches = np.load(config.TRAIN_PATCHES_PATH)
            train_targets = np.load(config.TRAIN_TARGETS_PATH)
            val_patches = np.load(config.VAL_PATCHES_PATH)
            val_targets = np.load(config.VAL_TARGETS_PATH)
            return train_patches, train_targets, val_patches, val_targets

    print("Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    def process_split(df):
        patches_list = []
        targets_list = []

        for _, row in df.iterrows():
            input_path = os.path.join(config.INPUT_DIR, row["input_path"])
            target_path = os.path.join(config.INPUT_DIR, row["target_path"])

            # Read images
            img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

            if img_in is None or img_tar is None:
                continue

            # Normalize [0, 1]
            img_in = utils.normalize_image(img_in)
            img_tar = utils.normalize_image(img_tar)

            # Extract patches
            p_in = extract_patches(img_in, config.PATCH_SIZE, config.STRIDE)
            p_tar = extract_patches(img_tar, config.PATCH_SIZE, config.STRIDE)

            patches_list.append(p_in)
            targets_list.append(p_tar)

        return np.concatenate(patches_list), np.concatenate(targets_list)

    train_patches, train_targets = process_split(df_train)
    val_patches, val_targets = process_split(df_val)

    # Cache data
    np.save(config.TRAIN_PATCHES_PATH, train_patches)
    np.save(config.TRAIN_TARGETS_PATH, train_targets)
    np.save(config.VAL_PATCHES_PATH, val_patches)
    np.save(config.VAL_TARGETS_PATH, val_targets)

    return train_patches, train_targets, val_patches, val_targets


class DenoisingDataset(Dataset):
    def __init__(self, patches, targets, augment=False):
        self.patches = torch.from_numpy(patches).float().unsqueeze(1)  # Add channel dim
        self.targets = torch.from_numpy(targets).float().unsqueeze(1)
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]
        y = self.targets[idx]

        if self.augment:
            # Random flip H
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])
            # Random flip V
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [1])
                y = torch.flip(y, [1])
            # Random Rotate 90
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, [1, 2])
            y = torch.rot90(y, k, [1, 2])

        # Target for network is NOISE (Input - Clean)
        noise = x - y

        return x, noise


# =============================================================================
# TRAINING & INFERENCE
# =============================================================================


def train_model():
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    # Data
    train_patches, train_targets, val_patches, val_targets = prepare_data()

    train_dataset = DenoisingDataset(
        train_patches, train_targets, augment=config.USE_AUGMENTATION
    )
    val_dataset = DenoisingDataset(val_patches, val_targets, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Model
    model = SE_ZI_ResDnCNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_blocks=config.NUM_BLOCKS,
        kernel_size=config.KERNEL_SIZE,
        padding=config.PADDING,
        use_se=config.USE_SE,
        se_reduction=config.SE_REDUCTION,
        zero_init_residual=config.ZERO_INIT_RESIDUAL,
    ).to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=config.ETA_MIN
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataset)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}"
        )

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_metric": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


def predict_full_image(model, image, device):
    """
    Predicts noise for a full image using Geometric Self-Ensemble (TTA).
    """
    model.eval()
    img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

    if not config.USE_TTA:
        with torch.no_grad():
            noise_pred = model(img_tensor)
        return noise_pred.squeeze().cpu().numpy()

    # TTA: 8 versions (D4 Dihedral Group)
    transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 1, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 2, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3]),
    ]

    inverse_transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, -1, [2, 3]),
        lambda x: torch.rot90(x, -2, [2, 3]),
        lambda x: torch.rot90(x, -3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(torch.rot90(x, -1, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, -2, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, -3, [2, 3]), [3]),
    ]

    noise_accum = torch.zeros_like(img_tensor)

    with torch.no_grad():
        for t, inv_t in zip(transforms, inverse_transforms):
            aug_img = t(img_tensor)
            pred_noise = model(aug_img)
            pred_noise = inv_t(pred_noise)
            noise_accum += pred_noise

    noise_avg = noise_accum / 8.0
    return noise_avg.squeeze().cpu().numpy()


def generate_submission():
    print("Generating submission...")
    device = torch.device(config.DEVICE)

    # Load Model
    model = SE_ZI_ResDnCNN(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        num_features=config.NUM_FEATURES,
        num_blocks=config.NUM_BLOCKS,
        kernel_size=config.KERNEL_SIZE,
        padding=config.PADDING,
        use_se=config.USE_SE,
        se_reduction=config.SE_REDUCTION,
        zero_init_residual=config.ZERO_INIT_RESIDUAL,
    ).to(device)

    start_epoch, best_metric = utils.load_checkpoint(
        config.BEST_MODEL_PATH, model, device=device
    )
    print(f"Loaded model from epoch {start_epoch} with val loss {best_metric:.6f}")

    # Load Test Metadata
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    submission_rows = []

    for _, row in df_test.iterrows():
        img_id = row["image_id"]
        input_path = os.path.join(config.INPUT_DIR, row["input_path"])

        # Read and Normalize
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        h, w = img_in.shape
        img_norm = utils.normalize_image(img_in)

        # Predict Noise
        noise_pred = predict_full_image(model, img_norm, device)

        # Clean Image = Input - Noise
        clean_pred = img_norm - noise_pred

        # Clip to valid range [0, 1]
        clean_pred = np.clip(clean_pred, 0.0, 1.0)

        base_id = os.path.splitext(img_id)[0]

        # Construct submission rows efficiently
        # id format: image_row_col (e.g., 110_1_1)
        # value: float [0, 1]

        # Create grid of IDs
        r_indices = np.arange(1, h + 1)
        c_indices = np.arange(1, w + 1)

        # We iterate to construct the list, as string ops in numpy are tricky
        ids = []
        vals = []

        for r in range(h):
            for c in range(w):
                ids.append(f"{base_id}_{r+1}_{c+1}")
                vals.append(clean_pred[r, c])

        df_img = pd.DataFrame({"id": ids, "value": vals})
        submission_rows.append(df_img)

    # Concat all and save
    full_submission = pd.concat(submission_rows)
    full_submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")


def main():
    train_model()
    generate_submission()


if __name__ == "__main__":
    # Execute pipeline
    main()
