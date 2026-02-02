import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gc

from library.config import Config
from library.utils import load_image, calculate_rmse, save_submission, seed_everything

# --- Model Architecture ---


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat([x, out], 1)


class RDB(nn.Module):
    """Residual Dense Block"""

    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        layers = []
        for i in range(C):
            layers.append(DenseLayer(G0 + i * G, G))
        self.layers = nn.Sequential(*layers)

        # Local Feature Fusion
        self.lff = nn.Conv2d(G0 + C * G, G0, kernel_size=1)

    def forward(self, x):
        out = self.layers(x)
        out = self.lff(out)
        return out + x  # Local Residual Learning


class RDN(nn.Module):
    """Residual Dense Network for Noise Prediction"""

    def __init__(self):
        super(RDN, self).__init__()
        G0 = Config.RDN_G0
        C = Config.RDN_NUM_LAYERS
        G = Config.RDN_G0  # Using G0 as growth rate
        D = Config.RDN_NUM_BLOCKS
        kernel_size = Config.RDN_KERNEL_SIZE
        in_channels = Config.CHANNELS

        # Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(in_channels, G0, kernel_size=kernel_size, padding=1)
        self.sfe2 = nn.Conv2d(G0, G0, kernel_size=kernel_size, padding=1)

        # Residual Dense Blocks
        self.rdbs = nn.ModuleList()
        for _ in range(D):
            self.rdbs.append(RDB(G0, C, G))

        # Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(D * G0, G0, kernel_size=1),
            nn.Conv2d(G0, G0, kernel_size=3, padding=1),
        )

        # Output Layer (Predicts Noise)
        self.output = nn.Conv2d(G0, in_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        f_1 = self.sfe1(x)
        f_x = self.sfe2(f_1)

        rdb_outs = []
        for rdb in self.rdbs:
            f_x = rdb(f_x)
            rdb_outs.append(f_x)

        # Global Feature Fusion
        f_cat = torch.cat(rdb_outs, dim=1)
        f_gff = self.gff(f_cat)

        # Global Residual Learning (Feature Space)
        # Combine deep features with shallow features
        f_df = f_gff + f_1

        # Predict Noise
        noise = self.output(f_df)

        return noise


# --- Data Handling ---


class PatchDataset(Dataset):
    def __init__(self, patches, targets=None, is_train=True):
        """
        patches: (N, H, W) numpy array
        targets: (N, H, W) numpy array, representing the NOISE (Input - Clean)
        """
        self.patches = torch.from_numpy(patches).float().unsqueeze(1)  # (N, 1, H, W)
        self.is_train = is_train
        if is_train and targets is not None:
            self.targets = (
                torch.from_numpy(targets).float().unsqueeze(1)
            )  # (N, 1, H, W)
        else:
            self.targets = None

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        if self.is_train:
            return self.patches[idx], self.targets[idx]
        return self.patches[idx]


def extract_patches(image, patch_size, stride):
    """Extracts patches from a single image."""
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y : y + patch_size, x : x + patch_size])
    return np.array(patches)


def prepare_data(load_cached_data=True):
    """
    Loads data, extracts patches, and caches them.
    Returns: (train_patches, train_noise_targets), (val_patches, val_noise_targets)
    """
    # Check cache
    if (
        load_cached_data
        and os.path.exists(Config.TRAIN_PATCHES_CACHE)
        and os.path.exists(Config.VAL_PATCHES_CACHE)
    ):
        print("Loading cached patches...")
        train_data = np.load(Config.TRAIN_PATCHES_CACHE, allow_pickle=True).item()
        val_data = np.load(Config.VAL_PATCHES_CACHE, allow_pickle=True).item()
        return (train_data["patches"], train_data["targets"]), (
            val_data["patches"],
            val_data["targets"],
        )

    print("Generating patches from scratch...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    def process_split(df, stride, augment=False):
        all_patches = []
        all_targets = []  # We predict noise: Input - Clean

        for _, row in df.iterrows():
            input_path = os.path.join(Config.INPUT_DIR, row["input_path"])
            target_path = os.path.join(Config.INPUT_DIR, row["target_path"])

            img_in = load_image(input_path)
            img_tar = load_image(target_path)

            # Calculate Noise Map
            noise_map = img_in - img_tar

            patches_in = extract_patches(img_in, Config.PATCH_SIZE, stride)
            patches_noise = extract_patches(noise_map, Config.PATCH_SIZE, stride)

            if augment:
                # Augmentation: Flips and Rotations
                aug_in = []
                aug_noise = []
                for p_in, p_n in zip(patches_in, patches_noise):
                    # Original
                    aug_in.append(p_in)
                    aug_noise.append(p_n)
                    # Flip H
                    aug_in.append(np.flip(p_in, axis=1))
                    aug_noise.append(np.flip(p_n, axis=1))
                    # Flip V
                    aug_in.append(np.flip(p_in, axis=0))
                    aug_noise.append(np.flip(p_n, axis=0))
                    # Rot 90
                    aug_in.append(np.rot90(p_in, k=1))
                    aug_noise.append(np.rot90(p_n, k=1))

                all_patches.extend(aug_in)
                all_targets.extend(aug_noise)
            else:
                all_patches.extend(patches_in)
                all_targets.extend(patches_noise)

        return np.array(all_patches), np.array(all_targets)

    # Process Train (Small stride for high density, with augmentation)
    print("Processing Training Set...")
    train_patches, train_targets = process_split(
        df_train, stride=Config.STRIDE, augment=True
    )

    # Process Val (Larger stride to reduce redundancy and evaluation time)
    print("Processing Validation Set...")
    val_patches, val_targets = process_split(
        df_val, stride=Config.PATCH_SIZE, augment=False
    )

    # Cache
    print(f"Saving cache to {Config.WORKING_DIR}...")
    np.save(
        Config.TRAIN_PATCHES_CACHE, {"patches": train_patches, "targets": train_targets}
    )
    np.save(Config.VAL_PATCHES_CACHE, {"patches": val_patches, "targets": val_targets})

    return (train_patches, train_targets), (val_patches, val_targets)


# --- Training & Inference ---


def train_model():
    seed_everything()
    device = torch.device(Config.DEVICE)

    # Data
    (train_x, train_y), (val_x, val_y) = prepare_data(load_cached_data=True)

    print(f"Training Patches: {train_x.shape}")
    print(f"Validation Patches: {val_x.shape}")

    train_dataset = PatchDataset(train_x, train_y, is_train=True)
    val_dataset = PatchDataset(val_x, val_y, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = RDN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    early_stop_counter = 0
    patience_limit = 10

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_mse += loss.item() * inputs.size(0)

        val_mse /= len(val_dataset)
        val_rmse = np.sqrt(val_mse)

        scheduler.step(val_rmse)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val RMSE: {val_rmse:.8f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), Config.MODEL_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience_limit:
            print("Early stopping triggered.")
            break

    print(f"Best Validation RMSE: {best_rmse:.8f}")

    # Free memory
    del (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        train_x,
        train_y,
        val_x,
        val_y,
    )
    gc.collect()
    torch.cuda.empty_cache()


def generate_submission():
    print("Generating submission...")
    device = torch.device(Config.DEVICE)

    # Load Model
    model = RDN().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Model checkpoint not found. Using untrained model.")

    model.eval()

    df_test = pd.read_csv(Config.TEST_METADATA)
    predictions = {}

    with torch.no_grad():
        for _, row in df_test.iterrows():
            img_id = row["image_id"]
            input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

            # Load full image
            img_in = load_image(input_path)
            h, w = img_in.shape

            # Prepare input tensor (1, 1, H, W)
            img_tensor = (
                torch.from_numpy(img_in).float().unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict Noise
            # Since RDN is fully convolutional, we can pass the whole image
            # If OOM occurs, we would need to tile, but 540x420 fits in A100 easily.
            noise_pred = model(img_tensor)

            noise_pred_np = noise_pred.squeeze().cpu().numpy()

            # Reconstruct Clean Image: Input - Noise
            clean_pred = img_in - noise_pred_np

            predictions[img_id] = clean_pred

    save_submission(predictions)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    Config.display()
    train_model()
    generate_submission()


# Execute main
main()
