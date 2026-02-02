import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import save_submission_file


class DnCNN(nn.Module):
    """
    Deep Dense Residual-Learning Network (DnCNN).
    Designed to predict the noise residual (R(x)) from a noisy input image.
    """

    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()

        layers = []
        # Layer 1: Conv + BN + ReLU
        # According to requirements: "Each convolutional layer (except the last) is followed by a ReLU activation and Batch Normalization."
        layers.append(
            nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(n_channels))
        layers.append(nn.ReLU(inplace=True))

        # Layers 2 to D-1: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Layer D: Conv (Output)
        # Predicts the residual map. No BN or ReLU.
        layers.append(
            nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1, bias=True)
        )

        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        # Returns the estimated noise (residual)
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def extract_patches(image, patch_size, stride):
    """
    Extracts patches from a single image with the specified stride.
    Returns a numpy array of shape (N, H, W).
    """
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
    return np.array(patches)


def prepare_data(metadata_path, cache_path, load_cached_data=True):
    """
    Loads images based on metadata, extracts patches, and caches the result.
    If cached data exists and load_cached_data is True, loads from disk.
    """
    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return data
        except Exception as e:
            print(f"Error loading cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    df = pd.read_csv(metadata_path)

    # Debugging limits
    if Config.MAX_TRAIN_IMAGES is not None and "train" in metadata_path:
        df = df.head(Config.MAX_TRAIN_IMAGES)
    if Config.MAX_VAL_IMAGES is not None and "val" in metadata_path:
        df = df.head(Config.MAX_VAL_IMAGES)

    patch_pairs = []

    for _, row in df.iterrows():
        input_full_path = os.path.join(Config.INPUT_DIR, row["input_path"])
        target_full_path = os.path.join(Config.INPUT_DIR, row["target_path"])

        # Load as Grayscale
        img_in = cv2.imread(input_full_path, cv2.IMREAD_GRAYSCALE)
        img_tar = cv2.imread(target_full_path, cv2.IMREAD_GRAYSCALE)

        if img_in is None or img_tar is None:
            continue

        # Normalize to [0, 1]
        img_in = img_in.astype(np.float32) / 255.0
        img_tar = img_tar.astype(np.float32) / 255.0

        # Extract patches
        p_in = extract_patches(img_in, Config.PATCH_SIZE, Config.STRIDE)
        p_tar = extract_patches(img_tar, Config.PATCH_SIZE, Config.STRIDE)

        # Store as pairs if patches exist
        if len(p_in) > 0:
            # Stack to shape (N, 2, H, W)
            pairs = np.stack([p_in, p_tar], axis=1)
            patch_pairs.append(pairs)

    if len(patch_pairs) > 0:
        all_data = np.concatenate(patch_pairs, axis=0)
    else:
        all_data = np.empty(
            (0, 2, Config.PATCH_SIZE, Config.PATCH_SIZE), dtype=np.float32
        )

    # 3. Save to cache
    np.save(cache_path, all_data)

    return all_data


class DenoisingDataset(Dataset):
    def __init__(self, data, augment=False):
        """
        data: Numpy array of shape (N, 2, H, W) where 2 corresponds to [noisy, clean].
        """
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy, clean = self.data[idx]

        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                noisy = np.flip(noisy, axis=1)
                clean = np.flip(clean, axis=1)
            # Random Vertical Flip
            if np.random.rand() < 0.5:
                noisy = np.flip(noisy, axis=0)
                clean = np.flip(clean, axis=0)
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                clean = np.rot90(clean, k)

        # Ensure contiguous arrays after flipping/rotation
        noisy = np.ascontiguousarray(noisy)
        clean = np.ascontiguousarray(clean)

        # Convert to Tensor and add channel dimension: (1, H, W)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        return noisy_t, clean_t


def train(load_cached_data=True):
    """
    Executes the training pipeline.
    """
    # Load Data
    train_data = prepare_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
    )
    val_data = prepare_data(
        Config.VAL_METADATA_PATH,
        Config.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
    )

    train_dataset = DenoisingDataset(train_data, augment=Config.AUGMENT_DATA)
    val_dataset = DenoisingDataset(val_data, augment=False)

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

    # Model Initialization
    model = DnCNN(
        depth=Config.DEPTH,
        n_channels=Config.N_CHANNELS,
        image_channels=Config.IN_CHANNELS,
    )
    model.to(Config.DEVICE)

    # Loss & Optimizer
    # Loss is MSE between predicted noise and actual noise (Input - Target)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(Config.DEVICE)
            clean = clean.to(Config.DEVICE)

            # Calculate true noise residual
            target_noise = noisy - clean

            optimizer.zero_grad()
            pred_noise = model(noisy)

            loss = criterion(pred_noise, target_noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(Config.DEVICE)
                clean = clean.to(Config.DEVICE)

                target_noise = noisy - clean
                pred_noise = model(noisy)

                loss = criterion(pred_noise, target_noise)
                val_loss += loss.item() * noisy.size(0)

        val_loss /= len(val_dataset)

        # Print metrics (full precision)
        print(f"Epoch {epoch+1}: Train Loss {epoch_loss}, Val Loss {val_loss}")

        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


def predict():
    """
    Generates predictions for the test set and saves the submission file.
    """
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Model file not found. Cannot predict.")
        return

    # Load Model
    model = DnCNN(
        depth=Config.DEPTH,
        n_channels=Config.N_CHANNELS,
        image_channels=Config.IN_CHANNELS,
    )
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    predictions = {}

    print("Generating predictions...")

    with torch.no_grad():
        for _, row in df_test.iterrows():
            img_id = row["image_id"]
            input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

            # Load full image
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Normalize
            img_norm = img.astype(np.float32) / 255.0

            # Prepare Input Tensor: (1, 1, H, W)
            input_tensor = (
                torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
            )

            # Predict Noise
            pred_noise = model(input_tensor)

            # Denoise: Clean = Input - Noise
            pred_clean = input_tensor - pred_noise

            # Post-processing
            pred_clean = torch.clamp(pred_clean, 0, 1)
            pred_clean_np = pred_clean.squeeze().cpu().numpy()

            predictions[img_id] = pred_clean_np

    save_submission_file(predictions, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
