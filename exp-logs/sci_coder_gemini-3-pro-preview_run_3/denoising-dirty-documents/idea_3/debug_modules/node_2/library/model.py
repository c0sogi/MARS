import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import (
    set_seed,
    calculate_rmse,
    save_checkpoint,
    normalize_image,
    denormalize_image,
    get_cached_data,
)

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class DRDN(nn.Module):
    """
    Dilated Residual Denoising Network (DRDN).
    Predicts the noise residual from a noisy input image.
    """

    def __init__(self):
        super(DRDN, self).__init__()

        self.in_channels = Config.IN_CHANNELS
        self.out_channels = Config.OUT_CHANNELS
        self.hidden_channels = Config.HIDDEN_CHANNELS
        self.kernel_size = Config.KERNEL_SIZE
        self.dilation_rates = Config.DILATION_RATES

        layers = []

        # 1. Feature Extraction (Input -> Hidden)
        # Standard convolution
        layers.append(
            nn.Conv2d(
                self.in_channels,
                self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # 2. Dilated Residual Blocks (Hidden -> Hidden)
        # Stack of dilated convolutions to expand receptive field
        for dilation in self.dilation_rates:
            padding = dilation * (self.kernel_size // 2)
            layers.append(
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )
            layers.append(nn.ReLU(inplace=True))

        # 3. Reconstruction (Hidden -> Output)
        # Projects back to noise map (1 channel). No activation (noise can be +/-).
        layers.append(
            nn.Conv2d(
                self.hidden_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                bias=True,
            )
        )

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Returns the predicted noise
        return self.model(x)


# -------------------------------------------------------------------------
# Data Processing & Dataset
# -------------------------------------------------------------------------


def _compute_patches(metadata_path, patch_size, stride, is_train=True):
    """
    Internal function to compute patches from images listed in metadata.
    Used by get_cached_data.
    """
    df = pd.read_csv(metadata_path)

    inputs = []
    targets = []

    for _, row in df.iterrows():
        input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

        # Load image in grayscale
        img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img_in is None:
            continue

        img_in = normalize_image(img_in)

        if is_train:
            target_path = os.path.join(Config.INPUT_DIR, row["target_path"])
            img_tar = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            if img_tar is None:
                continue
            img_tar = normalize_image(img_tar)
        else:
            img_tar = None

        h, w = img_in.shape

        # Extract patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch_in = img_in[y : y + patch_size, x : x + patch_size]
                inputs.append(patch_in)

                if is_train:
                    patch_tar = img_tar[y : y + patch_size, x : x + patch_size]
                    targets.append(patch_tar)

    inputs = np.array(inputs, dtype=np.float32)
    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    inputs = np.expand_dims(inputs, axis=1)

    if is_train:
        targets = np.array(targets, dtype=np.float32)
        targets = np.expand_dims(targets, axis=1)
        return np.stack([inputs, targets], axis=0)  # Stack to return single object
    else:
        return inputs


class DenoisingDataset(Dataset):
    def __init__(self, data_array, transform=None):
        """
        Args:
            data_array: Numpy array of shape (2, N, 1, H, W) for train (input, target)
                        or (N, 1, H, W) for inference.
            transform: Boolean, whether to apply augmentations.
        """
        self.is_train = False
        if data_array.ndim == 5 and data_array.shape[0] == 2:
            self.inputs = data_array[0]
            self.targets = data_array[1]
            self.is_train = True
        else:
            self.inputs = data_array
            self.targets = None

        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])

        if self.is_train:
            y = torch.from_numpy(self.targets[idx])

            if self.transform:
                # Random Horizontal Flip
                if torch.rand(1) < 0.5:
                    x = torch.flip(x, [2])
                    y = torch.flip(y, [2])

                # Random Vertical Flip
                if torch.rand(1) < 0.5:
                    x = torch.flip(x, [1])
                    y = torch.flip(y, [1])

                # Random Rotation (0, 90, 180, 270)
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    x = torch.rot90(x, k, [1, 2])
                    y = torch.rot90(y, k, [1, 2])

            return x, y
        else:
            return x


# -------------------------------------------------------------------------
# Training Logic
# -------------------------------------------------------------------------


def train_model(load_cached_data_flag=True):
    set_seed(Config.SEED)
    Config.create_directories()

    device = torch.device(Config.DEVICE)

    # --- Data Preparation ---
    print("Preparing training data...")
    # We cache the combined input/target array
    train_data_packed = get_cached_data(
        cache_filename="train_patches_packed.npy",
        compute_func=_compute_patches,
        load_cached_data=load_cached_data_flag,
        metadata_path=Config.TRAIN_METADATA_PATH,
        patch_size=Config.PATCH_SIZE,
        stride=Config.PATCH_STRIDE,
        is_train=True,
    )

    # Create Dataset and DataLoader
    train_dataset = DenoisingDataset(train_data_packed, transform=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Validation Data (Load full images for accurate metric, but for simplicity/batching in this loop
    # we will use patches or process 1 by 1. Given the metric is global RMSE, let's use patches for validation loss
    # monitoring, but ideally we should reconstruct full images.
    # For this implementation, we'll use a validation set of patches to track convergence.)
    print("Preparing validation data...")
    val_data_packed = get_cached_data(
        cache_filename="val_patches_packed.npy",
        compute_func=_compute_patches,
        load_cached_data=load_cached_data_flag,
        metadata_path=Config.VAL_METADATA_PATH,
        patch_size=Config.PATCH_SIZE,
        stride=Config.PATCH_STRIDE,  # Same stride or larger? Same for consistency.
        is_train=True,
    )
    val_dataset = DenoisingDataset(val_data_packed, transform=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # --- Model Setup ---
    model = DRDN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Calculate actual noise (Ground Truth for the residual)
            # Noise = Input - Clean Target
            noise_target = inputs - targets

            optimizer.zero_grad()
            noise_pred = model(inputs)

            loss = criterion(noise_pred, noise_target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_mse_accum = 0.0
        total_val_pixels = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                noise_pred = model(inputs)

                # Reconstruct Clean Image: Input - Predicted Noise
                clean_pred = inputs - noise_pred

                # Clip to valid range [0, 1]
                clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

                # Calculate Squared Error for RMSE
                # (Pred - Target)^2
                se = (clean_pred - targets) ** 2
                val_mse_accum += torch.sum(se).item()
                total_val_pixels += torch.numel(targets)

        val_rmse = float(np.sqrt(val_mse_accum / total_val_pixels))

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss} - Val RMSE: {val_rmse}"
        )

        # Scheduler step
        scheduler.step(val_rmse)

        # Checkpoint & Early Stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(model, optimizer, epoch, val_rmse, Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val RMSE: {best_val_rmse}")


# -------------------------------------------------------------------------
# Inference & Submission
# -------------------------------------------------------------------------


def predict_and_submit():
    set_seed(Config.SEED)
    Config.create_directories()
    device = torch.device(Config.DEVICE)

    # Load Model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model file not found at {Config.MODEL_SAVE_PATH}. Cannot predict.")
        return

    model = DRDN().to(device)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    submission_rows = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for _, row in df_test.iterrows():
            image_id_full = row["image_id"]
            image_id = os.path.splitext(image_id_full)[0]  # e.g., "110"
            input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

            # Load full image
            img_in = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img_in is None:
                continue

            h, w = img_in.shape

            # Normalize
            img_in_norm = normalize_image(img_in)

            # Convert to tensor: (1, 1, H, W)
            input_tensor = (
                torch.from_numpy(img_in_norm).unsqueeze(0).unsqueeze(0).to(device)
            )

            # Predict Noise
            # Since the model is fully convolutional, it can handle arbitrary sizes
            # (memory permitting). Test images are ~540x420, which fits easily on GPU.
            noise_pred = model(input_tensor)

            # Reconstruct: Clean = Input - Noise
            clean_pred = input_tensor - noise_pred
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Move to CPU and numpy
            clean_pred_np = clean_pred.squeeze().cpu().numpy()  # (H, W)

            # Denormalize?
            # Submission requires: "Intensity values range from 0 (black) to 1 (white)."
            # So we keep it in [0, 1]. We do NOT denormalize to 0-255.
            # However, usually pixel submissions are integers 0-255 unless specified.
            # Task Description: "Intensity values range from 0 (black) to 1 (white)."
            # Sample Submission: value is '1'. The column type is int64.
            # Wait. "value (int64) has 1 unique values: [1]". This is just the sample.
            # Usually grayscale is 0-255. 0-1 implies float.
            # "Metric: Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities."
            # Grayscale intensities are usually 0-255.
            # If I submit floats, and the metric expects 0-255, I will fail.
            # If I submit 0-255, and metric expects 0-1, I will fail.
            # Let's check "Dataset Information": "value (int64)".
            # This strongly suggests the submission expects integers.
            # Also "Intensity values range from 0 (black) to 1 (white)" in description is contradictory to "int64".
            # Usually "0 to 1" means float. But if column is int64...
            # Maybe it means binary? "scanned text". Text is binary-ish.
            # But "grayscale pixel intensities" implies continuous.
            # Let's look at the "Metric" again. "actual grayscale pixel intensities".
            # EDA Output: "Target Type: Image (Pixel Intensities 0-255)".
            # EDA Output: "Global Mean Intensity: 226.8392".
            # The target is 0-255.
            # Therefore, the submission should likely be 0-255.
            # BUT the description says "Intensity values range from 0 (black) to 1 (white)".
            # This is a conflict.
            # However, the sample submission has `value` as `1`.
            # If I have to guess, standard image tasks are 0-255.
            # But if the prompt explicitly says "0 (black) to 1 (white)", maybe I should scale?
            # But the CSV format is `int64`. You can't put 0.5 in int64.
            # If it's 0 or 1, it's binarized.
            # "clean the noise". Scanned text is often binarized.
            # But the metric is RMSE on "grayscale pixel intensities".
            # If I binarize, RMSE against a grayscale ground truth might be high.
            # Let's assume the description "0 to 1" is a mistake or refers to the concept, and the file format (int64) and EDA (0-255) are the truth.
            # Wait, if I submit 0 or 1 (int), and the ground truth is 0-255, the error will be huge.
            # Maybe the ground truth is also 0 and 1?
            # EDA: "Global Mean Intensity: 226.8392". This is definitely 0-255 scale.
            # If I submit 0/1, RMSE will be ~226.
            # So I MUST submit 0-255.
            # Why does description say 0 to 1? Maybe it means 0=Black, 1=White (conceptually) but represented as 0-255?
            # Or maybe the submission file expects 0 and 1 (binary mask)?
            # "remove the noise".
            # Let's look at the sample submission again. `110_1_1, 1`.
            # If I look at the `denormalize_image` function in `utils.py`: it returns `uint8` (0-255).
            # I will use `denormalize_image` to convert my 0-1 float prediction to 0-255 integer.
            # This satisfies `int64` and the EDA stats.

            clean_pred_uint8 = denormalize_image(clean_pred_np)

            # Melt to rows
            # Format: id={image_id}_{row}_{col}, value={intensity}
            # Rows/Cols are 1-based.
            for r in range(h):
                for c in range(w):
                    pixel_id = f"{image_id}_{r+1}_{c+1}"
                    val = clean_pred_uint8[r, c]
                    submission_rows.append({"id": pixel_id, "value": val})

    # Create DataFrame
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
