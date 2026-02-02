import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            # Check for shape mismatch (padding issues)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class InkDetector(nn.Module):
    """
    U-Net with ResNet34 Encoder for Vesuvius Ink Detection.
    Expects 3-channel input (Stratified Depth Projections).
    """

    def __init__(self):
        super().__init__()

        # --- Encoder (ResNet34) ---
        # Load pre-trained weights
        weights = (
            ResNet34_Weights.IMAGENET1K_V1
            if Config.ENCODER_WEIGHTS == "imagenet"
            else None
        )
        self.encoder = resnet34(weights=weights)

        # Extract layers for skip connections
        # Input: (B, 3, H, W)
        # Stem: Conv1 -> BN -> ReLU
        self.enc0 = nn.Sequential(
            self.encoder.conv1, self.encoder.bn1, self.encoder.relu
        )  # -> (64, H/2, W/2)
        self.pool = self.encoder.maxpool  # -> (64, H/4, W/4)
        self.enc1 = self.encoder.layer1  # -> (64, H/4, W/4)
        self.enc2 = self.encoder.layer2  # -> (128, H/8, W/8)
        self.enc3 = self.encoder.layer3  # -> (256, H/16, W/16)
        self.enc4 = self.encoder.layer4  # -> (512, H/32, W/32)

        # --- Decoder ---
        # Channels: 512 -> 256 -> 128 -> 64 -> 32 -> 16

        # Block 4: 512 (enc4) + 256 (enc3) -> 256
        self.dec4 = DecoderBlock(512, 256, 256)

        # Block 3: 256 (dec4) + 128 (enc2) -> 128
        self.dec3 = DecoderBlock(256, 128, 128)

        # Block 2: 128 (dec3) + 64 (enc1) -> 64
        self.dec2 = DecoderBlock(128, 64, 64)

        # Block 1: 64 (dec2) + 64 (enc0) -> 32
        self.dec1 = DecoderBlock(64, 64, 32)

        # Block 0: 32 (dec1) + 0 (no skip) -> 16
        self.dec0 = DecoderBlock(32, 0, 16)

        # Final Convolution
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        p0 = self.pool(e0)
        e1 = self.enc1(p0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1)  # Upsample to original resolution

        logits = self.final_conv(d0)
        return logits


class InkDataset(Dataset):
    """
    Dataset for Vesuvius Ink Detection.
    Handles Stratified Depth Projection and Caching.
    """

    def __init__(self, dataframe, mode="train", load_cached_data=True):
        self.df = dataframe
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.fragment_cache = {}

        # Ensure cache directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Pre-load/Cache fragments
        # We iterate over unique fragments to ensure the heavy 3D volume processing
        # happens only once per fragment.
        unique_fragments = self.df["fragment_id"].unique()
        for frag_id in unique_fragments:
            # Find volume path for this fragment
            row = self.df[self.df["fragment_id"] == frag_id].iloc[0]
            vol_path = os.path.join(Config.INPUT_DIR, row["volume_path"])

            # Load or Compute MIP
            self.fragment_cache[frag_id] = self._load_fragment(frag_id, vol_path)

    def _load_fragment(self, frag_id, vol_path):
        """
        Loads the fragment volume, computes stratified MIPs, and caches to disk.
        """
        cache_filename = f"fragment_{frag_id}_mip_3ch.npy"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to recompute if corrupt

        # Compute Stratified Projection
        slices = []
        for z in range(Config.Z_START, Config.Z_END):
            slice_path = os.path.join(vol_path, f"{z:02d}.tif")
            if os.path.exists(slice_path):
                # Load as grayscale
                img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    slices.append(img)
            else:
                pass

        if not slices:
            # If no slices found, create a placeholder (should not happen in valid data)
            # Assuming standard size based on metadata or error out
            raise ValueError(
                f"No slices found for fragment {frag_id} in range {Config.Z_START}-{Config.Z_END}"
            )

        volume = np.stack(slices, axis=0)  # (D, H, W)

        # Stratified MIP: Split D into 3 chunks
        D = volume.shape[0]
        chunk_size = max(1, D // Config.NUM_SUB_VOLUMES)

        mips = []
        for i in range(Config.NUM_SUB_VOLUMES):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < Config.NUM_SUB_VOLUMES - 1 else D

            # Handle case where chunk is empty
            if start >= D:
                mip = np.zeros_like(volume[0])
            else:
                chunk = volume[start:end, :, :]
                if chunk.shape[0] > 0:
                    mip = np.max(chunk, axis=0)
                else:
                    mip = np.zeros_like(volume[0])
            mips.append(mip)

        # Stack to (H, W, 3)
        stratified_mip = np.stack(mips, axis=-1)

        # Save to cache
        np.save(cache_path, stratified_mip)

        return stratified_mip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Retrieve full fragment image from memory cache
        full_image = self.fragment_cache[frag_id]  # (H_frag, W_frag, 3)

        # Crop
        img_h, img_w = full_image.shape[:2]
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        image = full_image[y:y_end, x:x_end, :]

        # Pad if necessary (e.g., at edges)
        pad_h = h - image.shape[0]
        pad_w = w - image.shape[1]

        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        # Load Mask and Label if available (Train/Val)
        mask = None
        label = None

        if self.mode in ["train", "validation"]:
            # Load mask
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_patch = mask_full[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                mask_patch = np.pad(
                    mask_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                )
            mask = mask_patch

            # Load label
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            label_full = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label_patch = label_full[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                label_patch = np.pad(
                    label_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                )
            label = label_patch

            # Normalization (0-1) for 16-bit input
            image = image.astype(np.float32) / 65535.0
            mask = (mask > 0).astype(np.float32)
            label = (label > 0).astype(np.float32)

            # Augmentation (Train only)
            if self.mode == "train":
                # Random Vertical Flip
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=0)
                    mask = np.flip(mask, axis=0)
                    label = np.flip(label, axis=0)
                # Random Horizontal Flip
                if np.random.rand() < 0.5:
                    image = np.flip(image, axis=1)
                    mask = np.flip(mask, axis=1)
                    label = np.flip(label, axis=1)
                # Random Rotate 90
                k = np.random.randint(0, 4)
                if k > 0:
                    image = np.rot90(image, k)
                    mask = np.rot90(mask, k)
                    label = np.rot90(label, k)

            # Convert to Tensor (C, H, W)
            # Image is (H, W, 3) -> (3, H, W)
            image = torch.from_numpy(image.copy()).permute(2, 0, 1)
            mask = torch.from_numpy(mask.copy()).unsqueeze(0)
            label = torch.from_numpy(label.copy()).unsqueeze(0)

            return image, label, mask

        else:
            # Inference mode (if metadata provided without labels)
            image = image.astype(np.float32) / 65535.0
            image = torch.from_numpy(image.copy()).permute(2, 0, 1)
            return image
