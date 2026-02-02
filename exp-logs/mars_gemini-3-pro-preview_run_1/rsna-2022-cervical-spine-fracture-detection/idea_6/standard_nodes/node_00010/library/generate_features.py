import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

from library import config, models, utils, datasets

# ====================================================
# DATASET FOR INFERENCE
# ====================================================


class InferenceSliceDataset(Dataset):
    """
    Dataset to load all slices of a single study for inference.
    Used for Stage 1 U-Net inference.
    """

    def __init__(self, dicom_files):
        self.dicom_files = dicom_files  # List of (slice_num, z_pos, path)

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        _, _, path = self.dicom_files[idx]

        # Load and Window
        img = utils.load_dicom_array(path)
        img = utils.apply_windowing(
            img, config.BONE_WINDOW_CENTER, config.BONE_WINDOW_WIDTH
        )

        # Resize to Full Size (512x512) if not already
        if (
            img.shape[0] != config.FULL_IMAGE_SIZE
            or img.shape[1] != config.FULL_IMAGE_SIZE
        ):
            img = cv2.resize(
                img,
                (config.FULL_IMAGE_SIZE, config.FULL_IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

        # To Tensor (C, H, W) -> (3, H, W) for U-Net backbone
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        img_tensor = img_tensor.repeat(3, 1, 1)

        return img_tensor


class InferenceCropDataset(Dataset):
    """
    Dataset to generate crops for Stage 2 Encoder inference.
    """

    def __init__(self, images, masks, centers):
        """
        images: List or Array of full size images (N, H, W)
        masks: List or Array of binary masks (N, H, W)
        centers: List of (y, x) tuples for cropping
        """
        self.images = images
        self.masks = masks
        self.centers = centers
        self.crop_size = config.CROP_IMAGE_SIZE
        self.full_size = config.FULL_IMAGE_SIZE

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 3-slice window: t-1, t, t+1
        # Handle boundary conditions by padding with zeros (or duplicating)
        channels = []
        for offset in [-1, 0, 1]:
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(self.images):
                channels.append(self.images[neighbor_idx])
            else:
                # Zero padding
                channels.append(
                    np.zeros((self.full_size, self.full_size), dtype=np.float32)
                )

        # 4th Channel: Mask from current slice
        channels.append(self.masks[idx])

        # Stack: (H, W, 4)
        combined = np.stack(channels, axis=-1)

        # Crop
        center = self.centers[idx]
        cropped = utils.crop_to_roi(combined, center, self.crop_size)

        # To Tensor: (4, H, W)
        tensor = torch.from_numpy(cropped).permute(2, 0, 1).float()

        # Normalize (0.5, 0.5) roughly
        tensor = (tensor - 0.5) / 0.5

        return tensor


# ====================================================
# MODEL LOADING
# ====================================================


def load_models(device):
    """
    Loads Stage 1 and Stage 2 models from checkpoints.
    """
    # Stage 1: U-Net
    unet = models.UNetLocalizer(
        pretrained=False
    )  # Pretrained weights not needed for loading state_dict
    unet_path = os.path.join(config.CHECKPOINT_DIR, "stage1_unet.pth")
    if os.path.exists(unet_path):
        try:
            unet.load_state_dict(torch.load(unet_path, map_location=device))
            print(f"Loaded Stage 1 checkpoint: {unet_path}")
        except Exception as e:
            print(f"Failed to load Stage 1 checkpoint: {e}. Using random init.")
    else:
        print("Stage 1 checkpoint not found. Using random initialization.")
    unet.to(device)
    unet.eval()

    # Stage 2: Encoder
    # Note: We need just the encoder part, but the checkpoint might be the wrapper or just encoder.
    # train_encoder.py saves `model.encoder.state_dict()`.
    encoder = models.MaskedCNNEncoder(pretrained=False)
    enc_path = os.path.join(config.CHECKPOINT_DIR, "stage2_encoder.pth")
    if os.path.exists(enc_path):
        try:
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
            print(f"Loaded Stage 2 checkpoint: {enc_path}")
        except Exception as e:
            print(f"Failed to load Stage 2 checkpoint: {e}. Using random init.")
    else:
        print("Stage 2 checkpoint not found. Using random initialization.")
    encoder.to(device)
    encoder.eval()

    return unet, encoder


# ====================================================
# PROCESSING LOGIC
# ====================================================


def process_study(study_uid, image_path, unet, encoder, device):
    """
    Process a single study:
    1. Load DICOMs
    2. Stage 1 Inference -> Masks, ROIs, Anat IDs
    3. Stage 2 Inference -> Visual Features
    4. Combine -> (Seq, 1287)
    """
    # 1. Load DICOMs
    full_path = os.path.join(config.INPUT_DIR, image_path)
    dicom_files = datasets.load_and_sort_dicoms(full_path)

    if not dicom_files:
        # Return dummy if empty
        return np.zeros((10, 1287), dtype=np.float32)

    # Prepare Dataset for Stage 1
    ds_stage1 = InferenceSliceDataset(dicom_files)
    dl_stage1 = DataLoader(
        ds_stage1,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Containers
    all_masks = []  # Binary masks
    all_anat_ids = []  # One-hot vectors
    all_centers = []  # (y, x) centers

    # Keep raw images in memory for Stage 2 cropping (numpy array)
    # Re-loading them in Stage 2 dataset is safer for memory if huge, but slower.
    # Given 220GB RAM, we can cache the windowed images.
    cached_images = []

    # 2. Stage 1 Inference
    with torch.no_grad():
        for batch_imgs in dl_stage1:
            batch_imgs = batch_imgs.to(device)

            # Store images for Stage 2
            # batch_imgs is (B, 3, H, W), take channel 0
            imgs_np = batch_imgs[:, 0, :, :].cpu().numpy()
            for i in range(imgs_np.shape[0]):
                cached_images.append(imgs_np[i])

            # Inference
            logits = unet(batch_imgs)  # (B, 8, H, W)
            probs = F.softmax(logits, dim=1)  # (B, 8, H, W)

            # Process batch
            probs_np = probs.cpu().numpy()

            for i in range(probs_np.shape[0]):
                p = probs_np[i]  # (8, H, W)

                # A. Anatomical ID
                # Check if class k is present (threshold sum or max)
                # Simple heuristic: if sum of prob map for class k > threshold
                # or max prob > 0.5
                anat_id = np.zeros(config.NUM_VERTEBRAE, dtype=np.float32)
                for k in range(config.NUM_VERTEBRAE):
                    # Class k+1 corresponds to C1..C7
                    # Check max probability pixel
                    if np.max(p[k + 1]) > 0.5:
                        anat_id[k] = 1.0
                all_anat_ids.append(anat_id)

                # B. Binary Mask (Bone vs Bg)
                # Sum probs of 1-7
                bone_prob = np.sum(p[1:], axis=0)
                binary_mask = (bone_prob > 0.5).astype(np.float32)
                all_masks.append(binary_mask)

                # C. ROI Center
                # Center of mass of the binary mask
                ys, xs = np.where(binary_mask > 0.5)
                if len(ys) > 0:
                    cy = np.mean(ys)
                    cx = np.mean(xs)
                else:
                    # Fallback to image center
                    cy, cx = config.FULL_IMAGE_SIZE // 2, config.FULL_IMAGE_SIZE // 2
                all_centers.append((cy, cx))

    # 3. Stage 2 Inference
    ds_stage2 = InferenceCropDataset(cached_images, all_masks, all_centers)
    dl_stage2 = DataLoader(
        ds_stage2,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    all_features = []

    with torch.no_grad():
        for batch_crops in dl_stage2:
            batch_crops = batch_crops.to(device)  # (B, 4, 256, 256)

            feats = encoder(batch_crops)  # (B, 1280)
            all_features.append(feats.cpu().numpy())

    if all_features:
        all_features = np.concatenate(all_features, axis=0)
    else:
        all_features = np.zeros((0, 1280), dtype=np.float32)

    all_anat_ids = np.array(all_anat_ids)  # (N, 7)

    # 4. Concatenate
    # Result: (N, 1280 + 7)
    if len(all_features) > 0:
        final_seq = np.concatenate([all_features, all_anat_ids], axis=1)
    else:
        final_seq = np.zeros((0, 1287), dtype=np.float32)

    return final_seq


def generate_features(load_cached_data=True):
    """
    Main driver to generate features for Train, Val, and Test sets.
    """
    print("Starting Feature Generation Pipeline...")

    # Ensure output directory exists
    feature_dir = os.path.join(config.CACHE_DIR, "features")
    os.makedirs(feature_dir, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Combine unique studies
    # We process by study UID.
    # Train and Val are disjoint. Test is separate.
    # Create a list of (UID, image_path) tuples
    studies_to_process = []

    for df in [train_df, val_df, test_df]:
        # Drop duplicates just in case (though metadata generation handles this)
        unique_studies = df[["StudyInstanceUID", "image_path"]].drop_duplicates()
        for _, row in unique_studies.iterrows():
            studies_to_process.append((row["StudyInstanceUID"], row["image_path"]))

    # Remove duplicates across sets if any (shouldn't be)
    studies_to_process = list(set(studies_to_process))

    print(f"Total studies to process: {len(studies_to_process)}")

    # Load Models
    unet, encoder = load_models(config.DEVICE)

    # Processing Loop
    processed_count = 0
    skipped_count = 0

    for uid, img_path in studies_to_process:
        save_path = os.path.join(feature_dir, f"{uid}.npy")

        # Check Cache
        if load_cached_data and os.path.exists(save_path):
            skipped_count += 1
            continue

        try:
            # Process
            features = process_study(uid, img_path, unet, encoder, config.DEVICE)

            # Save
            np.save(save_path, features)
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"Processed {processed_count} studies...")

        except Exception as e:
            print(f"Error processing study {uid}: {e}")
            # Save dummy to prevent pipeline failure? Better to skip.
            continue

    print(f"Feature Generation Completed.")
    print(f"Processed: {processed_count}")
    print(f"Skipped (Cached): {skipped_count}")
