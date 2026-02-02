import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    compute_map_batch,
    AverageMeter,
)
from library.losses import ConsistentCompoundLoss
from library.model_components import SaltUNet

# Alias SaltUNet to DeepResUNet as per task description
DeepResUNet = SaltUNet


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.
    Reads metadata from CSVs and loads images/masks from disk.
    """

    def __init__(self, mode="train", transform=None, debug=False):
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Load metadata based on mode
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if debug:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)

        self.ids = self.df["id"].values
        self.images = self.df["image_path"].values
        self.depths = self.df["z"].values

        # Masks are only available for train/val
        if mode != "test":
            self.masks = self.df["mask_path"].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Load Image
        img_path = os.path.join(self.input_dir, self.images[idx])
        # Load as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for safety
            image = np.zeros(
                (Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE), dtype=np.uint8
            )

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Load Depth
        depth = self.depths[idx]

        # Load Mask if available
        mask = None
        if self.mode != "test":
            mask_path = os.path.join(self.input_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(
                    (Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE), dtype=np.uint8
                )
            # Normalize and binarize
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # Prepare data dict for albumentations
        data = {"image": image}
        if mask is not None:
            data["mask"] = mask

        # Apply transforms (Padding, Flipping, etc.)
        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if mask is not None:
                mask = augmented["mask"]

        # Convert to Tensor (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask, depth, self.ids[idx]
        else:
            return image, depth, self.ids[idx]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    transforms = []

    # Pad 101x101 -> 128x128 using Reflection Padding
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )
    )

    # Augmentations for training
    if mode == "train":
        if Config.TTA_FLIP:
            transforms.append(A.HorizontalFlip(p=0.5))

    return A.Compose(transforms)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for images, masks, depths, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns list [logits, aux1, aux2] if deep supervision is on
        outputs = model(images, depths)

        if isinstance(outputs, list):
            logits = outputs[0]
            loss = criterion(logits, masks)

            # Add auxiliary losses with lower weight (0.5)
            # We interpolate aux outputs to match mask size
            for aux in outputs[1:]:
                aux_upsampled = F.interpolate(
                    aux, size=masks.shape[2:], mode="bilinear", align_corners=True
                )
                loss += 0.5 * criterion(aux_upsampled, masks)
        else:
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Runs validation and calculates mean Average Precision (mAP).
    """
    model.eval()
    map_score = AverageMeter()

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            masks = masks.cpu().numpy()  # Keep ground truth on CPU for metric calc
            depths = depths.to(device)

            outputs = model(images, depths)

            # In eval mode, model might still return list if we didn't toggle deep supervision flag,
            # but SaltUNet returns logits directly if not (self.training and self.deep_supervision).
            # We handle both just in case.
            if isinstance(outputs, list):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs)

            # Center Crop back to 101x101 to match Ground Truth
            h, w = probs.shape[2], probs.shape[3]
            orig_h, orig_w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE
            start_h = (h - orig_h) // 2
            start_w = (w - orig_w) // 2

            probs = probs[:, :, start_h : start_h + orig_h, start_w : start_w + orig_w]

            # Binarize predictions
            preds = (probs > 0.5).float().cpu().numpy()

            # Compute mAP
            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            batch_map = compute_map_batch(preds.squeeze(1), masks.squeeze(1))
            map_score.update(batch_map, images.size(0))

    return map_score.avg


def main():
    """
    Main execution pipeline.
    """
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    train_dataset = SaltDataset(
        mode="train", transform=get_transforms("train"), debug=Config.DEBUG
    )
    val_dataset = SaltDataset(
        mode="val", transform=get_transforms("val"), debug=Config.DEBUG
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 2. Model Initialization
    # -------------------------------------------------------------------------
    model = DeepResUNet().to(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 3. Optimization Setup
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS_PER_CYCLE, T_mult=1, eta_min=1e-6
    )

    criterion = ConsistentCompoundLoss().to(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_map = 0.0
    cycle_best_map = 0.0

    print(f"Starting training: {Config.EPOCHS} epochs, {Config.CYCLES} cycles.")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_map = validate(model, val_loader, Config.DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val mAP: {val_map:.5f}"
        )

        # Track best overall model
        if val_map > best_map:
            best_map = val_map
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

        # Track best model per cycle for Snapshot Ensembling
        # Reset cycle tracker at start of new cycle
        if (epoch - 1) % Config.EPOCHS_PER_CYCLE == 0:
            cycle_best_map = 0.0

        if val_map > cycle_best_map:
            cycle_best_map = val_map
            current_cycle = (epoch - 1) // Config.EPOCHS_PER_CYCLE + 1
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{current_cycle}.pth"),
            )

    # -------------------------------------------------------------------------
    # 5. Inference & Submission
    # -------------------------------------------------------------------------
    print("Training complete. Generating submission...")

    test_dataset = SaltDataset(
        mode="test", transform=get_transforms("val"), debug=Config.DEBUG
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Snapshot Ensemble Models
    models = []
    for c in Config.ENSEMBLE_CYCLES:
        path = os.path.join(Config.CHECKPOINT_DIR, f"best_cycle_{c}.pth")
        if os.path.exists(path):
            print(f"Loading ensemble checkpoint: {path}")
            m = DeepResUNet().to(Config.DEVICE)
            m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Checkpoint {path} missing.")

    if not models:
        print("No cycle checkpoints found. Loading best overall model.")
        path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        m = DeepResUNet().to(Config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        m.eval()
        models.append(m)

    submission_data = []

    with torch.no_grad():
        for images, depths, ids in tqdm(test_loader, desc="Inference"):
            images = images.to(Config.DEVICE)
            depths = depths.to(Config.DEVICE)

            # Accumulate predictions
            avg_probs = torch.zeros(
                (images.size(0), 1, Config.IMG_SIZE, Config.IMG_SIZE),
                device=Config.DEVICE,
            )

            for m in models:
                # Standard Prediction
                out = m(images, depths)
                if isinstance(out, list):
                    out = out[0]
                probs = torch.sigmoid(out)

                # Test Time Augmentation (Horizontal Flip)
                if Config.TTA_FLIP:
                    images_flipped = torch.flip(images, [3])
                    out_flipped = m(images_flipped, depths)
                    if isinstance(out_flipped, list):
                        out_flipped = out_flipped[0]
                    probs_flipped = torch.sigmoid(out_flipped)
                    probs_flipped = torch.flip(probs_flipped, [3])

                    probs = (probs + probs_flipped) / 2.0

                avg_probs += probs

            avg_probs /= len(models)

            # Crop back to 101x101
            h, w = avg_probs.shape[2], avg_probs.shape[3]
            orig_h, orig_w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE
            start_h = (h - orig_h) // 2
            start_w = (w - orig_w) // 2

            avg_probs = avg_probs[
                :, :, start_h : start_h + orig_h, start_w : start_w + orig_w
            ]

            # Threshold
            preds = (avg_probs > 0.5).cpu().numpy()

            # Encode
            for i in range(len(ids)):
                pred_mask = preds[i, 0]
                rle = rle_encode(pred_mask)
                submission_data.append([ids[i], rle])

    # Save to CSV
    sub_df = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# Execute main function
main()
