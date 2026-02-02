import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, fbeta_score, rle_encode
from library.architecture import MIPUNet
from library.dataset import InkDataset, get_transforms, get_fragment_mip


class BCEDiceLoss(nn.Module):
    """
    Composite loss function: Binary Cross Entropy + Dice Loss.
    """

    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # BCE Loss
        bce_loss = self.bce(preds, targets)

        # Dice Loss
        preds_prob = torch.sigmoid(preds)
        preds_flat = preds_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        # Add epsilon to avoid division by zero
        dice_score = (2.0 * intersection) / (union + 1e-7)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and average F0.5 score.
    """
    model.eval()
    running_loss = 0.0
    running_score = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities for metric calculation
            probs = torch.sigmoid(outputs)
            score = fbeta_score(probs, masks, beta=0.5)
            running_score += score

    return running_loss / len(loader), running_score / len(loader)


def train_model(load_cached_data=True):
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --- Data Preparation ---
    train_dataset = InkDataset(
        Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        load_cached_data=load_cached_data,
    )
    val_dataset = InkDataset(
        Config.VALID_METADATA_PATH,
        transform=get_transforms("valid"),
        load_cached_data=load_cached_data,
    )

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

    # --- Model Setup ---
    model = MIPUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = BCEDiceLoss()

    # --- Training Loop ---
    best_score = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F0.5: {val_score:.6f}"
        )

        # Step scheduler based on validation score
        scheduler.step(val_score)

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F0.5 score: {best_score:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best F0.5 Score: {best_score:.6f}")
    return best_model_path


def inference(model_path, load_cached_data=True):
    """
    Runs inference on the test set and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --- Load Model ---
    # We use encoder_weights=None because we are loading a state_dict
    model = MIPUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Load Test Metadata ---
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print("Test metadata not found. Skipping inference.")
        return

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    submission_data = []

    transforms = get_transforms("test")

    print(f"Starting inference on {len(test_df)} fragments...")

    for _, row in test_df.iterrows():
        frag_id = row["fragment_id"]
        vol_path = row["volume_path"]
        mask_path = row["mask_path"]

        # 1. Load Data
        # Load MIP (Cached or Computed)
        mip = get_fragment_mip(frag_id, vol_path, load_cached_data=load_cached_data)

        # Load Mask (to define valid area)
        mask_full_path = os.path.join(Config.INPUT_DIR, mask_path)
        mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = (mask_img > 0).astype(bool)

        # 2. Preprocess
        # Normalize to [0, 1] float32 (same as training)
        mip = mip.astype(np.float32) / 65535.0

        # Pad image to be divisible by TILE_SIZE
        h, w = mip.shape[:2]
        pad_h = (Config.TILE_SIZE - (h % Config.TILE_SIZE)) % Config.TILE_SIZE
        pad_w = (Config.TILE_SIZE - (w % Config.TILE_SIZE)) % Config.TILE_SIZE

        mip_padded = np.pad(
            mip, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

        # 3. Tiled Inference
        preds_padded = np.zeros_like(mip_padded)

        patches = []
        coords = []

        # Iterate over tiles
        for y in range(0, mip_padded.shape[0], Config.TILE_SIZE):
            for x in range(0, mip_padded.shape[1], Config.TILE_SIZE):
                patch = mip_padded[y : y + Config.TILE_SIZE, x : x + Config.TILE_SIZE]

                # Apply transforms (Normalize + ToTensor)
                augmented = transforms(image=patch)
                patch_tensor = augmented["image"]

                patches.append(patch_tensor)
                coords.append((y, x))

                # Run batch if full
                if len(patches) >= Config.BATCH_SIZE:
                    batch_tensor = torch.stack(patches).to(device)
                    with torch.no_grad():
                        outputs = model(batch_tensor)
                        probs = torch.sigmoid(outputs).cpu().numpy()

                    # Place back into canvas
                    for i, (py, px) in enumerate(coords):
                        preds_padded[
                            py : py + Config.TILE_SIZE, px : px + Config.TILE_SIZE
                        ] = probs[i, 0]

                    patches = []
                    coords = []

        # Process remaining patches
        if patches:
            batch_tensor = torch.stack(patches).to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()

            for i, (py, px) in enumerate(coords):
                preds_padded[py : py + Config.TILE_SIZE, px : px + Config.TILE_SIZE] = (
                    probs[i, 0]
                )

        # 4. Post-process
        # Crop back to original size
        preds = preds_padded[:h, :w]

        # Apply valid mask (zero out predictions outside the fragment)
        preds = preds * valid_mask

        # Threshold to binary
        binary_preds = (preds > 0.5).astype(np.uint8)

        # RLE Encode
        rle = rle_encode(binary_preds)
        submission_data.append({"Id": frag_id, "Predicted": rle})

    # --- Save Submission ---
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
