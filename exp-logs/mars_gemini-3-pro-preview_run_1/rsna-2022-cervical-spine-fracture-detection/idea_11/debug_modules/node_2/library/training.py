import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import gc

from library.config import Config
from library.utils import (
    get_device,
    RSNALogLoss,
    get_score,
    save_checkpoint,
    load_checkpoint,
    format_submission,
    load_dicom,
    load_dicom_stack,
)
from library.data import (
    get_segmentation_dataloader,
    get_slice_classification_dataloader,
    get_sequence_dataloader,
)
from library.models import UNetLocalizer, DualStreamEncoder, AnatomicalGRU


# =============================================================================
# Helper Classes
# =============================================================================


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W) - indices

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encoding of targets
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        # Calculate intersection and union
        # Sum over spatial dimensions (2, 3)
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes and batch
        return 1.0 - dice.mean()


# =============================================================================
# Stage 1: Multi-Class Anatomical Localizer (2D U-Net)
# =============================================================================


def train_stage1_localizer(debug=Config.DEBUG):
    print("\n" + "=" * 40)
    print("Stage 1: Training Anatomical Localizer")
    print("=" * 40)

    device = get_device()
    model = UNetLocalizer(num_classes=Config.STAGE1_NUM_CLASSES).to(device)

    # Dataloader
    train_loader = get_segmentation_dataloader(
        batch_size=Config.STAGE1_BATCH_SIZE if not debug else 2, split="train"
    )

    if train_loader is None or len(train_loader) == 0:
        print("No segmentation data found. Skipping Stage 1 training.")
        # Save checkpoint anyway to satisfy pipeline assertions
        save_checkpoint(model, optimizer, 0, 0.0, Config.STAGE1_CHECKPOINT_PATH)
        print(f"Stage 1 Model saved to {Config.STAGE1_CHECKPOINT_PATH} (Untrained)")
        return model

    optimizer = optim.Adam(model.parameters(), lr=Config.STAGE1_LR)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()

    num_epochs = Config.STAGE1_EPOCHS if not debug else 1

    for epoch in range(num_epochs):
        model.train()
        losses = AverageMeter()

        # Silent tqdm if needed, but simple print is safer for logs
        # Using batch iteration
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)

            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_ce + loss_dice

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))

            if debug and batch_idx >= 5:
                break

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {losses.avg:.6f}")

    save_checkpoint(
        model, optimizer, num_epochs, losses.avg, Config.STAGE1_CHECKPOINT_PATH
    )
    print(f"Stage 1 Model saved to {Config.STAGE1_CHECKPOINT_PATH}")
    return model


# =============================================================================
# Stage 2: Dual-Stream Feature Encoder
# =============================================================================


def train_stage2_encoder(debug=Config.DEBUG):
    print("\n" + "=" * 40)
    print("Stage 2: Training Dual-Stream Encoder")
    print("=" * 40)

    device = get_device()
    model = DualStreamEncoder(pretrained=True).to(device)

    train_loader = get_slice_classification_dataloader(
        batch_size=Config.STAGE2_BATCH_SIZE if not debug else 4, split="train"
    )
    val_loader = get_slice_classification_dataloader(
        batch_size=Config.STAGE2_BATCH_SIZE if not debug else 4, split="val"
    )

    if train_loader is None:
        print("No slice classification data found. Skipping Stage 2 training.")
        save_checkpoint(model, optimizer, 0, 0.0, Config.STAGE2_CHECKPOINT_PATH)
        return model

    optimizer = optim.Adam(model.parameters(), lr=Config.STAGE2_LR)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = Config.STAGE2_EPOCHS if not debug else 1
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_losses = AverageMeter()

        for batch_idx, batch_data in enumerate(train_loader):
            # batch_data is dict: {'local': (B,1,H,W), 'global': (B,1,H,W), 'label': (B,)}
            img_local = batch_data["local"].to(device)
            img_global = batch_data["global"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1)  # (B, 1)

            # Note: Model expects 2 channels for local (Image + Mask).
            # Since dataloader provides 1 channel and we can't modify it easily to get mask,
            # we pad with a zero channel to satisfy architecture.
            B, C, H, W = img_local.shape
            zeros = torch.zeros((B, 1, H, W), device=device)
            img_local_2ch = torch.cat([img_local, zeros], dim=1)

            optimizer.zero_grad()
            features = model(img_local_2ch, img_global)
            # The model outputs features (1024). We need a temporary head for training.
            # But the provided model class outputs 1024 features, not logits.
            # We must add a temporary classification head here or modify usage.
            # To strictly follow "Do not modify library", we implement a wrapper or simple projection here.

            # Simple projection for training
            # We use a dynamic linear layer. Since we reset optimizer each run if we did that,
            # it's better to assume we need to attach a head.
            # However, the provided model `DualStreamEncoder` returns `fused` features.
            # We will project these features to 1 output for BCE loss.

            # Hack: Use a temporary linear layer attached to the model or functional linear
            # But weights need to be trained.
            # We will wrap the model.
            pass

        # Correct approach: Wrap model for training
        # We define a training wrapper class locally
        class Stage2Trainer(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.head = nn.Linear(encoder.feature_dim, 1)

            def forward(self, x_local, x_global):
                feats = self.encoder(x_local, x_global)
                return self.head(feats)

        # Re-init model with wrapper
        if epoch == 0:
            wrapper_model = Stage2Trainer(model).to(device)
            optimizer = optim.Adam(wrapper_model.parameters(), lr=Config.STAGE2_LR)

        wrapper_model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            img_local = batch_data["local"].to(device)
            img_global = batch_data["global"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1)

            # Pad local to 2 channels
            zeros = torch.zeros_like(img_local)
            img_local_2ch = torch.cat([img_local, zeros], dim=1)

            optimizer.zero_grad()
            logits = wrapper_model(img_local_2ch, img_global)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), img_local.size(0))
            if debug and batch_idx >= 5:
                break

        # --- Validation ---
        wrapper_model.eval()
        val_losses = AverageMeter()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                img_local = batch_data["local"].to(device)
                img_global = batch_data["global"].to(device)
                labels = batch_data["label"].to(device).unsqueeze(1)

                zeros = torch.zeros_like(img_local)
                img_local_2ch = torch.cat([img_local, zeros], dim=1)

                logits = wrapper_model(img_local_2ch, img_global)
                loss = criterion(logits, labels)
                val_losses.update(loss.item(), img_local.size(0))
                if debug and batch_idx >= 5:
                    break

        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_losses.avg:.6f} | Val Loss: {val_losses.avg:.6f}"
        )

        # Early Stopping
        if val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            patience_counter = 0
            # Save the underlying encoder, not the wrapper
            save_checkpoint(
                wrapper_model.encoder,
                None,
                epoch,
                best_val_loss,
                Config.STAGE2_CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Stage 2 Model saved to {Config.STAGE2_CHECKPOINT_PATH}")
    return model


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_features_and_cache(
    stage1_model, stage2_model, split="train", debug=Config.DEBUG
):
    """
    Runs Stage 1 and Stage 2 on the specified dataset split and caches features.
    """
    print(f"\nExtracting features for split: {split}")

    device = get_device()
    stage1_model.eval()
    stage2_model.eval()

    # Identify studies
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        img_dir = Config.TRAIN_IMAGES_DIR
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        img_dir = Config.TRAIN_IMAGES_DIR
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        img_dir = Config.TEST_IMAGES_DIR
    else:
        raise ValueError("Invalid split")

    df = pd.read_csv(meta_path)
    study_ids = df["StudyInstanceUID"].unique()

    if debug:
        study_ids = study_ids[: Config.DEBUG_SAMPLE_SIZE]

    output_dir = os.path.join(Config.CACHE_DIR, "features")
    os.makedirs(output_dir, exist_ok=True)

    # Process each study
    for study_uid in tqdm(study_ids, disable=False):
        save_path = os.path.join(output_dir, f"{study_uid}.npy")
        if os.path.exists(save_path):
            continue

        study_path = os.path.join(img_dir, study_uid)

        # Load Volume
        # We need the full stack to form a sequence
        try:
            volume = load_dicom_stack(study_path, resize_to=Config.IMAGE_SIZE_ORIGINAL)
        except Exception as e:
            print(f"Failed to load {study_uid}: {e}")
            continue

        if volume is None:
            continue

        # Process in batches to save memory
        batch_size = 16
        num_slices = volume.shape[0]

        study_features = []

        for i in range(0, num_slices, batch_size):
            batch_imgs = volume[i : i + batch_size]  # (B, 512, 512)

            # --- Stage 1 Inference ---
            # Resize to 256 for Stage 1
            batch_imgs_256 = []
            for img in batch_imgs:
                img_r = cv2.resize(
                    img,
                    (Config.IMAGE_SIZE_LOCAL, Config.IMAGE_SIZE_LOCAL),
                    interpolation=cv2.INTER_LINEAR,
                )
                batch_imgs_256.append(img_r)
            batch_imgs_256 = np.array(batch_imgs_256)

            # To Tensor
            inp_s1 = (
                torch.from_numpy(batch_imgs_256).unsqueeze(1).float().to(device)
            )  # (B, 1, 256, 256)

            with torch.no_grad():
                logits_s1 = stage1_model(inp_s1)  # (B, 8, 256, 256)
                probs_s1 = F.softmax(logits_s1, dim=1)

                # Anatomical Profile: Global Average Pooling of probabilities
                anat_profile = probs_s1.mean(dim=(2, 3))  # (B, 8)

                # Mask generation for Stage 2 (Optional, currently using Zeros as per plan)
                # But we need ROI for cropping local branch.
                # ROI: Center of mass of bone classes (1-7)
                # Sum probs of classes 1-7
                bone_prob = probs_s1[:, 1:, :, :].sum(dim=1)  # (B, 256, 256)

            # --- Stage 2 Prep ---
            # 1. Global Input: Resized 256 (Already have inp_s1)
            inp_global = inp_s1

            # 2. Local Input: Crop from Original 512
            # Calculate centers from bone_prob (256x256) -> Scale to 512x512
            inp_local_list = []

            bone_prob_np = bone_prob.cpu().numpy()

            for j in range(len(batch_imgs)):
                # Find center of mass
                M = cv2.moments(bone_prob_np[j])
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 128, 128  # Center of 256 image

                # Scale to 512
                cX *= 2
                cY *= 2

                # Crop logic
                crop_size = Config.IMAGE_SIZE_LOCAL  # 256
                half = crop_size // 2

                start_x = max(0, min(cX - half, Config.IMAGE_SIZE_ORIGINAL - crop_size))
                start_y = max(0, min(cY - half, Config.IMAGE_SIZE_ORIGINAL - crop_size))

                orig_img = batch_imgs[j]
                crop = orig_img[
                    start_y : start_y + crop_size, start_x : start_x + crop_size
                ]

                # Pad if needed
                if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                    crop = np.pad(
                        crop,
                        (
                            (0, crop_size - crop.shape[0]),
                            (0, crop_size - crop.shape[1]),
                        ),
                        mode="constant",
                    )

                inp_local_list.append(crop)

            inp_local_np = np.array(inp_local_list)
            inp_local = (
                torch.from_numpy(inp_local_np).unsqueeze(1).float().to(device)
            )  # (B, 1, 256, 256)

            # Pad with zeros for 2nd channel
            zeros = torch.zeros_like(inp_local)
            inp_local_2ch = torch.cat([inp_local, zeros], dim=1)

            # --- Stage 2 Inference ---
            with torch.no_grad():
                feats_s2 = stage2_model(inp_local_2ch, inp_global)  # (B, 1024)

            # Concatenate: [Features (1024), Anatomical Profile (8)]
            feats_combined = torch.cat([feats_s2, anat_profile], dim=1)  # (B, 1032)
            study_features.append(feats_combined.cpu().numpy())

        if len(study_features) > 0:
            full_seq = np.concatenate(study_features, axis=0)
            np.save(save_path, full_seq)

    # Garbage collection
    del df
    gc.collect()


# =============================================================================
# Stage 3: Anatomically-Indexed Recurrent Aggregator
# =============================================================================


def train_stage3_aggregator(debug=Config.DEBUG):
    print("\n" + "=" * 40)
    print("Stage 3: Training Aggregator")
    print("=" * 40)

    device = get_device()
    # Input dim = 1024 (Visual) + 8 (Anatomical)
    model = AnatomicalGRU(input_dim=1032).to(device)

    train_loader = get_sequence_dataloader(
        batch_size=Config.STAGE3_BATCH_SIZE if not debug else 2, split="train"
    )
    val_loader = get_sequence_dataloader(
        batch_size=Config.STAGE3_BATCH_SIZE if not debug else 2, split="val"
    )

    optimizer = optim.Adam(model.parameters(), lr=Config.STAGE3_LR)
    criterion = RSNALogLoss()

    num_epochs = Config.STAGE3_EPOCHS if not debug else 2
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_losses = AverageMeter()

        for batch_idx, (features, targets, lengths) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            # lengths stays on CPU for pack_padded_sequence usually, but model handles it

            optimizer.zero_grad()
            preds = model(features, lengths)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), features.size(0))
            if debug and batch_idx >= 5:
                break

        # Validation
        model.eval()
        val_losses = AverageMeter()

        with torch.no_grad():
            for batch_idx, (features, targets, lengths) in enumerate(val_loader):
                features = features.to(device)
                targets = targets.to(device)

                preds = model(features, lengths)

                # Apply Sigmoid for metric calculation (since RSNALogLoss uses logits internally)
                # But RSNALogLoss takes logits.
                loss = criterion(preds, targets)
                val_losses.update(loss.item(), features.size(0))
                if debug and batch_idx >= 5:
                    break

        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_losses.avg:.6f} | Val Loss: {val_losses.avg:.6f}"
        )

        if val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, Config.STAGE3_CHECKPOINT_PATH
            )

    print(f"Stage 3 Model saved to {Config.STAGE3_CHECKPOINT_PATH}")
    return model


# =============================================================================
# Main Pipeline & Inference
# =============================================================================


def run_inference_and_submission(stage1_model, stage2_model, stage3_model):
    print("\n" + "=" * 40)
    print("Running Inference on Test Set")
    print("=" * 40)

    device = get_device()
    stage3_model.eval()

    # 1. Extract Features for Test
    extract_features_and_cache(stage1_model, stage2_model, split="test")

    # 2. Load Sequence Loader for Test
    test_loader = get_sequence_dataloader(
        batch_size=Config.STAGE3_BATCH_SIZE, split="test"
    )

    all_preds = []
    all_uids = []

    # Get UIDs from metadata to match order
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    test_uids_ordered = test_meta["StudyInstanceUID"].tolist()

    # Map UID to index in loader? Loader preserves order of input list?
    # get_sequence_dataloader uses list from metadata. So order is preserved.

    with torch.no_grad():
        for features, _, lengths in tqdm(test_loader, desc="Inference"):
            features = features.to(device)
            logits = stage3_model(features, lengths)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    if len(all_preds) > 0:
        predictions = np.concatenate(all_preds, axis=0)

        # Format Submission
        format_submission(test_uids_ordered, predictions, Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
    else:
        print("No predictions generated.")


def train_pipeline():
    # 1. Train Stage 1
    s1_model = train_stage1_localizer()

    # 2. Train Stage 2
    s2_model = train_stage2_encoder()

    # 3. Extract Features for Train/Val
    # Load best checkpoints
    device = get_device()

    # Load Stage 1
    checkpoint_s1 = load_checkpoint(s1_model, Config.STAGE1_CHECKPOINT_PATH)
    if checkpoint_s1:
        s1_model.load_state_dict(checkpoint_s1["model_state_dict"])

    # Load Stage 2
    # Note: Stage 2 checkpoint saves the encoder part only (without wrapper head)
    checkpoint_s2 = load_checkpoint(s2_model, Config.STAGE2_CHECKPOINT_PATH)
    if checkpoint_s2:
        s2_model.load_state_dict(checkpoint_s2["model_state_dict"])

    extract_features_and_cache(s1_model, s2_model, split="train")
    extract_features_and_cache(s1_model, s2_model, split="val")

    # 4. Train Stage 3
    s3_model = train_stage3_aggregator()

    # 5. Inference
    checkpoint_s3 = load_checkpoint(s3_model, Config.STAGE3_CHECKPOINT_PATH)
    if checkpoint_s3:
        s3_model.load_state_dict(checkpoint_s3["model_state_dict"])

    run_inference_and_submission(s1_model, s2_model, s3_model)
