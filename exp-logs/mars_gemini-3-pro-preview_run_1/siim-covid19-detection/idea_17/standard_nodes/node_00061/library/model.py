import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from library.config import Config
from library.dataset import load_data, SIIMDataset, get_transforms
from library.utils import AverageMeter, calculate_map, mask2bbox, seed_everything

# =============================================================================
# Model Architecture
# =============================================================================


class DecoderBlock(nn.Module):
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
            # Handle potential padding issues if shapes don't match exactly
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class StochasticResNet34UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: ResNet34 with Stochastic Depth
        # features_only=True returns features at strides [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            drop_path_rate=Config.DROP_PATH_RATE,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Feature channels for ResNet34: [64, 64, 128, 256, 512]
        enc_channels = self.encoder.feature_info.channels()

        # Decoder
        # Center: 512 -> 256
        self.center = nn.Sequential(
            nn.Conv2d(enc_channels[4], 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Dec4: In(256) + Skip(256) -> Out(256)
        self.dec4 = DecoderBlock(256, enc_channels[3], 256)
        # Dec3: In(256) + Skip(128) -> Out(128)
        self.dec3 = DecoderBlock(256, enc_channels[2], 128)
        # Dec2: In(128) + Skip(64) -> Out(64)
        self.dec2 = DecoderBlock(128, enc_channels[1], 64)
        # Dec1: In(64) + Skip(64) -> Out(32)
        self.dec1 = DecoderBlock(64, enc_channels[0], 32)

        # Final upsample to original resolution (stride 2 -> 1)
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Segmentation Head
        )

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(enc_channels[4], Config.NUM_CLASSES)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features: [c0(64, /2), c1(64, /4), c2(128, /8), c3(256, /16), c4(512, /32)]

        # Classification Branch (from deepest feature)
        global_feat = self.avgpool(features[4]).flatten(1)
        study_logits = self.fc(global_feat)

        # Segmentation Branch (U-Net Decoder)
        # Center
        x = self.center(features[4])

        # Decode
        x = self.dec4(x, features[3])
        x = self.dec3(x, features[2])
        x = self.dec2(x, features[1])
        x = self.dec1(x, features[0])

        # Final resolution
        mask_logits = self.final_conv(x)

        return study_logits, mask_logits


# =============================================================================
# Training Engine
# =============================================================================


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    losses = AverageMeter()

    # Loss functions
    criterion_study = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)

        with autocast():
            study_logits, mask_logits = model(images)

            # Study Loss (Multi-class classification)
            # labels are one-hot, CrossEntropyLoss expects class indices
            study_targets = torch.argmax(labels, dim=1)
            loss_study = criterion_study(study_logits, study_targets)

            # Mask Loss
            loss_mask = criterion_mask(mask_logits, masks)

            # Weighted Sum
            loss = (Config.STUDY_LOSS_WEIGHT * loss_study) + (
                Config.IMAGE_LOSS_WEIGHT * loss_mask
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=losses.avg, lr=optimizer.param_groups[0]["lr"])

    return losses.avg


def validate(model, loader, device):
    model.eval()
    losses = AverageMeter()

    criterion_study = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    # Store predictions for mAP calculation
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []

    study_preds = []
    study_gts = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            orig_dims = batch["orig_dim"].numpy()  # (N, 2) -> h, w

            study_logits, mask_logits = model(images)

            # Loss
            study_targets = torch.argmax(labels, dim=1)
            loss_study = criterion_study(study_logits, study_targets)
            loss_mask = criterion_mask(mask_logits, masks)
            loss = (Config.STUDY_LOSS_WEIGHT * loss_study) + (
                Config.IMAGE_LOSS_WEIGHT * loss_mask
            )
            losses.update(loss.item(), images.size(0))

            # Process Study Predictions
            probs = torch.softmax(study_logits, dim=1)
            study_preds.append(probs.cpu().numpy())
            study_gts.append(labels.cpu().numpy())

            # Process Mask Predictions for mAP
            mask_probs = torch.sigmoid(mask_logits).cpu().numpy()  # (N, 1, H, W)

            # GT Boxes (Need to extract from masks or load from metadata,
            # but here we extract from GT masks for consistency in validation loop)
            gt_masks = masks.cpu().numpy()

            for i in range(images.size(0)):
                # Resize mask prob back to original size for accurate box extraction
                h, w = orig_dims[i]

                # Prediction Boxes
                curr_mask = mask_probs[i, 0]
                curr_mask_resized = (
                    curr_mask  # We calculate mAP on 512x512 for speed in val
                )

                boxes = mask2bbox(curr_mask_resized, threshold=0.5)
                # Normalize scores for boxes (using mean pixel value in box)
                scores = []
                for b in boxes:
                    x1, y1, x2, y2 = b
                    # Clip coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(Config.IMG_SIZE, x2), min(Config.IMG_SIZE, y2)
                    if x2 > x1 and y2 > y1:
                        score = np.mean(curr_mask_resized[y1:y2, x1:x2])
                        scores.append(score)
                    else:
                        scores.append(0.0)

                if len(boxes) > 0:
                    pred_boxes_list.append(np.array(boxes))
                    pred_scores_list.append(np.array(scores))
                    pred_labels_list.append(np.zeros(len(boxes)))  # Class 0 for opacity
                else:
                    pred_boxes_list.append(np.empty((0, 4)))
                    pred_scores_list.append(np.array([]))
                    pred_labels_list.append(np.array([]))

                # GT Boxes
                curr_gt_mask = gt_masks[i, 0]
                g_boxes = mask2bbox(curr_gt_mask, threshold=0.5)
                if len(g_boxes) > 0:
                    gt_boxes_list.append(np.array(g_boxes))
                    gt_labels_list.append(np.zeros(len(g_boxes)))
                else:
                    gt_boxes_list.append(np.empty((0, 4)))
                    gt_labels_list.append(np.array([]))

    # Calculate mAP
    # Note: This is an approximation of the competition metric for validation monitoring
    image_map = calculate_map(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        num_classes=1,
    )

    # Study Accuracy/mAP (Simplified to Accuracy for monitoring, or mAP)
    # Competition uses mAP for study as well.
    # We can approximate study performance by accuracy of the argmax
    study_preds = np.concatenate(study_preds)
    study_gts = np.concatenate(study_gts)
    study_acc = (np.argmax(study_preds, axis=1) == np.argmax(study_gts, axis=1)).mean()

    return losses.avg, image_map, study_acc


def train_model():
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Data Loading
    train_data = load_data(Config.TRAIN_METADATA, "train", load_cached_data=True)
    val_data = load_data(Config.VAL_METADATA, "val", load_cached_data=True)

    train_dataset = SIIMDataset(train_data, "train", transform=get_transforms("train"))
    val_dataset = SIIMDataset(val_data, "val", transform=get_transforms("val"))

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

    # Model Setup
    model = StochasticResNet34UNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Total steps for Cosine Annealing
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS * steps_per_epoch, eta_min=Config.ETA_MIN
    )

    scaler = GradScaler()

    best_score = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        val_loss, val_map, val_acc = validate(model, val_loader, device)

        # Composite score: Average of Image mAP and Study Acc (Proxy for Study mAP)
        # In competition, it's 2/3 image mAP + 1/3 study mAP usually, but simple avg is fine for checkpointing
        composite_score = (val_map + val_acc) / 2

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Image mAP: {val_map:.6f} | Study Acc: {val_acc:.6f} | Score: {composite_score:.6f}"
        )

        if composite_score > best_score:
            best_score = composite_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! Score: {best_score:.6f}")

    print(f"Training Complete. Best Score: {best_score}")


# =============================================================================
# Inference Engine
# =============================================================================


def predict():
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Data
    test_data = load_data(Config.TEST_METADATA, "test", load_cached_data=True)
    test_dataset = SIIMDataset(test_data, "test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = StochasticResNet34UNet().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    results = []
    study_classes = ["negative", "typical", "indeterminate", "atypical"]

    print("Running Inference...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(device)
            ids = batch["id"]  # tuple of study_ids
            orig_dims = batch["orig_dim"].numpy()

            # TTA: Original + Horizontal Flip
            # Forward Original
            study_logits, mask_logits = model(images)
            study_probs = torch.softmax(study_logits, dim=1)
            mask_probs = torch.sigmoid(mask_logits)

            # Forward Flip
            images_flip = torch.flip(images, dims=[3])
            study_logits_f, mask_logits_f = model(images_flip)
            study_probs_f = torch.softmax(study_logits_f, dim=1)
            mask_probs_f = torch.sigmoid(mask_logits_f)
            mask_probs_f = torch.flip(mask_probs_f, dims=[3])

            # Average
            avg_study_probs = (study_probs + study_probs_f) / 2.0
            avg_mask_probs = (mask_probs + mask_probs_f) / 2.0

            avg_study_probs = avg_study_probs.cpu().numpy()
            avg_mask_probs = avg_mask_probs.cpu().numpy()

            # Generate Prediction Strings
            for i in range(len(ids)):
                study_id = ids[i]
                h, w = orig_dims[i]

                # 1. Study Prediction
                # Format: "class conf 0 0 1 1" for all classes
                study_pred_strs = []
                for idx, cls_name in enumerate(study_classes):
                    conf = avg_study_probs[i, idx]
                    study_pred_strs.append(f"{cls_name} {conf:.6f} 0 0 1 1")
                study_prediction_string = " ".join(study_pred_strs)

                results.append(
                    {
                        "id": f"{study_id}_study",
                        "PredictionString": study_prediction_string,
                    }
                )

                # 2. Image Prediction
                # Logic: If 'negative' is the highest class, predict none.
                # Else, predict boxes.

                # Check if negative is dominant
                neg_idx = 0  # 'negative' is first in our list
                pred_class_idx = np.argmax(avg_study_probs[i])

                if pred_class_idx == neg_idx:
                    image_prediction_string = "none 1 0 0 1 1"
                else:
                    # Extract Boxes
                    mask = avg_mask_probs[i, 0]
                    # Resize mask to original image dimensions for correct box coordinates
                    mask_resized = cv2.resize(
                        mask, (w, h), interpolation=cv2.INTER_LINEAR
                    )

                    boxes = mask2bbox(mask_resized, threshold=0.5)

                    if len(boxes) == 0:
                        image_prediction_string = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            # Calculate score for this box
                            # Use mean of probability map within box
                            box_score = np.mean(mask_resized[y1:y2, x1:x2])
                            box_strs.append(
                                f"opacity {box_score:.6f} {x1} {y1} {x2} {y2}"
                            )
                        image_prediction_string = " ".join(box_strs)

                # The image ID in submission is usually image_id_image.
                # However, our dataset loader returns study_id for 'id'.
                # We need to map study_id to image_id or assume 1-to-1 mapping.
                # In this dataset, test studies have images.
                # We need to get the image_id corresponding to this study.
                # The metadata loader logic in dataset.py puts study_id in 'ids'.
                # Let's retrieve the image_id from the metadata dataframe using the study_id.
                # Note: This is slightly inefficient inside the loop but robust.
                # Ideally, dataset should return image_id as well.
                # HACK: The test metadata csv has image_id. We can look it up or modify dataset.
                # Since we can't modify dataset.py, we rely on the fact that test.csv is loaded.
                # We will just append _image to the study_id if we can't find the image id,
                # BUT the submission requires specific image IDs.
                # Let's look at test_data dict loaded at start of function.
                # It has 'ids' which are study_ids.
                # We need image_ids.
                # We can re-read test.csv to get the mapping.
                pass

    # Fix for Image IDs:
    # We need to match the rows in results to the correct IDs.
    # The loop above iterates in order of the loader.
    # The loader is sequential (shuffle=False).
    # The test_data['ids'] contains study_ids.
    # We need to get the corresponding image_ids.
    test_df = pd.read_csv(Config.TEST_METADATA)
    # Ensure order matches loader
    # dataset.py loads arrays. If we assume the order in CSV matches arrays (it does in load_data),
    # we can just zip with the dataframe.

    final_results = []

    # We need to re-run the loop logic or just map the results.
    # Let's rebuild the results list properly using the dataframe index

    # Re-running inference logic in a cleaner way to align with DF
    # We already computed predictions, let's store them in lists and then merge with DF

    all_study_preds = []
    all_mask_preds = []

    # Re-run inference loop to collect arrays (since we need to align with DF)
    # To save time, I will just collect them in the loop above.
    # Let's rewrite the loop part to be efficient.

    # ... (Redoing the loop logic inside the predict function for correctness)

    # Efficient collection
    ptr = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(device)
            orig_dims = batch["orig_dim"].numpy()
            batch_size = images.size(0)

            # Forward
            study_logits, mask_logits = model(images)
            study_probs = torch.softmax(study_logits, dim=1)
            mask_probs = torch.sigmoid(mask_logits)

            # Flip
            images_flip = torch.flip(images, dims=[3])
            study_logits_f, mask_logits_f = model(images_flip)
            study_probs_f = torch.softmax(study_logits_f, dim=1)
            mask_probs_f = torch.sigmoid(mask_logits_f)
            mask_probs_f = torch.flip(mask_probs_f, dims=[3])

            avg_study_probs = (study_probs + study_probs_f) / 2.0
            avg_mask_probs = (mask_probs + mask_probs_f) / 2.0

            avg_study_probs = avg_study_probs.cpu().numpy()
            avg_mask_probs = avg_mask_probs.cpu().numpy()

            # Process batch
            for i in range(batch_size):
                # Get metadata from DF using pointer
                row = test_df.iloc[ptr]
                ptr += 1

                study_id = row["study_id"]
                image_id = row["image_id"]
                h, w = orig_dims[i]

                # 1. Study Prediction
                study_pred_strs = []
                for idx, cls_name in enumerate(study_classes):
                    conf = avg_study_probs[i, idx]
                    study_pred_strs.append(f"{cls_name} {conf:.6f} 0 0 1 1")
                study_str = " ".join(study_pred_strs)

                final_results.append(
                    {"id": f"{study_id}_study", "PredictionString": study_str}
                )

                # 2. Image Prediction
                neg_idx = 0
                pred_class_idx = np.argmax(avg_study_probs[i])

                if pred_class_idx == neg_idx:
                    img_str = "none 1 0 0 1 1"
                else:
                    mask = avg_mask_probs[i, 0]
                    mask_resized = cv2.resize(
                        mask, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                    boxes = mask2bbox(mask_resized, threshold=0.5)

                    if len(boxes) == 0:
                        img_str = "none 1 0 0 1 1"
                    else:
                        box_strs = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            box_score = np.mean(mask_resized[y1:y2, x1:x2])
                            box_strs.append(
                                f"opacity {box_score:.6f} {x1} {y1} {x2} {y2}"
                            )
                        img_str = " ".join(box_strs)

                final_results.append(
                    {"id": f"{image_id}_image", "PredictionString": img_str}
                )

    submission_df = pd.DataFrame(final_results)
    # Remove duplicates if any (though logic shouldn't produce them)
    submission_df = submission_df.drop_duplicates(subset=["id"])

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
