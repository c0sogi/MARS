import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.config import Config
from library.models import CenterNetDetector, ResNetClassifier
from library.dataset import KuzushijiDetectionDataset, KuzushijiClassificationDataset
from library.utils import get_affine_transform, affine_transform

# =============================================================================
# 1. Loss Functions
# =============================================================================


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for CenterNet Heatmap.
    Penalizes false positives and false negatives, with reduced penalty near GT centers.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        pred: (B, 1, H, W) - Sigmoid output
        gt: (B, 1, H, W) - Ground truth heatmap with Gaussian peaks
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative samples (background)
        # If gt is near 1 (near center), weight is small.
        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        # Loss for positive samples (peaks)
        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Loss for negative samples (background)
        neg_loss = (
            torch.log(1 - pred + 1e-12)
            * torch.pow(pred, self.alpha)
            * neg_weights
            * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss(nn.Module):
    """
    L1 Loss for regression heads (Size, Offset), masked by object presence.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (B, 2, H, W)
        target: (B, 2, H, W)
        mask: (B, 1, H, W) - 1 at object centers, 0 otherwise
        """
        # Expand mask to match channel dim
        mask = mask.expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by number of objects
        num_objs = (
            mask.sum() / 2
        )  # divide by 2 because mask is duplicated for 2 channels
        loss = loss / (num_objs + 1e-4)
        return loss


# =============================================================================
# 2. Training Engines
# =============================================================================


def train_detector(debug=False):
    print("\n=== Starting Detector Training ===")

    # Data Loaders
    train_ds = KuzushijiDetectionDataset(split="train", debug=debug)
    val_ds = KuzushijiDetectionDataset(split="val", debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model & Optimizer
    model = CenterNetDetector(pretrained=True).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.DETECTOR_LR)

    # Losses
    criterion_hm = ModifiedFocalLoss()
    criterion_reg = RegL1Loss()

    # Training Loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    epochs = 2 if debug else Config.DETECTOR_EPOCHS

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(Config.DEVICE)
            hm_gt = targets["hm"].to(Config.DEVICE)
            wh_gt = targets["wh"].to(Config.DEVICE)
            reg_gt = targets["reg"].to(Config.DEVICE)
            mask_gt = targets["reg_mask"].to(Config.DEVICE)

            optimizer.zero_grad()
            hm_pred, wh_pred, reg_pred = model(imgs)

            loss_hm = criterion_hm(hm_pred, hm_gt)
            loss_wh = criterion_reg(wh_pred, wh_gt, mask_gt)
            loss_off = criterion_reg(reg_pred, reg_gt, mask_gt)

            # Weighted sum: Heatmap is primary, Size is scaled down, Offset is standard
            loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_off

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(Config.DEVICE)
                hm_gt = targets["hm"].to(Config.DEVICE)
                wh_gt = targets["wh"].to(Config.DEVICE)
                reg_gt = targets["reg"].to(Config.DEVICE)
                mask_gt = targets["reg_mask"].to(Config.DEVICE)

                hm_pred, wh_pred, reg_pred = model(imgs)

                loss_hm = criterion_hm(hm_pred, hm_gt)
                loss_wh = criterion_reg(wh_pred, wh_gt, mask_gt)
                loss_off = criterion_reg(reg_pred, reg_gt, mask_gt)

                loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_off
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss}"
        )

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(), os.path.join(Config.MODEL_DIR, "detector_best.pth")
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered for Detector.")
                break

    return model


def train_classifier(debug=False):
    print("\n=== Starting Classifier Training ===")

    # Data Loaders
    train_ds = KuzushijiClassificationDataset(split="train", debug=debug)
    val_ds = KuzushijiClassificationDataset(split="val", debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model & Optimizer
    model = ResNetClassifier(pretrained=True).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.CLASSIFIER_LR)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0
    patience = 3
    patience_counter = 0

    epochs = 2 if debug else Config.CLASSIFIER_EPOCHS

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss_accum = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss_accum / len(train_loader)
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss_accum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss_accum / len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} Acc: {train_acc} | Val Loss: {avg_val_loss} Acc: {val_acc}"
        )

        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(Config.MODEL_DIR, "classifier_best.pth"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered for Classifier.")
                break

    return model


# =============================================================================
# 3. Inference & Submission
# =============================================================================


def _nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def generate_submission(detector=None, classifier=None):
    print("\n=== Generating Submission ===")

    # Load Models if not provided
    if detector is None:
        detector = CenterNetDetector(pretrained=False).to(Config.DEVICE)
        detector.load_state_dict(
            torch.load(
                os.path.join(Config.MODEL_DIR, "detector_best.pth"),
                map_location=Config.DEVICE,
            )
        )

    if classifier is None:
        classifier = ResNetClassifier(pretrained=False).to(Config.DEVICE)
        classifier.load_state_dict(
            torch.load(
                os.path.join(Config.MODEL_DIR, "classifier_best.pth"),
                map_location=Config.DEVICE,
            )
        )

    detector.eval()
    classifier.eval()

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load Class Map (Reverse lookup: ID -> Unicode)
    # We need to reconstruct the class map used in dataset.py
    # Since dataset.py sorts unique codes from train set, we do the same here
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    all_codes = set()
    for labels in df_train["labels"].dropna():
        parts = labels.split()
        for i in range(0, len(parts), 5):
            all_codes.add(parts[i])
    unique_codes = sorted(list(all_codes))
    id_to_code = {idx: code for idx, code in enumerate(unique_codes)}

    results = []

    # Process each image
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img_id = row["image_id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # 1. Load and Preprocess for Detector
        img_raw = cv2.imread(file_path)
        if img_raw is None:
            results.append({"image_id": img_id, "labels": ""})
            continue

        h_orig, w_orig = img_raw.shape[:2]

        # Affine Transform to 1024x1024 (Centered)
        c = np.array([w_orig / 2.0, h_orig / 2.0], dtype=np.float32)
        s = max(h_orig, w_orig) * 1.0

        # Input transform
        trans_input = get_affine_transform(
            c, s, 0, [Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE]
        )
        inp_img = cv2.warpAffine(
            img_raw,
            trans_input,
            (Config.DETECTOR_INPUT_SIZE, Config.DETECTOR_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
        )

        # Normalize
        inp_tensor = inp_img.astype(np.float32) / 255.0
        inp_tensor = (inp_tensor - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        inp_tensor = inp_tensor.transpose(2, 0, 1)
        inp_tensor = torch.from_numpy(inp_tensor).unsqueeze(0).float().to(Config.DEVICE)

        # 2. Run Detector
        with torch.no_grad():
            hm, wh, offset = detector(inp_tensor)
            hm = _nms(hm)  # Apply NMS

        # 3. Decode Detections
        # Flatten
        batch, channel, height, width = hm.shape
        hm = hm.view(batch, -1)
        scores, inds = torch.topk(hm, k=Config.MAX_DETECTIONS)

        # Filter by threshold
        mask = scores > Config.CONF_THRESHOLD
        scores = scores[mask]
        inds = inds[mask]

        if len(scores) == 0:
            results.append({"image_id": img_id, "labels": ""})
            continue

        # Get coordinates in Output Space (256x256)
        ys = (inds // width).float()
        xs = (inds % width).float()

        # Add offset (reg)
        # Gather offset values at indices
        offset = offset.view(batch, 2, -1)
        wh = wh.view(batch, 2, -1)

        reg_x = torch.gather(offset[:, 0, :], 1, inds.unsqueeze(0)).squeeze(0)
        reg_y = torch.gather(offset[:, 1, :], 1, inds.unsqueeze(0)).squeeze(0)
        w_det = torch.gather(wh[:, 0, :], 1, inds.unsqueeze(0)).squeeze(0)
        h_det = torch.gather(wh[:, 1, :], 1, inds.unsqueeze(0)).squeeze(0)

        xs = xs + reg_x
        ys = ys + reg_y

        # 4. Map to Original Coordinates
        # We use the inverse affine transform from the Output Space (256x256) to Original
        output_size = Config.DETECTOR_INPUT_SIZE // 4  # 256
        trans_output_inv = get_affine_transform(
            c, s, 0, [output_size, output_size], inv=1
        )

        # Prepare points for transform
        pts_out = torch.stack([xs, ys], dim=1).cpu().numpy()  # (N, 2)
        pts_orig = affine_transform(pts_out.T, trans_output_inv).T  # (N, 2)

        # Width/Height scaling
        resize_ratio = output_size / s
        w_orig_det = w_det.cpu().numpy() / resize_ratio
        h_orig_det = h_det.cpu().numpy() / resize_ratio

        # 5. Extract Crops and Classify
        crops = []
        valid_indices = []

        for i in range(len(pts_orig)):
            cx, cy = pts_orig[i]
            cw, ch = w_orig_det[i], h_orig_det[i]

            x1 = int(cx - cw / 2)
            y1 = int(cy - ch / 2)
            w_crop = int(cw)
            h_crop = int(ch)

            # Clamp
            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            w_crop = max(1, min(w_crop, w_orig - x1))
            h_crop = max(1, min(h_crop, h_orig - y1))

            crop = img_raw[y1 : y1 + h_crop, x1 : x1 + w_crop]
            if crop.size == 0:
                continue

            # Preprocess crop
            crop = cv2.resize(
                crop, (Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE)
            )
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array(
                [0.229, 0.224, 0.225]
            )
            crop = crop.transpose(2, 0, 1)

            crops.append(crop)
            valid_indices.append(i)

        if not crops:
            results.append({"image_id": img_id, "labels": ""})
            continue

        # Batch Classify
        crops_tensor = torch.from_numpy(np.stack(crops)).float().to(Config.DEVICE)

        # Process in chunks to avoid OOM if many detections
        batch_size_cls = Config.CLASSIFIER_BATCH_SIZE
        pred_labels = []

        with torch.no_grad():
            for i in range(0, len(crops_tensor), batch_size_cls):
                batch_crops = crops_tensor[i : i + batch_size_cls]
                outputs = classifier(batch_crops)
                _, preds = torch.max(outputs, 1)
                pred_labels.extend(preds.cpu().numpy())

        # 6. Filter and Format
        label_strs = []
        for i, pred_cls in enumerate(pred_labels):
            if pred_cls == Config.BACKGROUND_CLASS_ID:
                continue

            orig_idx = valid_indices[i]
            cx, cy = pts_orig[orig_idx]

            # Format: Unicode X Y
            if pred_cls in id_to_code:
                code = id_to_code[pred_cls]
                label_strs.append(f"{code} {int(cx)} {int(cy)}")

        results.append({"image_id": img_id, "labels": " ".join(label_strs)})

    # Save Submission
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_and_inference(debug=False):
    """
    Orchestrates the full pipeline.
    """
    # Train Detector
    detector = train_detector(debug=debug)

    # Train Classifier
    classifier = train_classifier(debug=debug)

    # Generate Submission
    generate_submission(detector, classifier)
