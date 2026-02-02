import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import (
    save_checkpoint,
    calculate_weighted_log_loss,
    load_dicom,
    get_bbox_from_mask,
    get_soft_anatomical_map,
    save_to_cache,
    load_from_cache,
    seed_everything,
)

# =============================================================================
# Loss Functions
# =============================================================================


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B, C, H, W) logits
        # targets: (B, H, W) long

        num_classes = preds.shape[1]
        preds = torch.softmax(preds, dim=1)

        # One-hot encode targets
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        # Calculate Dice for each class (skipping background if desired, but here we include all)
        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes and batch
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        loss_ce = self.ce(preds, targets)
        loss_dice = self.dice(preds, targets)
        return 0.5 * loss_ce + 0.5 * loss_dice


# =============================================================================
# Stage 1: Segmentation (UNetLocalizer)
# =============================================================================


def train_segmentor(
    train_loader,
    val_loader,
    model,
    optimizer,
    device,
    epochs,
    early_stopping_patience=3,
):
    criterion = SegmentationLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Stage 1: Segmentation Training")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_dice = validate_segmentor(val_loader, model, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Dice: {val_dice}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, Config.STAGE1_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break


def validate_segmentor(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            # Calculate Dice for monitoring (simplified macro average)
            preds = torch.argmax(outputs, dim=1)
            # Simple accuracy for now as proxy or implement full dice metric
            # Let's use the inverse of the DiceLoss component
            dice_obj = DiceLoss()
            batch_dice_loss = dice_obj(outputs, masks)
            dice_score += (1.0 - batch_dice_loss.item()) * images.size(0)

    val_loss /= len(val_loader.dataset)
    dice_score /= len(val_loader.dataset)

    return val_loss, dice_score


# =============================================================================
# Stage 1 Inference & Caching
# =============================================================================


def generate_stage1_results(metadata_df, model, device, load_cached_data=True):
    """
    Runs Stage 1 inference on all studies in metadata_df to generate:
    1. ROI Coordinates (for cropping)
    2. Anatomical Maps (for Stage 3)
    3. Bone Masks (for Stage 2 input)

    Results are cached to disk.
    """
    cache_file = "stage1_inference_results.parquet"

    if load_cached_data:
        cached_df = load_from_cache(cache_file, use_parquet=True)
        if cached_df is not None:
            print(f"Loaded Stage 1 results from cache: {len(cached_df)} rows")
            return cached_df

    print("Generating Stage 1 Inference Results...")
    model.eval()
    results = []

    # Group by study to process volume-wise
    # But metadata_df might be slice-level or study-level.
    # For efficiency, we assume we iterate unique studies.
    unique_studies = metadata_df["StudyInstanceUID"].unique()

    for study_uid in unique_studies:
        # Locate study path
        # We need to find the image path. We can look it up in metadata_df.
        study_row = metadata_df[metadata_df["StudyInstanceUID"] == study_uid].iloc[0]
        image_dir = os.path.join(Config.INPUT_DIR, study_row["image_path"])

        if not os.path.exists(image_dir):
            continue

        # List all DICOMs
        dcm_files = [f for f in os.listdir(image_dir) if f.endswith(".dcm")]
        # Sort by instance number
        dcm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        # Process in batches
        batch_size = 16
        for i in range(0, len(dcm_files), batch_size):
            batch_files = dcm_files[i : i + batch_size]
            batch_imgs = []

            for f in batch_files:
                img = load_dicom(
                    os.path.join(image_dir, f), output_size=Config.ORIGINAL_SIZE
                )
                batch_imgs.append(img)

            # Stack: (B, H, W) -> (B, 1, H, W)
            batch_tensor = (
                torch.tensor(np.array(batch_imgs), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)  # (B, 8, H, W)
                preds = torch.argmax(probs, dim=1)  # (B, H, W)

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for j, f_name in enumerate(batch_files):
                slice_idx = int(os.path.splitext(f_name)[0]) - 1  # 0-based index

                # 1. ROI
                mask = preds_np[j]
                bbox = get_bbox_from_mask(mask, margin=10)
                if bbox is None:
                    # Default center crop
                    center = Config.ORIGINAL_SIZE // 2
                    roi_center = [center, center]
                else:
                    y_min, x_min, y_max, x_max = bbox
                    roi_center = [(y_min + y_max) // 2, (x_min + x_max) // 2]

                # 2. Anatomical Map
                # Sum probabilities over spatial dims for each class to get presence
                # Or use the mask. Using mask is harder (binary).
                # Using soft probs is better.
                # But `get_soft_anatomical_map` uses mask labels. Let's stick to that for consistency.
                anat_map = get_soft_anatomical_map(mask, num_classes=7)

                # 3. Save Mask (Optional, for Stage 2 input)
                # To save space, we might not save every mask to disk unless necessary.
                # We can save the path if we save the file.
                # For this implementation, let's save the mask to a temporary folder.
                mask_save_dir = os.path.join(
                    Config.CACHE_DIR, "stage1_masks", study_uid
                )
                os.makedirs(mask_save_dir, exist_ok=True)
                mask_path = os.path.join(mask_save_dir, f"{slice_idx}.npy")
                np.save(mask_path, mask.astype(np.uint8))

                results.append(
                    {
                        "StudyInstanceUID": study_uid,
                        "slice_index": slice_idx,
                        "roi_y": roi_center[0],
                        "roi_x": roi_center[1],
                        "anatomical_map": anat_map.tolist(),  # Store as list
                        "mask_file": mask_path,
                    }
                )

    results_df = pd.DataFrame(results)
    save_to_cache(results_df, cache_file, use_parquet=True)
    return results_df


# =============================================================================
# Stage 2: Feature Encoder (DetailEncoder)
# =============================================================================


def train_encoder(
    train_loader,
    val_loader,
    model,
    optimizer,
    device,
    epochs,
    early_stopping_patience=3,
):
    # Unweighted BCEWithLogitsLoss for binary fracture classification per slice
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Stage 2: Encoder Training")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            # Model outputs features, we need to add a temporary classification head for training
            # However, DetailEncoder returns features.
            # We assume for training phase, we attach a linear layer or the model has one.
            # The provided DetailEncoder in models.py returns features (num_classes=0).
            # We need to wrap it or add a head here.
            # To strictly follow "Do not modify provided files", we add a head dynamically here.

            features = model(images)
            # Simple linear probe for training
            if not hasattr(model, "fc_probe"):
                model.fc_probe = nn.Linear(model.feature_dim, 1).to(device)
                # Add probe parameters to optimizer if not already
                optimizer.add_param_group({"params": model.fc_probe.parameters()})

            logits = model.fc_probe(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_acc = validate_encoder(val_loader, model, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Acc: {val_acc}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the encoder (backbone), not the probe
            save_checkpoint(model, optimizer, epoch, val_loss, Config.STAGE2_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break


def validate_encoder(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # Ensure probe exists
    if not hasattr(model, "fc_probe"):
        # If validating before training, create dummy probe
        model.fc_probe = nn.Linear(model.feature_dim, 1).to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            features = model(images)
            logits = model.fc_probe(features)
            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == (labels > 0.5)).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    acc = correct / total if total > 0 else 0.0

    return val_loss, acc


# =============================================================================
# Feature Extraction & Caching for Stage 3
# =============================================================================


def extract_features(
    metadata_df, stage1_df, stage2_model, device, load_cached_data=True
):
    """
    Runs Stage 2 inference on crops defined by Stage 1 results.
    Saves features and anatomical maps to .npy files for Stage 3.
    """
    feature_dir = os.path.join(Config.CACHE_DIR, "features")
    os.makedirs(feature_dir, exist_ok=True)

    # Check if all studies are already cached
    unique_studies = metadata_df["StudyInstanceUID"].unique()

    if load_cached_data:
        all_exist = True
        for uid in unique_studies:
            if not os.path.exists(os.path.join(feature_dir, f"{uid}.npy")):
                all_exist = False
                break
        if all_exist:
            print("All features found in cache.")
            return feature_dir

    print("Extracting features for Stage 3...")
    stage2_model.eval()

    # Create a lookup for Stage 1 results
    # stage1_df has columns: StudyInstanceUID, slice_index, roi_y, roi_x, anatomical_map, mask_file
    # We index by (StudyUID, slice_index)
    s1_lookup = stage1_df.set_index(["StudyInstanceUID", "slice_index"])

    for study_uid in unique_studies:
        save_path = os.path.join(feature_dir, f"{study_uid}.npy")
        if load_cached_data and os.path.exists(save_path):
            continue

        study_row = metadata_df[metadata_df["StudyInstanceUID"] == study_uid].iloc[0]
        image_dir = os.path.join(Config.INPUT_DIR, study_row["image_path"])

        if not os.path.exists(image_dir):
            continue

        dcm_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(".dcm")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        study_features = []
        study_anat_maps = []

        # Process in batches to save time
        batch_size = 32
        for i in range(0, len(dcm_files), batch_size):
            batch_files = dcm_files[i : i + batch_size]
            batch_crops = []

            for f in batch_files:
                slice_idx = int(os.path.splitext(f)[0]) - 1

                # Get Stage 1 info
                try:
                    s1_info = s1_lookup.loc[(study_uid, slice_idx)]
                    roi_y, roi_x = s1_info["roi_y"], s1_info["roi_x"]
                    anat_map = np.array(s1_info["anatomical_map"])
                    mask_file = s1_info["mask_file"]
                except KeyError:
                    # Fallback
                    roi_y, roi_x = Config.ORIGINAL_SIZE // 2, Config.ORIGINAL_SIZE // 2
                    anat_map = np.zeros(7)
                    mask_file = None

                study_anat_maps.append(anat_map)

                # Load 3 slices + Mask
                # Logic similar to CropClassificationDataset but inline
                slices = []
                for offset in [-1, 0, 1]:
                    target_idx = slice_idx + offset + 1  # 1-based file
                    path = os.path.join(image_dir, f"{target_idx}.dcm")
                    if not os.path.exists(path):
                        path = os.path.join(image_dir, f"{slice_idx + 1}.dcm")
                    img = load_dicom(path, output_size=Config.ORIGINAL_SIZE)
                    slices.append(img)

                img_3ch = np.stack(slices, axis=-1)

                # Load mask
                mask_4th = np.zeros(
                    (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), dtype=np.float32
                )
                if mask_file and os.path.exists(mask_file):
                    m = np.load(mask_file)
                    if m.shape != (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE):
                        import cv2

                        m = cv2.resize(
                            m,
                            (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_4th = (m > 0).astype(np.float32)

                combined = np.concatenate(
                    [img_3ch, mask_4th[:, :, np.newaxis]], axis=-1
                )

                # Crop
                crop_size = Config.CROP_SIZE
                half = crop_size // 2
                y_min = max(0, int(roi_y) - half)
                y_max = min(Config.ORIGINAL_SIZE, int(roi_y) + half)
                x_min = max(0, int(roi_x) - half)
                x_max = min(Config.ORIGINAL_SIZE, int(roi_x) + half)

                # Adjust edges
                if y_max - y_min < crop_size:
                    if y_min == 0:
                        y_max = crop_size
                    else:
                        y_min = Config.ORIGINAL_SIZE - crop_size
                if x_max - x_min < crop_size:
                    if x_min == 0:
                        x_max = crop_size
                    else:
                        x_min = Config.ORIGINAL_SIZE - crop_size

                crop = combined[y_min:y_max, x_min:x_max, :]
                batch_crops.append(crop)

            # To Tensor: (B, 4, H, W)
            batch_tensor = (
                torch.tensor(np.array(batch_crops), dtype=torch.float32)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            with torch.no_grad():
                feats = stage2_model(batch_tensor)  # (B, 1280)

            study_features.append(feats.cpu().numpy())

        # Concatenate study data
        if len(study_features) > 0:
            full_features = np.concatenate(study_features, axis=0)
            full_anat_maps = np.array(study_anat_maps)
        else:
            full_features = np.zeros((0, Config.STAGE2_FEATURE_DIM))
            full_anat_maps = np.zeros((0, 7))

        # Save
        data_to_save = {"features": full_features, "anatomical_map": full_anat_maps}
        np.save(save_path, data_to_save)

    return feature_dir


# =============================================================================
# Stage 3: Aggregator (HierarchicalRNN)
# =============================================================================


def train_aggregator(
    train_loader,
    val_loader,
    model,
    optimizer,
    device,
    epochs,
    early_stopping_patience=5,
):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Stage 3: Aggregator Training")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, anat_maps, targets in train_loader:
            features = features.to(device)
            anat_maps = anat_maps.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, probs = model(features, anat_maps)

            loss = calculate_weighted_log_loss(targets, probs, device=device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss = validate_aggregator(val_loader, model, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, Config.STAGE3_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break


def validate_aggregator(val_loader, model, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for features, anat_maps, targets in val_loader:
            features = features.to(device)
            anat_maps = anat_maps.to(device)
            targets = targets.to(device)

            logits, probs = model(features, anat_maps)
            loss = calculate_weighted_log_loss(targets, probs, device=device)

            val_loss += loss.item() * features.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss


# =============================================================================
# Inference Pipeline for Submission
# =============================================================================


def generate_submission(model, test_loader, test_df, device):
    """
    Generates submission.csv using the trained Stage 3 model.
    Assumes test_loader yields (features, anat_maps, dummy_targets) in order of test_df.
    """
    model.eval()
    predictions = []

    print("Generating Submission...")

    with torch.no_grad():
        for i, (features, anat_maps, _) in enumerate(test_loader):
            features = features.to(device)
            anat_maps = anat_maps.to(device)

            _, probs = model(features, anat_maps)
            probs_np = probs.cpu().numpy()

            # Batch size in test_loader corresponds to patients
            # We need to map back to StudyInstanceUID
            start_idx = i * test_loader.batch_size
            end_idx = start_idx + features.size(0)
            batch_uids = test_df.iloc[start_idx:end_idx]["StudyInstanceUID"].values

            for j, uid in enumerate(batch_uids):
                # probs_np[j] is shape (8,) -> [C1, C2, C3, C4, C5, C6, C7, Patient]
                p = probs_np[j]

                # Format rows
                row_ids = [
                    f"{uid}_C1",
                    f"{uid}_C2",
                    f"{uid}_C3",
                    f"{uid}_C4",
                    f"{uid}_C5",
                    f"{uid}_C6",
                    f"{uid}_C7",
                    f"{uid}_patient_overall",
                ]

                for k, row_id in enumerate(row_ids):
                    predictions.append({"row_id": row_id, "fractured": float(p[k])})

    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
