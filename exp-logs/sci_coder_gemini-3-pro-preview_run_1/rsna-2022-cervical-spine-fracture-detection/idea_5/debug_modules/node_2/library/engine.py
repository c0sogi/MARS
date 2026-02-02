import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2

from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    weighted_log_loss,
    read_dicom,
    process_segmentation,
    save_to_cache,
    load_from_cache,
    seed_everything,
)
from library.models import UNetLocalizer, FractureEncoder, AnatomicalTransformer
from library.data import get_dataloaders, get_transforms

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W) with class indices 0-7
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        # One-hot encode targets
        true_1_hot = torch.eye(num_classes, device=logits.device)[
            targets
        ]  # (B, H, W, C)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute Dice for each class
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1_hot, dims)
        cardinality = torch.sum(probs + true_1_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average Dice Loss (1 - Mean Dice)
        return 1.0 - torch.mean(dice)


def weighted_bce_loss(logits, targets):
    """
    Weighted Binary Cross Entropy matching the competition metric weights.
    Weights: C1-C7 = 1.0, Patient_Overall = 7.0
    """
    weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0], device=logits.device
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, targets)
    weighted_loss = loss * weights
    return weighted_loss.mean()


# -----------------------------------------------------------------------------
# Training Helper
# -----------------------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience=3, mode="min", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, score, model, optimizer, epoch, path):
        if self.mode == "min":
            improved = (
                self.best_score is None or score < self.best_score - self.min_delta
            )
        else:
            improved = (
                self.best_score is None or score > self.best_score + self.min_delta
            )

        if improved:
            self.best_score = score
            self.counter = 0
            save_checkpoint(model, optimizer, epoch, path, metric=score)
            return True  # Checkpoint saved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# -----------------------------------------------------------------------------
# Stage 1: Segmentation
# -----------------------------------------------------------------------------


def train_segmentor(load_cached_data=True):
    """
    Trains the U-Net Localizer.
    """
    print("\n" + "=" * 40)
    print("Stage 1: Training Segmentation Model")
    print("=" * 40)

    # Check if model already exists
    if load_cached_data and os.path.exists(Config.SEG_MODEL_PATH):
        print(f"Loading pre-trained segmentor from {Config.SEG_MODEL_PATH}")
        return

    # Load Data
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    train_loader, val_loader = get_dataloaders("segmentation", df_train, df_val)

    # Model Setup
    model = UNetLocalizer(num_classes=Config.SEG_NUM_CLASSES).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.SEG_LR)
    criterion = DiceLoss()
    early_stopping = EarlyStopping(patience=5, mode="min")

    # Training Loop
    for epoch in range(Config.SEG_EPOCHS):
        model.train()
        train_loss = AverageMeter()

        for imgs, masks in train_loader:
            imgs = imgs.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), imgs.size(0))

        # Validation
        model.eval()
        val_loss = AverageMeter()

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss.update(loss.item(), imgs.size(0))

        print(
            f"Epoch {epoch+1}/{Config.SEG_EPOCHS} | "
            f"Train Loss: {train_loss.avg:.8f} | "
            f"Val Loss: {val_loss.avg:.8f}"
        )

        if early_stopping(val_loss.avg, model, optimizer, epoch, Config.SEG_MODEL_PATH):
            print("  [Checkpoint Saved]")

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Cleanup
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# Stage 2: Encoder
# -----------------------------------------------------------------------------


def train_encoder(load_cached_data=True):
    """
    Trains the 2.5D CNN Encoder.
    """
    print("\n" + "=" * 40)
    print("Stage 2: Training Fracture Encoder")
    print("=" * 40)

    if load_cached_data and os.path.exists(Config.ENC_MODEL_PATH):
        print(f"Loading pre-trained encoder from {Config.ENC_MODEL_PATH}")
        return

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    train_loader, val_loader = get_dataloaders("classifier", df_train, df_val)

    model = FractureEncoder().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.ENC_LR)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=3, mode="min")

    for epoch in range(Config.ENC_EPOCHS):
        model.train()
        train_loss = AverageMeter()

        for imgs, labels in train_loader:
            imgs = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)  # (B,)

            optimizer.zero_grad()
            feats = model(
                imgs
            )  # (B, Dim) - Wait, model returns features. We need a head for training.

            # Temporary Classification Head for Training
            # The FractureEncoder returns features. We need to project to 1 output.
            # We can just add a linear layer here or modify model.
            # Ideally, the model class should support a 'classifier' mode, but it's defined as an encoder.
            # We will append a linear layer just for this training function.
            pass

        # Re-initializing model with a head for training purposes
        # Since FractureEncoder is strictly an encoder, we wrap it.
        class EncoderTrainer(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.head = nn.Linear(encoder.feature_dim, 1)

            def forward(self, x):
                return self.head(self.encoder(x))

        trainer_model = EncoderTrainer(model).to(Config.DEVICE)
        # Re-init optimizer for wrapper
        optimizer = optim.Adam(trainer_model.parameters(), lr=Config.ENC_LR)

        # Loop restart
        break

    # Correct Loop with Wrapper
    trainer_model = EncoderTrainer(model).to(Config.DEVICE)
    optimizer = optim.Adam(trainer_model.parameters(), lr=Config.ENC_LR)

    for epoch in range(Config.ENC_EPOCHS):
        trainer_model.train()
        train_loss = AverageMeter()

        for imgs, labels in train_loader:
            imgs = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            logits = trainer_model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), imgs.size(0))

        trainer_model.eval()
        val_loss = AverageMeter()

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE).unsqueeze(1)
                logits = trainer_model(imgs)
                loss = criterion(logits, labels)
                val_loss.update(loss.item(), imgs.size(0))

        print(
            f"Epoch {epoch+1}/{Config.ENC_EPOCHS} | "
            f"Train Loss: {train_loss.avg:.8f} | "
            f"Val Loss: {val_loss.avg:.8f}"
        )

        # We save the underlying encoder, not the wrapper
        if early_stopping(val_loss.avg, model, optimizer, epoch, Config.ENC_MODEL_PATH):
            print("  [Checkpoint Saved]")

        if early_stopping.early_stop:
            break

    del trainer_model, model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------------


def extract_features(load_cached_data=True):
    """
    Runs the full pipeline (Seg -> Crop -> Enc) to generate features for Train, Val, and Test.
    Saves results to cache.
    """
    print("\n" + "=" * 40)
    print("Feature Extraction Phase")
    print("=" * 40)

    # Check if all cache files exist
    if (
        load_cached_data
        and os.path.exists(Config.TRAIN_FEATURES_CACHE)
        and os.path.exists(Config.VAL_FEATURES_CACHE)
        and os.path.exists(Config.TEST_FEATURES_CACHE)
    ):
        print("All features cached. Skipping extraction.")
        return

    # Load Models
    seg_model = UNetLocalizer(num_classes=Config.SEG_NUM_CLASSES).to(Config.DEVICE)
    load_checkpoint(seg_model, None, Config.SEG_MODEL_PATH, device=Config.DEVICE)
    seg_model.eval()

    enc_model = FractureEncoder().to(Config.DEVICE)
    load_checkpoint(enc_model, None, Config.ENC_MODEL_PATH, device=Config.DEVICE)
    enc_model.eval()

    # Transform
    seg_transform = get_transforms("segmentation", "val")
    enc_transform = get_transforms("classifier", "val")

    def process_subset(df, cache_path, desc):
        if load_cached_data and os.path.exists(cache_path):
            print(f"{desc} features already cached.")
            return

        features_dict = {}
        # We assume anatomical IDs are implicitly handled or stored.
        # For the aggregator, we need to store them alongside features.
        # But the SequenceDataset expects features_dict and anatomical_ids_dict separately or combined.
        # I'll modify the cache to store a dict: {uid: {'features': ..., 'anat_ids': ...}}
        # But data.py expects separate dicts passed to constructor.
        # I will save a single dictionary where value is (features, anat_ids).

        # Actually data.py loads:
        # train_feats = np.load(..., allow_pickle=True).item()
        # And expects train_feats[uid] to be the array.
        # I will stick to saving features in one file.
        # I need to handle anatomical IDs. I'll save them in the same dict structure or a separate file.
        # To match data.py signature: SequenceDataset(..., features_dict, anatomical_ids_dict)
        # I will save the dict as {uid: features} and assume anat_ids are derived or I'll save a separate file.
        # Let's save a combined object and modify loading in main or here.
        # Since I cannot modify data.py, I must conform to what I wrote there.
        # data.py: train_ds = SequenceDataset(df_train, train_feats, train_anat)
        # It initializes train_anat = {} inside get_dataloaders. This implies data.py expects me to provide it
        # or it defaults to zeros.
        # To make it work properly, I should update get_dataloaders logic? No, I can't modify data.py.
        # Wait, get_dataloaders in data.py has:
        #   train_feats = np.load(Config.TRAIN_FEATURES_CACHE...).item()
        #   train_anat = {}
        # It seems I forgot to implement loading anat_ids in data.py's get_dataloaders.
        # However, SequenceDataset handles anat_ids being empty (defaults to zeros).
        # To enable the Transformer's anatomical embedding, I really need those IDs.
        # But since I can't change data.py, I will pack the anatomical IDs into the feature vector?
        # No, feature dim is fixed.
        # Workaround: I will save the features dict. The Transformer will work with empty anat_ids (all 0),
        # effectively disabling the explicit anatomical embedding but still using the learned positional embedding.
        # Given the constraints, this is the best path. The model will rely on sequence order.

        print(f"Processing {desc} ({len(df)} studies)...")

        # Debug subset
        if Config.DEBUG:
            df = df.iloc[:5]

        for _, row in tqdm(df.iterrows(), total=len(df)):
            uid = row["StudyInstanceUID"]
            img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # 1. Load Volume
            # Get all dcm files
            try:
                files = sorted(
                    [f for f in os.listdir(img_dir) if f.endswith(".dcm")],
                    key=lambda x: int(os.path.splitext(x)[0]),
                )
            except:
                continue

            if not files:
                continue

            # Load images into a stack
            # To save memory, we process in chunks or slice-by-slice
            # But we need 2.5D context (z-1, z, z+1).

            # We'll iterate slices.
            patient_feats = []

            # Pre-load all images? 500 slices * 512*512 * 4 bytes ~ 100MB. Feasible.
            vol_imgs = []
            for f in files:
                path = os.path.join(img_dir, f)
                img = read_dicom(
                    path,
                    Config.WINDOW_CENTER,
                    Config.WINDOW_WIDTH,
                    target_size=(256, 256),
                )  # Resize for Seg
                vol_imgs.append(img)
            vol_imgs = np.array(vol_imgs)  # (D, 256, 256)

            # 2. Predict Segmentation (Batch)
            # Create batches
            batch_size = 32
            masks_pred = []

            for i in range(0, len(vol_imgs), batch_size):
                batch = vol_imgs[i : i + batch_size]
                batch_tensor = (
                    torch.from_numpy(batch).unsqueeze(1).to(Config.DEVICE)
                )  # (B, 1, H, W)

                # Normalize (using same stats as training)
                batch_tensor = (batch_tensor - Config.PIXEL_MEAN) / Config.PIXEL_STD

                with torch.no_grad():
                    logits = seg_model(batch_tensor)
                    preds = torch.argmax(logits, dim=1)  # (B, H, W)
                masks_pred.append(preds.cpu().numpy())

            masks_vol = np.concatenate(masks_pred, axis=0)  # (D, 256, 256)

            # 3. Extract Features
            # We need high-res crops. Re-read DICOMs at 512x512 for Encoder?
            # Yes. EfficientNet expects 256x256 input, but cropped from 512.
            # Reading again is slow.
            # Optimization: Read once at 512, resize copy for Seg.

            # Let's refine the loop to do it slice by slice to save memory and IO.
            # But we need neighbors.

            # Re-implementation of loop for memory efficiency:
            # Read 512 image. Resize to 256 for Seg. Predict.
            # Determine Crop. Crop 512 image.
            # Buffer previous slices for 2.5D.

            # To keep code simple and fast enough:
            # We will use the already loaded 256 images for segmentation.
            # For encoding, we ideally want crops from 512.
            # If we crop from 256, we lose resolution.
            # Given constraints, I will reload the specific crop area from 512 or just read the file again.
            # Reading 500 files twice is okay.

            patient_feats = []

            # Prepare batch for encoder
            enc_batch = []

            for z in range(len(files)):
                # Neighbors
                indices = [max(0, z - 1), z, min(len(files) - 1, z + 1)]

                # Get Mask for current z
                mask_slice = masks_vol[z]  # (256, 256)

                # Determine Center from mask
                # mask classes: 0=Bg, 1-7=C1-C7
                # If any bone (class > 0)
                ys, xs = np.where(mask_slice > 0)
                if len(ys) > 0:
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    # Scale center back to 512 (since mask is 256)
                    cy, cx = cy * 2, cx * 2
                else:
                    cy, cx = 256, 256

                # Load 3 slices at 512
                # This is IO heavy.
                # Optimization: Cache the last loaded slices.
                # But for simplicity:
                slice_imgs = []
                for idx in indices:
                    img_512 = read_dicom(
                        os.path.join(img_dir, files[idx]),
                        Config.WINDOW_CENTER,
                        Config.WINDOW_WIDTH,
                    )
                    slice_imgs.append(img_512)

                img_rgb = np.stack(slice_imgs, axis=-1)  # (512, 512, 3)

                # Create Mask Channel (upsample mask to 512)
                mask_bin = (mask_slice > 0).astype(np.float32)
                mask_bin_512 = cv2.resize(
                    mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST
                )

                # Crop 256x256 around cy, cx
                crop_h, crop_w = Config.ENC_IMAGE_SIZE
                x1 = max(0, cx - crop_w // 2)
                y1 = max(0, cy - crop_h // 2)
                x2 = min(512, x1 + crop_w)
                y2 = min(512, y1 + crop_h)

                # Adjust
                if x2 - x1 < crop_w:
                    x1 = max(0, 512 - crop_w)
                    x2 = min(512, x1 + crop_w)
                if y2 - y1 < crop_h:
                    y1 = max(0, 512 - crop_h)
                    y2 = min(512, y1 + crop_h)

                img_crop = img_rgb[y1:y2, x1:x2, :]
                mask_crop = mask_bin_512[y1:y2, x1:x2]

                # Concat
                inp = np.concatenate(
                    [img_crop, mask_crop[:, :, np.newaxis]], axis=-1
                )  # (256, 256, 4)

                # Transform
                # Albumentations expects image
                aug = enc_transform(image=inp)
                inp_tensor = aug["image"]  # (4, 256, 256)

                enc_batch.append(inp_tensor)

                # Run batch if full
                if len(enc_batch) >= Config.ENC_BATCH_SIZE:
                    batch_t = torch.stack(enc_batch).to(Config.DEVICE)
                    with torch.no_grad():
                        feats = enc_model(batch_t)
                    patient_feats.append(feats.cpu().numpy())
                    enc_batch = []

            # Process remaining
            if enc_batch:
                batch_t = torch.stack(enc_batch).to(Config.DEVICE)
                with torch.no_grad():
                    feats = enc_model(batch_t)
                patient_feats.append(feats.cpu().numpy())

            if patient_feats:
                full_feats = np.concatenate(patient_feats, axis=0)  # (D, Dim)
                features_dict[uid] = full_feats
            else:
                # Fallback
                features_dict[uid] = np.zeros(
                    (1, Config.ENC_FEATURE_DIM), dtype=np.float32
                )

        # Save
        np.save(cache_path, features_dict)
        print(f"Saved features to {cache_path}")

    # Process all splits
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    process_subset(df_train, Config.TRAIN_FEATURES_CACHE, "Train")
    process_subset(df_val, Config.VAL_FEATURES_CACHE, "Validation")
    process_subset(df_test, Config.TEST_FEATURES_CACHE, "Test")

    del seg_model, enc_model
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# Stage 3: Aggregator
# -----------------------------------------------------------------------------


def train_aggregator(load_cached_data=True):
    """
    Trains the Transformer Aggregator.
    """
    print("\n" + "=" * 40)
    print("Stage 3: Training Aggregator")
    print("=" * 40)

    # Ensure features exist
    if not os.path.exists(Config.TRAIN_FEATURES_CACHE):
        print("Features not found. Please run extract_features first.")
        return

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Note: get_dataloaders for aggregator will load the cached features internally
    train_loader, val_loader = get_dataloaders("aggregator", df_train, df_val)

    model = AnatomicalTransformer().to(Config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.AGG_LR, weight_decay=Config.AGG_WEIGHT_DECAY
    )
    early_stopping = EarlyStopping(patience=5, mode="min")

    for epoch in range(Config.AGG_EPOCHS):
        model.train()
        train_loss = AverageMeter()

        for feats, anat_ids, mask, targets in train_loader:
            feats = feats.to(Config.DEVICE)
            anat_ids = anat_ids.to(Config.DEVICE)
            mask = mask.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()
            logits = model(feats, anat_ids, mask)
            loss = weighted_bce_loss(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), feats.size(0))

        model.eval()
        val_loss = AverageMeter()
        val_metric = AverageMeter()

        with torch.no_grad():
            for feats, anat_ids, mask, targets in val_loader:
                feats = feats.to(Config.DEVICE)
                anat_ids = anat_ids.to(Config.DEVICE)
                mask = mask.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                logits = model(feats, anat_ids, mask)
                loss = weighted_bce_loss(logits, targets)
                val_loss.update(loss.item(), feats.size(0))

                # Metric
                probs = torch.sigmoid(logits)
                metric = weighted_log_loss(targets, probs)
                val_metric.update(metric, feats.size(0))

        print(
            f"Epoch {epoch+1}/{Config.AGG_EPOCHS} | "
            f"Train Loss: {train_loss.avg:.8f} | "
            f"Val Loss: {val_loss.avg:.8f} | "
            f"Val Metric: {val_metric.avg:.8f}"
        )

        if early_stopping(val_loss.avg, model, optimizer, epoch, Config.AGG_MODEL_PATH):
            print("  [Checkpoint Saved]")

        if early_stopping.early_stop:
            break

    # Generate Submission
    generate_submission(model)


def generate_submission(model):
    print("\nGenerating Submission...")
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load test features
    test_feats_dict = np.load(Config.TEST_FEATURES_CACHE, allow_pickle=True).item()

    # Prepare results
    results = []

    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            uid = row["StudyInstanceUID"]

            if uid in test_feats_dict:
                feats = test_feats_dict[uid]
            else:
                feats = np.zeros((10, Config.ENC_FEATURE_DIM), dtype=np.float32)

            # Prepare tensor (Batch=1)
            # Pad/Truncate logic duplicated from Dataset for simplicity
            seq_len = feats.shape[0]
            max_len = Config.AGG_MAX_SEQ_LEN

            if seq_len > max_len:
                start = (seq_len - max_len) // 2
                feats = feats[start : start + max_len]
                mask = np.ones(max_len)
            else:
                pad_len = max_len - seq_len
                feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
                mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])

            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(Config.DEVICE)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(Config.DEVICE)
            anat_ids_t = torch.zeros((1, max_len), dtype=torch.long).to(
                Config.DEVICE
            )  # Dummy IDs

            logits = model(feats_t, anat_ids_t, mask_t)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

            # Map to submission format
            # Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall
            targets = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

            for i, t in enumerate(targets):
                results.append({"row_id": f"{uid}_{t}", "fractured": probs[i]})

    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
