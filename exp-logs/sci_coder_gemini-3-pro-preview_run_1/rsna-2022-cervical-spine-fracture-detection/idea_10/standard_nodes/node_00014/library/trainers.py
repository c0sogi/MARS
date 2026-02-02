import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import (
    get_logger,
    save_checkpoint,
    load_checkpoint,
    calculate_weighted_log_loss,
)
from library.losses import DiceBCELoss, UnweightedBCELoss, CompetitionWeightedLoss
from library.models import AnatomicalSegmentor, FractureEncoder, HCHRNAggregator
from library.data import (
    SegmentationDataset,
    SliceClassificationDataset,
    SequenceDataset,
    collate_fn_sequence,
    load_dicom_slice,
)


class FractureDetectionTrainer:
    """
    Orchestrates the training and inference of the 3-stage HCH-RN pipeline.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.logger = get_logger("trainer")

        # Models
        self.segmentor = None
        self.encoder = None
        self.aggregator = None

    def _get_optimizer(self, model, lr):
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)

    def train_segmentor(self, train_df, val_df):
        """
        Stage 1: Train the Multi-Task Anatomical Segmentor (U-Net).
        """
        self.logger.info("Starting Stage 1: Segmentation Training")

        # Data
        train_dataset = SegmentationDataset(train_df, phase="train")
        val_dataset = SegmentationDataset(val_df, phase="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_SEG_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.TRAIN_SEG_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Loss
        self.segmentor = AnatomicalSegmentor(pretrained=True).to(self.device)
        criterion = DiceBCELoss()
        optimizer = self._get_optimizer(self.segmentor, Config.TRAIN_SEG_LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.TRAIN_SEG_EPOCHS):
            # Training
            self.segmentor.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                mask_logits, _, _ = self.segmentor(images)
                loss = criterion(mask_logits, masks.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            self.segmentor.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    mask_logits, _, _ = self.segmentor(images)
                    loss = criterion(mask_logits, masks.float())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{Config.TRAIN_SEG_EPOCHS} - Seg Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

            # Checkpoint & Early Stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "state_dict": self.segmentor.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                    },
                    os.path.join(Config.CHECKPOINT_DIR, "stage1_segmentor.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    self.logger.info("Early stopping triggered for Stage 1.")
                    break

        # Load best model
        load_checkpoint(
            os.path.join(Config.CHECKPOINT_DIR, "stage1_segmentor.pth"),
            self.segmentor,
            device=self.device,
        )

    def train_encoder(self, train_df, val_df):
        """
        Stage 2: Train the Mask-Guided High-Resolution Encoder (2.5D CNN).
        """
        self.logger.info("Starting Stage 2: Encoder Training")

        # Data
        train_dataset = SliceClassificationDataset(train_df, phase="train")
        val_dataset = SliceClassificationDataset(val_df, phase="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_CLS_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.TRAIN_CLS_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model & Loss
        self.encoder = FractureEncoder(pretrained=True).to(self.device)
        criterion = UnweightedBCELoss()
        optimizer = self._get_optimizer(self.encoder, Config.TRAIN_CLS_LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.TRAIN_CLS_EPOCHS):
            # Training
            self.encoder.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                logits = self.encoder(images)
                # Logits are (B, 1280), we need to project to scalar for BCE training?
                # Wait, FractureEncoder outputs (B, 1280) embedding.
                # To train it as a classifier, we need a temporary head or modification.
                # However, provided FractureEncoder in models.py outputs embeddings.
                # We need to attach a linear head for training purposes.
                # Since we cannot modify models.py, we attach a temporary head here.
                pass

            # NOTE: The provided FractureEncoder returns embeddings (1280).
            # To train it, we wrap it with a classification head.
            # We define a wrapper class locally.

            break  # Breaking because we need to redefine the loop with the wrapper.

        # Redefine logic with wrapper
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.head = nn.Linear(Config.CLS_EMBED_DIM, 1)

            def forward(self, x):
                feat = self.encoder(x)
                return self.head(feat).squeeze(-1)

        wrapper = EncoderWrapper(self.encoder).to(self.device)
        optimizer = self._get_optimizer(wrapper, Config.TRAIN_CLS_LR)

        for epoch in range(Config.TRAIN_CLS_EPOCHS):
            wrapper.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = wrapper(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            wrapper.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = wrapper(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{Config.TRAIN_CLS_EPOCHS} - Cls Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "state_dict": self.encoder.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                    },
                    os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    self.logger.info("Early stopping triggered for Stage 2.")
                    break

        # Load best encoder state
        load_checkpoint(
            os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth"),
            self.encoder,
            device=self.device,
        )

    def extract_features_and_cache(
        self, metadata_df, split_name, load_cached_data=True
    ):
        """
        Runs inference with Stage 1 & 2 models to generate features for Stage 3.
        Caches results to disk.
        """
        # Determine cache path
        if split_name == "train":
            cache_path = Config.CACHE_FEATURES_TRAIN
        elif split_name == "val":
            cache_path = Config.CACHE_FEATURES_VAL
        else:
            cache_path = Config.CACHE_FEATURES_TEST

        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading cached features for {split_name} from {cache_path}"
            )
            return

        self.logger.info(f"Extracting features for {split_name}...")

        # Ensure models are loaded
        if self.segmentor is None:
            self.segmentor = AnatomicalSegmentor(pretrained=False).to(self.device)
            try:
                load_checkpoint(
                    os.path.join(Config.CHECKPOINT_DIR, "stage1_segmentor.pth"),
                    self.segmentor,
                    device=self.device,
                )
            except:
                self.logger.warning(
                    "Could not load Stage 1 checkpoint. Using random weights (debug?)."
                )

        if self.encoder is None:
            self.encoder = FractureEncoder(pretrained=False).to(self.device)
            try:
                load_checkpoint(
                    os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth"),
                    self.encoder,
                    device=self.device,
                )
            except:
                self.logger.warning(
                    "Could not load Stage 2 checkpoint. Using random weights (debug?)."
                )

        self.segmentor.eval()
        self.encoder.eval()

        # Transforms
        seg_transform = A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE_SEG[0], width=Config.IMG_SIZE_SEG[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        )

        cls_resize = A.Resize(
            height=Config.IMG_SIZE_CLS[0], width=Config.IMG_SIZE_CLS[1]
        )
        cls_norm = A.Normalize(mean=(0.5,), std=(0.5,))  # Applied after stacking
        to_tensor = ToTensorV2()

        feature_data = {}

        # Iterate over studies
        uids = metadata_df["StudyInstanceUID"].unique()
        if Config.DEBUG:
            uids = uids[: Config.DEBUG_SAMPLE_SIZE]

        for uid in uids:
            # Find image directory
            row = metadata_df[metadata_df["StudyInstanceUID"] == uid].iloc[0]
            image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # List and sort DICOMs
            dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
            if not dcm_files:
                continue
            # Sort numerically
            dcm_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            # Load volume
            # To save memory, we might process in batches, but for simplicity and speed on A100:
            # Load all slice raw data
            slices = []
            for f in dcm_files:
                slices.append(load_dicom_slice(f))  # Returns 512x512 numpy

            num_slices = len(slices)

            # ---------------------------
            # Stage 1 Inference
            # ---------------------------
            # Prepare batch
            seg_batch = []
            for s in slices:
                aug = seg_transform(image=s)
                seg_batch.append(aug["image"])

            seg_batch = torch.stack(seg_batch).to(self.device)  # (D, 1, 256, 256)

            # Inference in chunks if D is large
            chunk_size = 32
            global_contexts = []
            anatomical_probs = []
            masks = []

            with torch.no_grad():
                for i in range(0, num_slices, chunk_size):
                    batch = seg_batch[i : i + chunk_size]
                    m_logits, a_logits, g_ctx = self.segmentor(batch)

                    global_contexts.append(g_ctx.cpu())
                    anatomical_probs.append(torch.softmax(a_logits, dim=1).cpu())
                    masks.append(torch.sigmoid(m_logits).cpu())

            global_contexts = torch.cat(global_contexts)  # (D, 1280)
            anatomical_probs = torch.cat(anatomical_probs)  # (D, 8)
            masks = torch.cat(masks)  # (D, 1, 256, 256)

            # ---------------------------
            # Stage 2 Inference
            # ---------------------------
            # Prepare inputs: 3-slice stack + mask
            # Mask needs to be resized to CLS size if different (here both 256)
            # Slices need to be resized to CLS size

            cls_inputs = []

            # Resize raw slices to CLS size first to avoid repeated resize
            resized_slices = [cv2.resize(s, Config.IMG_SIZE_CLS) for s in slices]

            for i in range(num_slices):
                # 3-slice stack
                # Handle boundaries by clamping
                idx_prev = max(0, i - 1)
                idx_curr = i
                idx_next = min(num_slices - 1, i + 1)

                s_prev = resized_slices[idx_prev]
                s_curr = resized_slices[idx_curr]
                s_next = resized_slices[idx_next]

                stack = np.stack([s_prev, s_curr, s_next], axis=-1)  # (256, 256, 3)

                # Mask (from Stage 1)
                # mask is (1, 256, 256), convert to (256, 256, 1)
                m = masks[i].permute(1, 2, 0).numpy()
                m = (m > 0.5).astype(np.float32)  # Binary

                # Combine
                combined = np.concatenate([stack, m], axis=-1)  # (256, 256, 4)

                # Normalize manually (-1 to 1)
                combined = (combined - 0.5) / 0.5

                # To Tensor
                tensor = (
                    torch.from_numpy(combined).permute(2, 0, 1).float()
                )  # (4, 256, 256)
                cls_inputs.append(tensor)

            cls_inputs = torch.stack(cls_inputs).to(self.device)

            local_features = []
            with torch.no_grad():
                for i in range(0, num_slices, chunk_size):
                    batch = cls_inputs[i : i + chunk_size]
                    feats = self.encoder(batch)
                    local_features.append(feats.cpu())

            local_features = torch.cat(local_features)  # (D, 1280)

            # ---------------------------
            # Aggregate & Store
            # ---------------------------
            # Concatenate Local + Global
            # SequenceDataset expects "features" to be the input to RNN (excluding probs if separate)
            # But SequenceDataset logic: combined = cat([features, probs])
            # So here we save features = cat(local, global)

            combined_feats = torch.cat([local_features, global_contexts], dim=1).numpy()
            probs_np = anatomical_probs.numpy()

            # Get targets if available
            targets = np.zeros(8, dtype=np.float32)
            if split_name != "test":
                target_cols = [
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                    "C7",
                    "patient_overall",
                ]
                targets = row[target_cols].values.astype(np.float32)

            feature_data[uid] = {
                "features": combined_feats.astype(np.float32),
                "probs": probs_np.astype(np.float32),
                "targets": targets,
            }

        # Save to cache
        np.save(cache_path, feature_data)
        self.logger.info(f"Saved extracted features to {cache_path}")

    def train_aggregator(self, train_df, val_df):
        """
        Stage 3: Train the Hybrid-Feature Recurrent Aggregator (Bi-GRU).
        """
        self.logger.info("Starting Stage 3: Aggregator Training")

        # Ensure features are extracted
        self.extract_features_and_cache(train_df, "train")
        self.extract_features_and_cache(val_df, "val")

        # Data
        train_dataset = SequenceDataset(train_df, phase="train")
        val_dataset = SequenceDataset(val_df, phase="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_RNN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn_sequence,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.TRAIN_RNN_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn_sequence,
        )

        # Model & Loss
        self.aggregator = HCHRNAggregator().to(self.device)
        criterion = CompetitionWeightedLoss()
        optimizer = self._get_optimizer(self.aggregator, Config.TRAIN_RNN_LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.TRAIN_RNN_EPOCHS):
            # Training
            self.aggregator.train()
            train_loss = 0.0
            for features, probs, targets, _ in train_loader:
                features, probs, targets = (
                    features.to(self.device),
                    probs.to(self.device),
                    targets.to(self.device),
                )

                optimizer.zero_grad()
                # Features in loader are already concatenated [Local, Global, Probs]
                # But model forward takes (features, probs).
                # Model expects features to be the sequence input.
                # In SequenceDataset, we concatenated them.
                # The model definition: forward(features, anatomical_probs)
                # It passes 'features' to GRU. The GRU input dim is Config.RNN_INPUT_DIM.
                # Config.RNN_INPUT_DIM = 1280+1280+8.
                # So we pass the full concatenated tensor as the first arg.

                logits = self.aggregator(features, probs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            self.aggregator.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for features, probs, targets, _ in val_loader:
                    features, probs, targets = (
                        features.to(self.device),
                        probs.to(self.device),
                        targets.to(self.device),
                    )
                    logits = self.aggregator(features, probs)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()

                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)

            # Calculate exact competition metric on validation set
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            metric_loss = calculate_weighted_log_loss(y_pred, y_true)

            scheduler.step(avg_val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{Config.TRAIN_RNN_EPOCHS} - RNN Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Metric: {metric_loss:.6f}"
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "state_dict": self.aggregator.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                    },
                    os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    self.logger.info("Early stopping triggered for Stage 3.")
                    break

        # Load best model
        load_checkpoint(
            os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth"),
            self.aggregator,
            device=self.device,
        )

    def predict_test_set(self, test_df):
        """
        Runs the full inference pipeline on the test set and generates submission.csv.
        """
        self.logger.info("Starting Test Set Prediction")

        # 1. Extract Features
        self.extract_features_and_cache(test_df, "test")

        # 2. Load Aggregator
        if self.aggregator is None:
            self.aggregator = HCHRNAggregator().to(self.device)
            load_checkpoint(
                os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth"),
                self.aggregator,
                device=self.device,
            )
        self.aggregator.eval()

        # 3. Inference
        test_dataset = SequenceDataset(test_df, phase="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.TRAIN_RNN_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn_sequence,
        )

        results = []
        cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        with torch.no_grad():
            for features, probs, uids in test_loader:
                features, probs = features.to(self.device), probs.to(self.device)
                logits = self.aggregator(features, probs)
                preds = torch.sigmoid(logits).cpu().numpy()

                for i, uid in enumerate(uids):
                    p = preds[i]
                    for j, col in enumerate(cols):
                        results.append({"row_id": f"{uid}_{col}", "fractured": p[j]})

        # 4. Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
