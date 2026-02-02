import os
import gc
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import glob

from library.config import Config
from library.utils import (
    read_dicom,
    load_checkpoint,
    seed_everything,
    save_to_cache,
    load_from_cache,
)
from library.models import UNetLocalizer, FractureEncoder, AnatomicalTransformer
from library.data import get_transforms


class InferencePipeline:
    """
    End-to-end inference pipeline for the Anatomically-Guided Transformer solution.
    Handles:
    1. Feature Extraction (Stage 1 Seg + Stage 2 Enc) with caching.
    2. Sequence Aggregation (Stage 3 Transformer).
    3. Submission generation.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.test_metadata = pd.read_csv(Config.TEST_METADATA_PATH)

        # Cache file for inference data (features + anatomical IDs)
        self.cache_filename = "test_inference_data.npy"

        seed_everything(Config.SEED)

    def _load_stage1_model(self):
        model = UNetLocalizer(num_classes=Config.SEG_NUM_CLASSES).to(self.device)
        load_checkpoint(model, None, Config.SEG_MODEL_PATH, device=self.device)
        model.eval()
        return model

    def _load_stage2_model(self):
        model = FractureEncoder().to(self.device)
        load_checkpoint(model, None, Config.ENC_MODEL_PATH, device=self.device)
        model.eval()
        return model

    def _load_stage3_model(self):
        model = AnatomicalTransformer().to(self.device)
        load_checkpoint(model, None, Config.AGG_MODEL_PATH, device=self.device)
        model.eval()
        return model

    def extract_data(self, load_cached_data=True):
        """
        Runs Stage 1 (Segmentation) and Stage 2 (Encoder) on the test set.
        Extracts features and anatomical IDs for each study.
        Implements caching logic.
        """
        # 1. Check Cache
        cached_data = load_from_cache(self.cache_filename)
        if load_cached_data and cached_data is not None:
            print(f"Loaded cached inference data from {self.cache_filename}")
            return cached_data.item()

        print("Starting Feature Extraction (Stage 1 & 2)...")

        # 2. Load Models
        seg_model = self._load_stage1_model()
        enc_model = self._load_stage2_model()

        # Transforms
        seg_transform = get_transforms("segmentation", "val")
        enc_transform = get_transforms("classifier", "val")

        inference_data = {}  # {uid: {'features': np.array, 'anat_ids': np.array}}

        # 3. Process Studies
        # Debug option
        df = self.test_metadata
        if Config.DEBUG:
            df = df.iloc[:5]
            print("Running in DEBUG mode (5 samples)")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            uid = row["StudyInstanceUID"]
            img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # List DICOM files
            try:
                files = sorted(
                    [f for f in os.listdir(img_dir) if f.endswith(".dcm")],
                    key=lambda x: int(os.path.splitext(x)[0]),
                )
            except Exception as e:
                print(f"Error listing files for {uid}: {e}")
                continue

            if not files:
                continue

            num_slices = len(files)

            # --- Stage 1: Segmentation ---
            # Process in batches to save memory
            masks_vol = []

            # We need to read images. To optimize, we read once at 512,
            # resize copy to 256 for Seg, keep 512 for Enc.
            # Given memory constraints (220GB RAM is plenty, but GPU is 40GB),
            # we can load the whole volume into RAM (CPU) easily.

            vol_imgs_512 = []
            for f in files:
                path = os.path.join(img_dir, f)
                img = read_dicom(path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)
                vol_imgs_512.append(img)

            vol_imgs_512 = np.array(vol_imgs_512)  # (D, 512, 512)

            # Resize for Segmentation (D, 256, 256)
            vol_imgs_256 = []
            for i in range(num_slices):
                resized = cv2.resize(
                    vol_imgs_512[i],
                    Config.SEG_IMAGE_SIZE,
                    interpolation=cv2.INTER_LINEAR,
                )
                vol_imgs_256.append(resized)
            vol_imgs_256 = np.array(vol_imgs_256)

            # Batched Prediction
            batch_size = Config.SEG_BATCH_SIZE
            seg_preds = []

            with torch.no_grad():
                for i in range(0, num_slices, batch_size):
                    batch = vol_imgs_256[i : i + batch_size]
                    # Normalize
                    batch = (batch - Config.PIXEL_MEAN) / Config.PIXEL_STD
                    batch_t = (
                        torch.from_numpy(batch).unsqueeze(1).float().to(self.device)
                    )  # (B, 1, H, W)

                    logits = seg_model(batch_t)
                    preds = (
                        torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                    )  # (B, H, W)
                    seg_preds.append(preds)

            masks_vol = np.concatenate(seg_preds, axis=0)  # (D, 256, 256)

            # --- Stage 2: Encoding ---
            study_features = []
            study_anat_ids = []

            enc_batch = []

            for z in range(num_slices):
                # 1. Determine Anatomical ID for this slice
                # Mode of non-zero pixels
                mask_slice = masks_vol[z]
                flat_mask = mask_slice.flatten()
                # Filter background (0)
                bone_pixels = flat_mask[flat_mask > 0]
                if len(bone_pixels) > 0:
                    counts = np.bincount(bone_pixels, minlength=8)
                    anat_id = np.argmax(counts)
                else:
                    anat_id = 0
                study_anat_ids.append(anat_id)

                # 2. Prepare Encoder Input
                # Neighbors (z-1, z, z+1)
                indices = [max(0, z - 1), z, min(num_slices - 1, z + 1)]
                img_rgb = np.stack(
                    [vol_imgs_512[idx] for idx in indices], axis=-1
                )  # (512, 512, 3)

                # Mask Channel (Upsample mask to 512)
                mask_bin = (mask_slice > 0).astype(np.float32)
                mask_bin_512 = cv2.resize(
                    mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST
                )

                # Crop
                # Find center from mask
                ys, xs = np.where(mask_slice > 0)
                if len(ys) > 0:
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    cy, cx = cy * 2, cx * 2  # Scale to 512
                else:
                    cy, cx = 256, 256

                crop_h, crop_w = Config.ENC_IMAGE_SIZE
                x1 = max(0, cx - crop_w // 2)
                y1 = max(0, cy - crop_h // 2)
                x2 = min(512, x1 + crop_w)
                y2 = min(512, y1 + crop_h)

                # Boundary checks
                if x2 - x1 < crop_w:
                    x1 = max(0, 512 - crop_w)
                    x2 = min(512, x1 + crop_w)
                if y2 - y1 < crop_h:
                    y1 = max(0, 512 - crop_h)
                    y2 = min(512, y1 + crop_h)

                img_crop = img_rgb[y1:y2, x1:x2, :]
                mask_crop = mask_bin_512[y1:y2, x1:x2]

                # Concat (256, 256, 4)
                inp = np.concatenate([img_crop, mask_crop[:, :, np.newaxis]], axis=-1)

                # Transform
                aug = enc_transform(image=inp)
                inp_tensor = aug["image"]  # (4, 256, 256)

                enc_batch.append(inp_tensor)

                # Run Batch if full
                if len(enc_batch) >= Config.ENC_BATCH_SIZE:
                    batch_t = torch.stack(enc_batch).to(self.device)
                    with torch.no_grad():
                        feats = enc_model(batch_t)
                    study_features.append(feats.cpu().numpy())
                    enc_batch = []

            # Process remaining
            if enc_batch:
                batch_t = torch.stack(enc_batch).to(self.device)
                with torch.no_grad():
                    feats = enc_model(batch_t)
                study_features.append(feats.cpu().numpy())

            # Aggregate study results
            if study_features:
                full_feats = np.concatenate(study_features, axis=0)
                full_ids = np.array(study_anat_ids, dtype=np.int64)
                inference_data[uid] = {"features": full_feats, "anat_ids": full_ids}
            else:
                # Fallback for empty studies
                inference_data[uid] = {
                    "features": np.zeros((1, Config.ENC_FEATURE_DIM), dtype=np.float32),
                    "anat_ids": np.zeros((1,), dtype=np.int64),
                }

            # Explicit cleanup
            del vol_imgs_512, vol_imgs_256, masks_vol

        # 4. Save to Cache
        save_to_cache(inference_data, self.cache_filename)
        print(f"Saved inference data to {self.cache_filename}")

        # Cleanup models
        del seg_model, enc_model
        torch.cuda.empty_cache()
        gc.collect()

        return inference_data

    def predict(self, load_cached_data=True):
        """
        Main entry point.
        1. Gets data (features + anat_ids).
        2. Runs Stage 3 (Transformer).
        3. Generates submission.csv.
        """
        # 1. Get Data
        data_dict = self.extract_data(load_cached_data=load_cached_data)

        print("Starting Prediction (Stage 3)...")

        # 2. Load Model
        model = self._load_stage3_model()

        results = []
        targets = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        # 3. Iterate and Predict
        # We iterate over the test metadata to ensure we cover all required UIDs
        # even if extraction failed for some (though extract_data handles fallbacks).

        df = self.test_metadata
        if Config.DEBUG:
            df = df.iloc[:5]

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            uid = row["StudyInstanceUID"]

            if uid in data_dict:
                feats = data_dict[uid]["features"]
                anat_ids = data_dict[uid]["anat_ids"]
            else:
                # Fallback
                feats = np.zeros((10, Config.ENC_FEATURE_DIM), dtype=np.float32)
                anat_ids = np.zeros((10,), dtype=np.int64)

            # Pad / Truncate
            seq_len = feats.shape[0]
            max_len = Config.AGG_MAX_SEQ_LEN

            if seq_len > max_len:
                start = (seq_len - max_len) // 2
                feats = feats[start : start + max_len]
                anat_ids = anat_ids[start : start + max_len]
                mask = np.ones(max_len, dtype=np.float32)
            else:
                pad_len = max_len - seq_len
                feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
                anat_ids = np.pad(anat_ids, (0, pad_len), mode="constant")
                mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)]).astype(
                    np.float32
                )

            # To Tensor
            feats_t = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)
            anat_ids_t = torch.from_numpy(anat_ids).long().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                logits = model(feats_t, anat_ids_t, mask_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

            # Store results
            for i, t in enumerate(targets):
                results.append({"row_id": f"{uid}_{t}", "fractured": probs[i]})

        # 4. Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        return sub_df
