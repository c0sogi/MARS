import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

from library.config import Config
from library.utils import setup_logger
from library.data import load_dicom_array
from library.models import UNetLocalizer, DualStreamEncoder, SpinalGraphAggregator


class InferencePipeline:
    def __init__(self):
        self.logger = setup_logger("Inference")
        self.device = Config.DEVICE

        # Initialize Models
        self.logger.info("Initializing models...")
        self.seg_model = UNetLocalizer(n_classes=8).to(self.device)
        self.enc_model = DualStreamEncoder(feature_dim=1280).to(self.device)
        self.agg_model = SpinalGraphAggregator(
            input_dim=1280,
            hidden_dim=Config.GRU_HIDDEN_DIM,
            gcn_dim=Config.GCN_HIDDEN_DIM,
        ).to(self.device)

        # Load Checkpoints
        self._load_checkpoint(self.seg_model, "stage1_unet.pth")
        self._load_checkpoint(self.enc_model, "stage2_encoder.pth")
        self._load_checkpoint(self.agg_model, "fracture_aggregator.pth")

        # Set to Eval Mode
        self.seg_model.eval()
        self.enc_model.eval()
        self.agg_model.eval()

    def _load_checkpoint(self, model, filename):
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.logger.info(f"Loaded checkpoint: {path}")
        else:
            self.logger.warning(
                f"Checkpoint not found: {path}. Using random initialization."
            )

    def run_inference(self):
        self.logger.info("Starting Inference Pipeline...")

        # Load Test Metadata
        if not os.path.exists(Config.TEST_METADATA_PATH):
            self.logger.error("Test metadata not found.")
            return

        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        study_uids = test_df["StudyInstanceUID"].unique()

        results = []

        # Process each study
        for uid in tqdm(study_uids, desc="Processing Studies"):
            try:
                # 1. Get Data
                img_dir = os.path.join(Config.TEST_IMAGES_DIR, uid)
                # Fallback to train dir if test dir doesn't exist (for debugging/validation on train set)
                if not os.path.exists(img_dir):
                    img_dir = os.path.join(Config.TRAIN_IMAGES_DIR, uid)

                if not os.path.exists(img_dir):
                    self.logger.warning(
                        f"Image directory not found for {uid}. Skipping."
                    )
                    self._append_dummy_prediction(results, uid)
                    continue

                slice_files = sorted(
                    glob.glob(os.path.join(img_dir, "*.dcm")),
                    key=lambda x: int(os.path.basename(x).replace(".dcm", "")),
                )

                if not slice_files:
                    self._append_dummy_prediction(results, uid)
                    continue

                # 2. Extract Features (Stage 1 & 2)
                study_feats, study_probs = self._process_study_slices(slice_files)

                # 3. Aggregate (Stage 3)
                if study_feats is not None:
                    preds = self._predict_patient(study_feats, study_probs)
                    self._append_prediction(results, uid, preds)
                else:
                    self._append_dummy_prediction(results, uid)

            except Exception as e:
                self.logger.error(f"Error processing {uid}: {e}")
                self._append_dummy_prediction(results, uid)

        # Save Submission
        submission_df = pd.DataFrame(results, columns=["row_id", "fractured"])
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        self.logger.info(f"Submission saved to {sub_path}")

    def _process_study_slices(self, slice_files):
        """
        Runs Stage 1 (Localization) and Stage 2 (Encoding) on all slices of a study.
        Returns:
            features: (Seq_Len, 1280)
            probs: (Seq_Len, 8)
        """
        batch_size = 16
        all_feats = []
        all_probs = []

        # Pre-load all images to avoid IO bottlenecks during batching if memory permits
        # For very large studies, we might need to do this inside the loop, but 1500 slices * 512x512 is heavy.
        # We will load in batches.

        for i in range(0, len(slice_files), batch_size):
            batch_paths = slice_files[i : i + batch_size]

            # Load Original Images (512x512)
            orig_imgs = [
                load_dicom_array(f, size=Config.IMG_SIZE_ORIG) for f in batch_paths
            ]
            orig_imgs_np = np.array(orig_imgs)  # (B, 512, 512)

            # --- Stage 1: Localization ---
            # Resize to 256 for UNet
            imgs_seg = []
            for img in orig_imgs_np:
                resized = cv2.resize(
                    img, (Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL)
                )
                imgs_seg.append(resized)

            imgs_seg = np.array(imgs_seg)
            imgs_seg = (imgs_seg - Config.PIXEL_MEAN) / Config.PIXEL_STD
            imgs_seg_t = (
                torch.tensor(imgs_seg, dtype=torch.float32).unsqueeze(1).to(self.device)
            )

            with torch.no_grad():
                logits_seg = self.seg_model(imgs_seg_t)
                probs_seg = F.softmax(logits_seg, dim=1)  # (B, 8, 256, 256)

            # Anatomical Probs (Global Average Pooling)
            anat_probs = probs_seg.mean(dim=(2, 3)).cpu().numpy()
            all_probs.append(anat_probs)

            # --- Stage 2: Encoding ---
            # Calculate Crop Centers
            bone_mask = probs_seg[:, 1:, :, :].sum(dim=1)  # (B, 256, 256)

            local_batch = []
            global_batch = []

            for b in range(len(batch_paths)):
                # Global Input (already resized and normalized in imgs_seg)
                global_batch.append(imgs_seg[b])

                # Local Crop
                mask_b = bone_mask[b].cpu().numpy()
                M = cv2.moments(mask_b)
                if M["m00"] > 0:
                    cx_model = int(M["m10"] / M["m00"])
                    cy_model = int(M["m01"] / M["m00"])
                    scale = Config.IMG_SIZE_ORIG / Config.IMG_SIZE_MODEL
                    cx = int(cx_model * scale)
                    cy = int(cy_model * scale)
                else:
                    cx, cy = Config.IMG_SIZE_ORIG // 2, Config.IMG_SIZE_ORIG // 2

                # Crop from Original 512
                crop_size = Config.IMG_SIZE_MODEL
                half = crop_size // 2
                start_x = int(np.clip(cx - half, 0, Config.IMG_SIZE_ORIG - crop_size))
                start_y = int(np.clip(cy - half, 0, Config.IMG_SIZE_ORIG - crop_size))

                l_img = orig_imgs_np[
                    b, start_y : start_y + crop_size, start_x : start_x + crop_size
                ]

                # Handle edge cases where crop might be smaller (should be handled by clip, but safety check)
                if l_img.shape != (crop_size, crop_size):
                    l_img = cv2.resize(l_img, (crop_size, crop_size))

                l_img = (l_img - Config.PIXEL_MEAN) / Config.PIXEL_STD

                # Local Mask Heuristic (thresholding local image)
                l_mask = (l_img > 0.2).astype(np.float32)

                # Stack: (2, 256, 256)
                l_combined = np.stack([l_img, l_mask], axis=0)
                local_batch.append(l_combined)

            local_t = torch.tensor(np.array(local_batch), dtype=torch.float32).to(
                self.device
            )
            global_t = (
                torch.tensor(np.array(global_batch), dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )

            with torch.no_grad():
                feats = self.enc_model(local_t, global_t)  # (B, 1280)
                all_feats.append(feats.cpu().numpy())

        if not all_feats:
            return None, None

        full_feats = np.concatenate(all_feats, axis=0)
        full_probs = np.concatenate(all_probs, axis=0)

        return full_feats, full_probs

    def _predict_patient(self, features, probs):
        """
        Runs Stage 3 (Aggregation) on the sequence.
        """
        # Prepare inputs: (Batch=1, Seq_Len, Dim)
        feats_t = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        probs_t = torch.tensor(probs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vert_probs, patient_prob = self.agg_model(feats_t, probs_t)

        # Extract results
        # vert_probs: (1, 7) -> C1..C7
        # patient_prob: (1, 1)

        return {
            "C1": vert_probs[0, 0].item(),
            "C2": vert_probs[0, 1].item(),
            "C3": vert_probs[0, 2].item(),
            "C4": vert_probs[0, 3].item(),
            "C5": vert_probs[0, 4].item(),
            "C6": vert_probs[0, 5].item(),
            "C7": vert_probs[0, 6].item(),
            "patient_overall": patient_prob[0, 0].item(),
        }

    def _append_prediction(self, results, uid, preds):
        for k, v in preds.items():
            results.append([f"{uid}_{k}", v])

    def _append_dummy_prediction(self, results, uid):
        # Default low probability if failed
        default_prob = 0.05
        # Patient overall slightly higher as it's an 'any' logic
        default_patient = 0.1

        for i in range(1, 8):
            results.append([f"{uid}_C{i}", default_prob])
        results.append([f"{uid}_patient_overall", default_patient])
