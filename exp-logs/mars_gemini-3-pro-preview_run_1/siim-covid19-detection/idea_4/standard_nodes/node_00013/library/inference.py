import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.model import ResNet18Unet
from library.utils import read_dicom, mask2box, seed_everything


class TestDataset(Dataset):
    """
    Dataset class for the Test set.
    Returns image tensors and metadata required for post-processing (original dims).
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read original image to get dims
        # We read with size=None to get original dims, then resize for model
        # Optimization: read_dicom can resize, but we need orig size.
        # pydicom is fast enough to read headers, but read_dicom reads pixels.
        # We will read full image, get size, then resize using albumentations or cv2.

        # Read full image
        image_full = read_dicom(file_path, size=None, fix_monochrome=True)
        h_orig, w_orig = image_full.shape[:2]

        # Resize for model input if not using Albumentations Resize (which we are)
        # But Albumentations is preferred for consistency
        image = image_full

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "study_id": row["study_id"],
            "image_id": row["image_id"],
            "h_orig": h_orig,
            "w_orig": w_orig,
        }


def get_test_transforms():
    return A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


class Predictor:
    """
    Handles inference, post-processing, and submission generation.
    """

    def __init__(self, model_path=None):
        self.device = Config.DEVICE
        self.model = ResNet18Unet()

        if model_path is None:
            model_path = Config.BEST_MODEL_PATH

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model path {model_path} not found. Using random weights (DEBUG mode)."
            )

        self.model.to(self.device)
        self.model.eval()

    def generate_submission(self):
        """
        Runs the full inference pipeline and saves submission.csv.
        """
        seed_everything(Config.SEED)

        # 1. Load Metadata
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        dataset = TestDataset(test_df, transforms=get_test_transforms())
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Running inference on {len(test_df)} images...")

        # Storage for aggregation
        # study_id -> list of logits
        study_logits_map = {}
        # image_id -> (mask_pred, h_orig, w_orig, study_id)
        image_data_map = {}

        # 2. Inference Loop
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                images = batch["image"].to(self.device)
                study_ids = batch["study_id"]
                image_ids = batch["image_id"]
                h_origs = batch["h_orig"].numpy()
                w_origs = batch["w_orig"].numpy()

                # Forward
                # Model returns (study_logits, mask_logits) in eval mode
                s_logits, m_logits = self.model(images)

                s_probs = torch.softmax(s_logits, dim=1).cpu().numpy()
                m_probs = torch.sigmoid(m_logits).cpu().numpy()  # (B, 1, H, W)

                for i in range(len(images)):
                    sid = study_ids[i]
                    iid = image_ids[i]

                    # Store study probs for aggregation
                    if sid not in study_logits_map:
                        study_logits_map[sid] = []
                    study_logits_map[sid].append(s_probs[i])

                    # Store image data
                    image_data_map[iid] = {
                        "mask": m_probs[i, 0],  # (H, W)
                        "h_orig": h_origs[i],
                        "w_orig": w_origs[i],
                        "study_id": sid,
                    }

        # 3. Post-Processing & String Generation
        results = []

        # Process Study Level First (Aggregation)
        study_predictions = {}  # sid -> (class_name, confidence)

        print("Aggregating study predictions...")
        for sid, probs_list in study_logits_map.items():
            # Mean pooling of probabilities across images in the study
            avg_probs = np.mean(np.stack(probs_list), axis=0)
            best_idx = np.argmax(avg_probs)
            confidence = avg_probs[best_idx]
            label_name = Config.CLASS_LABELS[best_idx]

            study_predictions[sid] = (label_name, confidence)

            # Format: "class_name confidence 0 0 1 1"
            pred_string = f"{label_name} {confidence:.6f} 0 0 1 1"
            results.append({"id": f"{sid}_study", "PredictionString": pred_string})

        # Process Image Level
        print("Generating image predictions...")
        for iid, data in image_data_map.items():
            sid = data["study_id"]
            study_label, _ = study_predictions[sid]

            # Gating Logic
            if Config.GATING_ENABLED and study_label == "Negative for Pneumonia":
                pred_string = "none 1 0 0 1 1"
            else:
                # Extract Boxes
                mask = data["mask"]
                h_orig = data["h_orig"]
                w_orig = data["w_orig"]

                # Threshold
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Get boxes in 512x512 space
                boxes_512 = mask2box(binary_mask)

                if not boxes_512:
                    pred_string = "none 1 0 0 1 1"
                else:
                    # Scale boxes to original dimensions
                    scale_x = w_orig / Config.IMG_SIZE
                    scale_y = h_orig / Config.IMG_SIZE

                    box_strings = []
                    for box in boxes_512:
                        x1, y1, x2, y2 = box

                        # Calculate confidence for this box
                        # Mean probability within the box region
                        box_region = mask[y1:y2, x1:x2]
                        if box_region.size > 0:
                            conf = np.mean(box_region)
                        else:
                            conf = 0.0

                        # Scale coordinates
                        x1_o = x1 * scale_x
                        y1_o = y1 * scale_y
                        x2_o = x2 * scale_x
                        y2_o = y2 * scale_y

                        box_strings.append(
                            f"opacity {conf:.4f} {x1_o:.4f} {y1_o:.4f} {x2_o:.4f} {y2_o:.4f}"
                        )

                    pred_string = " ".join(box_strings)

            results.append({"id": f"{iid}_image", "PredictionString": pred_string})

        # 4. Save Submission
        submission_df = pd.DataFrame(results)
        # Ensure column order
        submission_df = submission_df[["id", "PredictionString"]]

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
