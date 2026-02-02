import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import MultiTaskEfficientDet
from library.dataset import CovidDataset
from library.utils import collate_fn


class InferenceRunner:
    """
    Handles inference on the test set and submission generation.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

    def predict(self):
        """
        Loads the best model, runs inference on the test set, and generates the submission CSV.
        """
        # 1. Load Model
        print(f"Loading model from {self.config.BEST_MODEL_PATH}...")
        if not os.path.exists(self.config.BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.config.BEST_MODEL_PATH}"
            )

        model = MultiTaskEfficientDet(self.config)
        state_dict = torch.load(self.config.BEST_MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # 2. Load Test Data
        # load_cached_data=True ensures we use the cache logic defined in dataset.py
        test_dataset = CovidDataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        # 3. Run Inference
        study_preds = {}  # Map: study_id -> list of probability arrays (one per image)
        image_preds = {}  # Map: image_id -> prediction string

        print("Running inference on test set...")
        with torch.no_grad():
            for images, targets, ids in test_loader:
                images = images.to(self.device)

                # Forward pass
                # In eval mode, model returns list of dicts with 'boxes', 'scores', 'labels', 'study_probs'
                detections = model(images)

                for i, det in enumerate(detections):
                    img_id = ids[i]
                    # Dataset provides study_id in the target dict even for test set
                    study_id = targets[i]["study_id"]

                    # Extract outputs
                    probs = det["study_probs"].cpu().numpy()
                    boxes = det["boxes"].cpu().numpy()
                    scores = det["scores"].cpu().numpy()

                    # --- Study Level Aggregation ---
                    if study_id not in study_preds:
                        study_preds[study_id] = []
                    study_preds[study_id].append(probs)

                    # --- Image Level Logic ---
                    # Index 0 corresponds to "Negative for Pneumonia"
                    neg_prob = probs[0]

                    # If the model is confident this is negative, predict 'none'
                    if neg_prob > self.config.STUDY_CONF_THRESHOLD:
                        pred_str = "none 1 0 0 1 1"
                    else:
                        box_strings = []
                        for b, s in zip(boxes, scores):
                            # Format: opacity confidence xmin ymin xmax ymax
                            box_strings.append(
                                f"opacity {s:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                            )

                        if not box_strings:
                            # If no boxes passed the detection threshold/NMS
                            pred_str = "none 1 0 0 1 1"
                        else:
                            pred_str = " ".join(box_strings)

                    image_preds[img_id] = pred_str

        # 4. Format Submission
        study_rows = []
        image_rows = []

        # Mapping indices to submission strings
        # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
        idx_to_label = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}

        # Process Study Predictions
        for study_id, probs_list in study_preds.items():
            # Average probabilities across all images in the study
            avg_probs = np.mean(probs_list, axis=0)

            # Select the class with the highest probability
            idx = np.argmax(avg_probs)
            label = idx_to_label[idx]
            conf = avg_probs[idx]

            # Format: class_id confidence 0 0 1 1
            pred_string = f"{label} {conf:.4f} 0 0 1 1"
            study_rows.append(
                {"id": f"{study_id}_study", "PredictionString": pred_string}
            )

        # Process Image Predictions
        for img_id, pred_str in image_preds.items():
            image_rows.append({"id": f"{img_id}_image", "PredictionString": pred_str})

        # Combine
        df_study = pd.DataFrame(study_rows)
        df_image = pd.DataFrame(image_rows)

        # Handle case where one might be empty (unlikely given dataset structure)
        if df_study.empty and df_image.empty:
            df_sub = pd.DataFrame(columns=["id", "PredictionString"])
        else:
            df_sub = pd.concat([df_study, df_image], ignore_index=True)

        # 5. Save
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
