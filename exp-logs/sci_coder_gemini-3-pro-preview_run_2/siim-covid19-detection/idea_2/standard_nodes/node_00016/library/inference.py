import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import SIIMDataset
from library.model import MultiTaskFasterRCNN
from library.utils import get_device, collate_fn


def generate_submission():
    """
    Generates the submission.csv file for the test set.
    Loads the trained model, performs inference, and formats the output
    according to the competition requirements.
    """
    print("Initializing Inference...")

    # 1. Setup
    device = get_device()
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Data
    # Batch size 1 is used to simplify the mapping of predictions to IDs
    test_dataset = SIIMDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 3. Load Model
    model = MultiTaskFasterRCNN()
    model.to(device)

    # Checkpoint loading logic: Prioritize current working dir, fallback to idea_1
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    fallback_path = os.path.join("./working/idea_1", "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    elif os.path.exists(fallback_path):
        print(f"Loading model from fallback {fallback_path}...")
        checkpoint = torch.load(fallback_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print(
            "Warning: No checkpoint found! Using random weights (predictions will be random)."
        )

    model.eval()

    # 4. Inference Loop
    # We collect raw predictions first to handle study-level aggregation
    study_predictions = {}  # study_id -> list of (probs, max_box_conf)
    image_predictions = []  # list of dicts for final dataframe

    # Class mapping for submission format
    class_names = ["negative", "typical", "indeterminate", "atypical"]

    print("Running inference on test set...")
    with torch.no_grad():
        for images, targets in test_loader:
            # Move to device
            images = list(img.to(device) for img in images)

            # Forward pass
            # detections: list of dicts {'boxes': ..., 'scores': ..., 'labels': ...}
            # global_logits: tensor of shape (B, 4)
            detections, global_logits = model(images)

            # Process each image in the batch
            for i in range(len(images)):
                # Metadata
                img_idx = targets[i]["image_id"].item()
                row = test_dataset.df.iloc[img_idx]
                image_id = row["image_id"]
                study_id = row["StudyInstanceUID"]

                # --- Process Global Head (Study Level) ---
                probs = torch.softmax(global_logits[i], dim=0).cpu().numpy()

                # --- Process Detections (Image Level) ---
                boxes = detections[i]["boxes"].cpu().numpy()
                scores = detections[i]["scores"].cpu().numpy()

                # Calculate max box confidence for ensemble logic
                max_box_conf = 0.0
                if len(scores) > 0:
                    max_box_conf = np.max(scores)

                # Store study data for aggregation
                if study_id not in study_predictions:
                    study_predictions[study_id] = []
                study_predictions[study_id].append((probs, max_box_conf))

                # --- Generate Image Prediction String ---
                # We determine the image-level output immediately, but final filtering
                # based on the study label (Negative vs others) happens conceptually,
                # but practically we output boxes if they exist.
                # However, the prompt says: "If the study-level prediction is 'Negative', all boxes will be suppressed."
                # Since study prediction is aggregated, we might need a 2-pass approach or
                # just generate the string now and fix it if we want perfect consistency.
                # Given the constraints, we'll generate the opacity string based on boxes.
                # If the final study label is negative, we will overwrite this in the final dataframe construction.

                valid_indices = scores > Config.DETECTION_THRESHOLD
                valid_boxes = boxes[valid_indices]
                valid_scores = scores[valid_indices]

                if len(valid_boxes) == 0:
                    pred_str = "none 1 0 0 1 1"
                else:
                    parts = []
                    for b, s in zip(valid_boxes, valid_scores):
                        # Format: opacity conf xmin ymin xmax ymax
                        parts.append(
                            f"opacity {s:.6f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                        )
                    pred_str = " ".join(parts)

                image_predictions.append(
                    {
                        "id": f"{image_id}_image",
                        "PredictionString": pred_str,
                        "study_id": study_id,  # Keep track to check consistency later
                    }
                )

    # 5. Aggregate and Format Study Predictions
    final_results = []
    study_decisions = {}  # Map study_id -> is_negative (bool)

    for study_id, preds in study_predictions.items():
        # preds is list of (probs, max_box_conf)
        # Average probabilities across images in the study
        avg_probs = np.mean([p[0] for p in preds], axis=0)
        # Take the maximum box confidence observed in the study
        study_max_box_conf = np.max([p[1] for p in preds])

        # --- Ensemble Logic ---
        # If the Global Head predicts 'negative' (index 0) but we have a high confidence box,
        # we shift probability to the most likely positive class.
        # This prevents "Negative" diagnosis when a clear opacity is detected.

        # Heuristic: If max_box_conf > 0.5, dampen negative probability
        if study_max_box_conf > 0.5:
            avg_probs[0] *= 0.5
            # Renormalize
            avg_probs /= avg_probs.sum()

        # Select best class
        best_idx = np.argmax(avg_probs)
        best_conf = avg_probs[best_idx]
        best_label = class_names[best_idx]

        # Record decision for image-level consistency
        is_negative = best_idx == 0
        study_decisions[study_id] = is_negative

        # Format: "label confidence 0 0 1 1"
        study_str = f"{best_label} {best_conf:.6f} 0 0 1 1"

        final_results.append({"id": f"{study_id}_study", "PredictionString": study_str})

    # 6. Finalize Image Predictions (Consistency Check)
    for img_pred in image_predictions:
        s_id = img_pred["study_id"]
        # If the study was decided to be Negative, suppress all boxes
        if study_decisions.get(s_id, False):
            final_results.append(
                {"id": img_pred["id"], "PredictionString": "none 1 0 0 1 1"}
            )
        else:
            final_results.append(
                {"id": img_pred["id"], "PredictionString": img_pred["PredictionString"]}
            )

    # 7. Save Submission
    df = pd.DataFrame(final_results)
    # Sort by ID for cleanliness
    df = df.sort_values("id")
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df.head())
