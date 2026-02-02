import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library import config, dataset, model, utils


def generate_submission(load_cached_data=True):
    """
    Loads the trained model, runs inference on the test set, and generates
    the submission.csv file in the required format.

    Args:
        load_cached_data (bool): Whether to use cached image data if available.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    if not os.path.exists(config.TEST_METADATA_PATH):
        print(
            f"Test metadata not found at {config.TEST_METADATA_PATH}. Cannot generate submission."
        )
        return

    # 2. Load Data
    # We use the provided dataset class which handles caching and transforms
    df_test = pd.read_csv(config.TEST_METADATA_PATH)
    test_dataset = dataset.SIIMDataset(
        df_test, split="test", load_cached_data=load_cached_data
    )

    # Pin memory helps with faster data transfer to GPU
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    # Initialize architecture
    net = model.MultiTaskUNet(
        pretrained=False
    )  # Pretrained weights not needed for inference, we load checkpoint

    # Load checkpoint
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(
            f"Checkpoint not found at {config.CHECKPOINT_PATH}. Cannot generate submission."
        )
        return

    state_dict = torch.load(config.CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # 4. Inference
    study_probs_map = {}  # Map study_id -> list of probability arrays (one per image)
    image_predictions = {}  # Map image_id -> prediction string

    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]
            image_ids = batch["image_id"]

            # Forward pass
            seg_logits, class_logits = net(images)

            # Convert logits to probabilities
            # Segmentation: (B, 1, H, W) -> (B, H, W)
            seg_probs = torch.sigmoid(seg_logits).squeeze(1).cpu().numpy()
            # Classification: (B, NumClasses)
            class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()

            # Process batch
            for i in range(len(images)):
                s_id = study_ids[i]
                i_id = image_ids[i]
                c_prob = class_probs[i]
                mask_prob = seg_probs[i]

                # Collect study-level probabilities for aggregation later
                if s_id not in study_probs_map:
                    study_probs_map[s_id] = []
                study_probs_map[s_id].append(c_prob)

                # Generate Image-level Prediction
                # Strategy:
                # If the specific image is classified as "Negative for Pneumonia" (index 0),
                # we predict "none". Otherwise, we extract boxes from the mask.

                pred_class_idx = np.argmax(c_prob)

                # Index 0 corresponds to "Negative for Pneumonia" in config.STUDY_LABELS
                if pred_class_idx == 0:
                    image_predictions[i_id] = "none 1 0 0 1 1"
                else:
                    # Extract boxes from segmentation mask
                    boxes = utils.mask_to_boxes(
                        mask_prob, threshold=config.MASK_THRESHOLD
                    )

                    # Format boxes into string
                    # If no boxes found despite positive class, utils.format_prediction_string returns "none..."
                    # However, usually we want to trust the class head or the mask.
                    # If boxes is empty, format_prediction_string returns "none 1 0 0 1 1".
                    image_predictions[i_id] = utils.format_prediction_string(boxes)

    # 5. Format Submission
    submission_rows = []

    # Process Study Level (Aggregation)
    # We average the probability vectors of all images in a study
    for s_id, probs_list in study_probs_map.items():
        avg_probs = np.mean(probs_list, axis=0)
        best_idx = np.argmax(avg_probs)
        confidence = avg_probs[best_idx]
        label_name = config.STUDY_LABELS[best_idx]

        # Format label: "Negative for Pneumonia" -> "negative", "Typical Appearance" -> "typical", etc.
        # We take the first word and lowercase it.
        short_label = label_name.split(" ")[0].lower()

        # Study prediction format: "class confidence 0 0 1 1"
        pred_string = f"{short_label} {confidence:.6f} 0 0 1 1"

        submission_rows.append({"id": f"{s_id}_study", "PredictionString": pred_string})

    # Process Image Level
    for i_id, pred_str in image_predictions.items():
        submission_rows.append({"id": f"{i_id}_image", "PredictionString": pred_str})

    # 6. Save to CSV
    df_sub = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH} with {len(df_sub)} rows.")
