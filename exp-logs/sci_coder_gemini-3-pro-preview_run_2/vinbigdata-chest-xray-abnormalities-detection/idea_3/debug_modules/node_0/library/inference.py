import os
import torch
import pandas as pd
import numpy as np
import torchvision
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, set_seed, format_prediction_string
from library.dataset import get_dataloaders
from library.model import get_model
from library.preprocess import DicomPreprocessor


def predict(
    model, test_loader, device, conf_threshold=0.05, iou_threshold=0.5, use_tta=False
):
    """
    Generates predictions for the test set using the provided model.

    Args:
        model (torch.nn.Module): The trained object detection model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.
        conf_threshold (float): Minimum confidence score to keep a detection.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.
        use_tta (bool): If True, applies horizontal flip Test Time Augmentation.

    Returns:
        list: A list of dictionaries containing 'image_id' and 'PredictionString'.
    """
    model.eval()
    results = []

    with torch.no_grad():
        # Iterate over test loader
        # targets in test_loader contain {'image_id': str, 'original_size': (H, W)}
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)

            # 1. Forward Pass (Original)
            outputs = model(images)

            if use_tta:
                # 2. Forward Pass (Flipped)
                # Flip images horizontally (dim 2 for C,H,W tensor)
                images_flipped = [torch.flip(img, [2]) for img in images]
                outputs_flipped = model(images_flipped)

                # Merge predictions
                for i in range(len(outputs)):
                    h, w = images[i].shape[1], images[i].shape[2]

                    # Original predictions
                    boxes = outputs[i]["boxes"]
                    scores = outputs[i]["scores"]
                    labels = outputs[i]["labels"]

                    # Flipped predictions
                    boxes_f = outputs_flipped[i]["boxes"]
                    scores_f = outputs_flipped[i]["scores"]
                    labels_f = outputs_flipped[i]["labels"]

                    if len(boxes_f) > 0:
                        # Invert flip for boxes: [xmin, ymin, xmax, ymax]
                        # New xmin = width - old_xmax
                        # New xmax = width - old_xmin
                        boxes_f_restored = boxes_f.clone()
                        boxes_f_restored[:, 0] = w - boxes_f[:, 2]
                        boxes_f_restored[:, 2] = w - boxes_f[:, 0]

                        # Concatenate
                        boxes = torch.cat([boxes, boxes_f_restored], dim=0)
                        scores = torch.cat([scores, scores_f], dim=0)
                        labels = torch.cat([labels, labels_f], dim=0)

                    # Update outputs[i] to point to combined tensors for NMS step
                    outputs[i]["boxes"] = boxes
                    outputs[i]["scores"] = scores
                    outputs[i]["labels"] = labels

            # 3. Post-Processing
            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"]

                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                # Filter by Confidence Threshold
                mask = scores >= conf_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                # Apply Non-Maximum Suppression (NMS)
                if len(boxes) > 0:
                    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                # Move to CPU/Numpy for formatting
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()

                # Map Model Labels back to Dataset Class IDs
                # Dataset: 0-13 (Findings), 14 (No Finding)
                # Model (Torchvision): 0 (Background), 1 (Finding 0), ..., 14 (Finding 13)
                # We subtract 1 to get the original class ID.
                final_labels = [int(l) - 1 for l in labels]

                # Format Prediction String
                # Note: format_prediction_string handles empty boxes by returning "14 1 0 0 1 1"
                pred_str = format_prediction_string(boxes, scores, final_labels)
                results.append({"image_id": image_id, "PredictionString": pred_str})

    return results


def run_inference(checkpoint_path=None, use_tta=False):
    """
    Main execution function for inference.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to Config.MODEL_SAVE_PATH.
        use_tta (bool): Whether to use Test Time Augmentation.
    """
    # 1. Setup Environment
    logger = get_logger(os.path.join(Config.LOG_DIR, "inference.log"))
    set_seed(Config.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Inference started on device: {device}")

    # 2. Data Loading
    logger.info("Initializing DataLoaders...")
    try:
        # We only need the test loader
        _, _, test_loader = get_dataloaders(load_cached_data=True)
    except FileNotFoundError:
        logger.warning("Cached data not found. Running offline preprocessing...")
        preprocessor = DicomPreprocessor()
        preprocessor.run(load_cached_data=False)
        _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = get_model(num_classes=Config.NUM_CLASSES, img_size=Config.IMG_SIZE)
    model.to(device)

    # 4. Load Weights
    if checkpoint_path is None:
        checkpoint_path = Config.MODEL_SAVE_PATH

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"No checkpoint found at {checkpoint_path}. Using initialized weights (Caution: Results will be random)."
        )

    # 5. Run Prediction
    logger.info(
        f"Generating predictions (TTA={'Enabled' if use_tta else 'Disabled'})..."
    )
    results = predict(
        model,
        test_loader,
        device,
        conf_threshold=Config.CONFIDENCE_THRESHOLD,
        iou_threshold=0.5,  # Standard NMS threshold
        use_tta=use_tta,
    )

    # 6. Save Submission
    df_sub = pd.DataFrame(results)

    # Save to working directory (standard workflow)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Save to ./submission/submission.csv (specific task requirement)
    req_sub_dir = "./submission"
    os.makedirs(req_sub_dir, exist_ok=True)
    req_sub_path = os.path.join(req_sub_dir, "submission.csv")
    df_sub.to_csv(req_sub_path, index=False)
    logger.info(f"Submission copy saved to {req_sub_path}")
