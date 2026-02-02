import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from scipy.ndimage import label
from collections import defaultdict

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import prepare_data, UWMadisonDataset, get_transforms
from library.model import SegFormer


def predict_volume(
    model_path=Config.MODEL_SAVE_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Loads the test dataset and model, runs inference, and groups predictions by case/day.

    Args:
        model_path (str): Path to the saved model weights.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
        num_workers (int): Number of workers for data loading.

    Returns:
        dict: A dictionary where keys are 'case_day' strings and values are lists of
              dictionaries containing slice prediction data.
    """
    set_seed(Config.SEED)

    print(f"Loading test metadata from {Config.TEST_METADATA_PATH}...")
    # Load test data
    test_df = prepare_data(Config.TEST_METADATA_PATH, mode="test")
    test_dataset = UWMadisonDataset(test_df, get_transforms("test"), mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize Model
    print("Initializing model...")
    model = SegFormer(
        backbone_name=Config.BACKBONE,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # No need to download pretrained weights for inference
    ).to(device)

    # Load Weights
    if not os.path.exists(model_path):
        print(
            f"Warning: Model file not found at {model_path}. Returning empty predictions."
        )
        return {}

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Storage for 3D processing
    # Structure: case_data['caseXXX_dayYY'] = list of slice info
    case_data = defaultdict(list)

    print("Running inference loop...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            ids = batch["id"]
            # Retrieve original dimensions for resizing later
            orig_h = batch["orig_h"].numpy()
            orig_w = batch["orig_w"].numpy()

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Binarize with threshold 0.5
            preds = (probs > 0.5).float().cpu().numpy()

            # Group by case_day
            for i, img_id in enumerate(ids):
                # ID format: caseXXX_dayYY_slice_ZZZZ
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_idx = int(parts[3])

                case_data[case_day].append(
                    {
                        "slice_idx": slice_idx,
                        "pred": preds[i],  # Shape: (C, 256, 256)
                        "id": img_id,
                        "h": orig_h[i],
                        "w": orig_w[i],
                    }
                )

    return case_data


def post_process_3d(case_data):
    """
    Applies 3D Connected Component Analysis to keep only the largest object per class.

    Args:
        case_data (dict): Dictionary of raw predictions grouped by case_day.

    Returns:
        dict: The updated case_data with post-processed prediction masks.
    """
    print("Applying 3D post-processing...")

    num_classes = Config.NUM_CLASSES

    for case_day, slices in case_data.items():
        # Sort slices by index to ensure correct Z-ordering
        slices.sort(key=lambda x: x["slice_idx"])

        # Stack predictions to form a 3D volume: (C, D, H, W)
        # slices[i]['pred'] is (C, H, W)
        vol_stack = np.stack([s["pred"] for s in slices], axis=1)

        # Process each class channel independently
        for c in range(num_classes):
            class_vol = vol_stack[c]  # (D, H, W)

            # Label connected components
            labeled_vol, num_features = label(class_vol)

            if num_features > 1:
                # Calculate size of each component (background is 0)
                sizes = [np.sum(labeled_vol == k) for k in range(1, num_features + 1)]

                # Identify the label of the largest component
                largest_k = np.argmax(sizes) + 1

                # Create mask keeping only the largest component
                class_vol = (labeled_vol == largest_k).astype(float)

                # Update the volume stack
                vol_stack[c] = class_vol

        # Distribute the processed volume back to the slice dictionaries
        for i, s_info in enumerate(slices):
            # Extract the C channels for this slice index
            s_info["pred"] = vol_stack[:, i, :, :]

    return case_data


def create_submission(case_data, submission_path=Config.SUBMISSION_PATH):
    """
    Resizes masks to original dimensions, RLE encodes them, and saves the submission file.

    Args:
        case_data (dict): Dictionary of processed predictions.
        submission_path (str): Path to save the CSV.
    """
    print("Generating submission file...")

    results = []
    class_names = ["large_bowel", "small_bowel", "stomach"]

    # Iterate over all cases
    for case_day, slices in case_data.items():
        # Ensure sorted order (though post_process_3d sorts them, good to be safe)
        slices.sort(key=lambda x: x["slice_idx"])

        for s_info in slices:
            pred_mask = s_info["pred"]  # (C, 256, 256)
            orig_h = s_info["h"]
            orig_w = s_info["w"]
            img_id = s_info["id"]

            rles = []
            for c in range(Config.NUM_CLASSES):
                mask_c = pred_mask[c]

                # Resize to original image dimensions
                # Use Nearest Neighbor to maintain binary nature
                mask_orig = cv2.resize(
                    mask_c, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                # Encode
                rle = rle_encode(mask_orig)
                rles.append(rle)

            # Add rows for each class
            for c_idx, c_name in enumerate(class_names):
                results.append(
                    {"id": img_id, "class": c_name, "predicted": rles[c_idx]}
                )

    # Create DataFrame and Save
    if results:
        sub_df = pd.DataFrame(results)
        # Ensure correct column order
        sub_df = sub_df[["id", "class", "predicted"]]
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print("No results to save. Creating empty submission file.")
        # Create empty file with headers just in case
        pd.DataFrame(columns=["id", "class", "predicted"]).to_csv(
            submission_path, index=False
        )


def run_inference(model_path=Config.MODEL_SAVE_PATH):
    """
    Orchestrates the full inference pipeline.
    """
    # 1. Predict
    raw_predictions = predict_volume(model_path=model_path)

    if not raw_predictions:
        print("Inference failed or no data found.")
        return

    # 2. Post-process (3D CCA)
    processed_predictions = post_process_3d(raw_predictions)

    # 3. Create Submission
    create_submission(processed_predictions)
