import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from library import config, utils, model, dataset, bbox_handler


def get_idx_to_class_map():
    """
    Reconstructs the index-to-category mapping used during training.
    The dataset maps sorted unique category_ids to indices 0..N-1.
    We need the inverse to map model predictions back to category_ids.
    """
    if os.path.exists(config.TRAIN_METADATA_PATH):
        train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
        # Sort unique category IDs to ensure deterministic mapping matching the Dataset class
        unique_classes = sorted(train_df["category_id"].unique())
        # Create map: model_index -> category_id
        return {idx: cls_id for idx, cls_id in enumerate(unique_classes)}
    else:
        raise FileNotFoundError(
            f"Training metadata not found at {config.TRAIN_METADATA_PATH}. "
            "Cannot reconstruct class mapping."
        )


def run_inference(
    checkpoint_path=config.BEST_MODEL_PATH,
    output_path=config.SUBMISSION_FILE_PATH,
    batch_size=config.BATCH_SIZE,
    device=None,
):
    """
    Runs inference on the test set using the trained model and generates a submission file.
    Implements Test Time Augmentation (TTA) by averaging predictions of original and flipped images.

    Args:
        checkpoint_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device to run inference on.
    """
    if device is None:
        device = utils.get_device()

    print(f"Inference device: {device}")

    # 1. Reconstruct Class Mapping
    print("Reconstructing class mapping...")
    idx_to_class = get_idx_to_class_map()

    # 2. Load Model
    print(f"Loading model architecture: {config.MODEL_NAME}...")
    net = model.get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # We are loading custom weights, not ImageNet
        device=device,
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # 3. Prepare Data
    print("Preparing test dataset...")
    # Initialize BBoxHandler to ensure consistent cropping logic
    bbox_h = bbox_handler.BBoxHandler(load_cached_data=True)

    test_dataset = dataset.WildCamDataset(
        metadata_path=config.TEST_METADATA_PATH, mode="test", bbox_handler=bbox_h
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Prediction Loop
    print("Starting inference with TTA (Horizontal Flip)...")
    results = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # --- TTA Step 1: Original Image ---
            outputs_orig = net(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # --- TTA Step 2: Horizontally Flipped Image ---
            # Flip along width dimension (dim 3 for NCHW format)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flip = net(images_flipped)
            probs_flip = F.softmax(outputs_flip, dim=1)

            # --- Average Probabilities ---
            avg_probs = (probs_orig + probs_flip) / 2.0

            # --- Get Predictions ---
            _, preds = torch.max(avg_probs, dim=1)

            # Move to CPU for processing
            preds_np = preds.cpu().numpy()

            # Map indices back to original category IDs
            for img_id, pred_idx in zip(image_ids, preds_np):
                category_id = idx_to_class.get(pred_idx)
                if category_id is None:
                    # Fallback should not happen if mapping is correct
                    print(f"Warning: Predicted index {pred_idx} not found in mapping.")
                    category_id = 0

                results.append({"Id": img_id, "Predicted": category_id})

    # 5. Save Submission
    print(f"Generating submission file with {len(results)} predictions...")
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    utils.ensure_directory(os.path.dirname(output_path))

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
