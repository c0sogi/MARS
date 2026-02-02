import os
import torch
import pandas as pd
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything, get_logger, load_hierarchy_mappings
from library.data import get_dataloaders
from library.model import HierarchicalEfficientNet

# Initialize logger
logger = get_logger("inference")


def load_model(checkpoint_path, device):
    """
    Initializes the model architecture and loads the trained weights.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (str): Device to load the model onto.

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    # Initialize model architecture
    # We use pretrained=False because we are about to load our own weights
    model = HierarchicalEfficientNet(pretrained=False)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def predict_tta(model, loader, device):
    """
    Performs inference on the test set using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (str): Computation device.

    Returns:
        tuple: (list of image_ids, list of predicted_labels)
    """
    model.eval()
    all_ids = []
    all_preds = []

    logger.info("Starting inference with TTA (Horizontal Flip)...")

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # Forward pass 1: Original images
            outputs = model(images)
            logits = outputs["species"]
            probs = torch.softmax(logits, dim=1)

            # Forward pass 2: Horizontally flipped images
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            logits_flipped = outputs_flipped["species"]
            probs_flipped = torch.softmax(logits_flipped, dim=1)

            # Average probabilities
            avg_probs = (probs + probs_flipped) / 2.0

            # Get predictions
            preds = torch.argmax(avg_probs, dim=1)

            all_ids.extend(image_ids)
            all_preds.extend(preds.cpu().numpy())

    return all_ids, all_preds


def generate_submission(image_ids, predictions, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        image_ids (list): List of image identifiers.
        predictions (list): List of predicted category IDs.
        output_path (str): Path to save the submission CSV.
    """
    df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")


def run_inference(
    checkpoint_path=None,
    batch_size=Config.STAGE2_BATCH_SIZE,
    img_size=Config.STAGE2_IMG_SIZE,
    device=Config.DEVICE,
    debug=Config.DEBUG,
):
    """
    Main orchestration function for inference.

    Args:
        checkpoint_path (str, optional): Path to model checkpoint. Defaults to Stage 2 best model.
        batch_size (int): Batch size for inference.
        img_size (int): Image resolution (should match Stage 2 training).
        device (str): Device to run inference on.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    seed_everything(Config.SEED)

    # Default to the best model from Stage 2 if not specified
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR_STAGE2, "best_model.pth")

    # 1. Prepare Data
    # We only need the test loader. get_dataloaders returns (train, val, test)
    logger.info(f"Initializing dataloaders with image size {img_size}...")
    _, _, test_loader = get_dataloaders(
        img_size=img_size, batch_size=batch_size, debug=debug
    )

    # 2. Load Model
    model = load_model(checkpoint_path, device)

    # 3. Predict
    ids, preds = predict_tta(model, test_loader, device)

    # Decode predictions (contiguous index -> original category_id)
    # Cite debug_lesson_1: Map contiguous indices back to original sparse IDs
    hierarchy_df = load_hierarchy_mappings(
        Config.TRAIN_METADATA_JSON, Config.HIERARCHY_CACHE_PATH, load_cached_data=True
    )
    # Create mapping: species_label -> category_id
    label_map = dict(zip(hierarchy_df["species_label"], hierarchy_df["category_id"]))
    decoded_preds = [label_map.get(p, p) for p in preds]

    # 4. Save Submission
    generate_submission(ids, decoded_preds, Config.SUBMISSION_PATH)
