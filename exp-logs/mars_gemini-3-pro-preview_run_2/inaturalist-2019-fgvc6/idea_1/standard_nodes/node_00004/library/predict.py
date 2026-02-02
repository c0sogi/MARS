import os
import torch
import pandas as pd
from library.config import Config
from library.model import SpeciesModel
from library.dataset import get_loaders
from library.utils import set_seed


def generate_predictions(
    model_path=None, output_path=None, device=None, batch_size=None
):
    """
    Generates predictions for the test set using a trained model checkpoint.

    Args:
        model_path (str, optional): Path to the model checkpoint.
                                    Defaults to Config.WORKING_DIR/model_best.pth.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to Config.SUBMISSION_FILE_PATH.
        device (str, optional): Device to run inference on ('cpu' or 'cuda').
                                Defaults to Config.DEVICE.
        batch_size (int, optional): Batch size for inference.
                                    Defaults to Config.BATCH_SIZE.
    """
    # Set defaults if not provided
    if model_path is None:
        model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    if output_path is None:
        output_path = Config.SUBMISSION_FILE_PATH

    if device is None:
        device = Config.DEVICE

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Loading model from {model_path}...")
    print(f"Inference device: {device}")

    # Initialize Model
    model = SpeciesModel()

    # Load Checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Handle case where checkpoint saves 'state_dict' vs direct model
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load weights into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get DataLoaders
    # We only need the test_loader.
    # get_loaders uses Config.BATCH_SIZE, so we update it temporarily if a custom batch_size is passed.
    print("Preparing data loaders...")
    original_batch_size = Config.BATCH_SIZE
    Config.BATCH_SIZE = batch_size

    try:
        _, _, test_loader = get_loaders()
    finally:
        # Restore original config to avoid side effects
        Config.BATCH_SIZE = original_batch_size

    # Ensure mapping exists
    if not hasattr(test_loader.dataset, "idx_to_class"):
        raise ValueError("Test dataset is missing 'idx_to_class' attribute.")

    idx_to_class = test_loader.dataset.idx_to_class

    predictions = []
    print(f"Starting inference on {len(test_loader.dataset)} test images...")

    with torch.no_grad():
        for i, (images, image_ids) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Get top 5 predictions
            # outputs: (Batch_Size, Num_Classes)
            _, topk_indices = torch.topk(outputs, k=5, dim=1)

            topk_indices = topk_indices.cpu().numpy()
            image_ids = image_ids.numpy()

            for img_id, indices in zip(image_ids, topk_indices):
                # Map model indices (0..N-1) to original category IDs
                cat_ids = [str(idx_to_class[idx]) for idx in indices]

                # Format: "cat_id1 cat_id2 cat_id3 cat_id4 cat_id5"
                pred_str = " ".join(cat_ids)

                predictions.append({"id": img_id, "predicted": pred_str})

    # Create DataFrame
    df = pd.DataFrame(predictions)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
    print(f"Total predictions generated: {len(df)}")
