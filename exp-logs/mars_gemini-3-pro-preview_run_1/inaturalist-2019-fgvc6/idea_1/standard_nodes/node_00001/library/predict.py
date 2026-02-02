import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.dataset import INatDataset, get_transforms
from library.model import get_mobilenet_model


def generate_submission(
    checkpoint_path=None, device=None, batch_size=None, num_workers=None
):
    """
    Generates predictions for the test set using a trained model and saves them to a CSV file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint. Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        device (str, optional): Device to run inference on ('cuda' or 'cpu'). Defaults to Config.DEVICE.
        batch_size (int, optional): Batch size for the data loader. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to Config.NUM_WORKERS.
    """
    # Set defaults from Config if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if device is None:
        device = Config.DEVICE
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Initializing model on {device}...")
    model = get_mobilenet_model(
        pretrained=False, num_classes=Config.NUM_CLASSES, device=device
    )

    # Load model weights
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Predictions will be based on random/initialized weights."
        )

    model.eval()

    # Prepare Test Data
    print("Setting up test dataset and loader...")
    test_dataset = INatDataset(
        csv_path=Config.TEST_CSV, mode="test", transform=get_transforms(stage="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    predictions = []
    print("Starting inference on test set...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get top 5 predictions (indices of the classes)
            # outputs shape: [batch_size, num_classes]
            _, top5_indices = torch.topk(outputs, k=5, dim=1)

            # Move to CPU and convert to numpy for processing
            top5_indices = top5_indices.cpu().numpy()

            # Handle image_ids (convert Tensor to numpy if necessary)
            if isinstance(image_ids, torch.Tensor):
                image_ids = image_ids.numpy()

            # Format predictions
            for img_id, preds in zip(image_ids, top5_indices):
                # Join class indices with spaces
                pred_str = " ".join(map(str, preds))
                predictions.append({"id": img_id, "predicted": pred_str})

    # Create DataFrame and save to CSV
    df = pd.DataFrame(predictions)

    # Ensure columns are in the correct order as per sample submission (though usually id, predicted is standard)
    df = df[["id", "predicted"]]

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission generated successfully with {len(df)} rows.")
    print(f"Saved to: {Config.SUBMISSION_PATH}")
