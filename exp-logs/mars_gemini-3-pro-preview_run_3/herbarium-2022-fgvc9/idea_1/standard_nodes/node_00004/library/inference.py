import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import PlantClassifier
from library.dataset import get_dataloaders
from library.utils import get_label_mappings


def generate_submission(
    checkpoint_path=None,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    debug=Config.DEBUG,
):
    """
    Loads the best trained model, generates predictions for the test set,
    and saves the submission file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CACHE_DIR/best_model.pth.
        batch_size (int): Batch size for the test loader.
        device (str): Device to run inference on.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    # 1. Setup Paths and Device
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Inference device: {device}")

    # 2. Initialize Model
    print("Initializing model...")
    model = PlantClassifier(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 3. Load Checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # 4. Get Test DataLoader
    # We only need the test loader here
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(val_batch_size=batch_size, debug=debug)

    # 5. Generate Predictions
    predictions = []
    image_ids = []

    print(f"Generating predictions for {len(test_loader.dataset)} images...")

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Use mixed precision for inference efficiency
            with torch.cuda.amp.autocast():
                outputs = model(images)

            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    # 6. Create Submission DataFrame
    # Ensure Id is treated as integer to match sample submission format
    # The dataset returns ids as strings, so we convert them.
    try:
        image_ids_int = [int(x) for x in image_ids]
    except ValueError:
        # Fallback if IDs are not integers
        image_ids_int = image_ids

    # Map predictions back to original category_ids
    _, label2id = get_label_mappings(Config.TRAIN_METADATA_PATH)
    final_predictions = [label2id[p] for p in predictions]

    df_submission = pd.DataFrame({"Id": image_ids_int, "Predicted": final_predictions})

    # 7. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(df_submission)}")
