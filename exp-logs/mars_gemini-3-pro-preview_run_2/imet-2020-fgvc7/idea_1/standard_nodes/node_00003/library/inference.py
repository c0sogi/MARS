import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import set_seed
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkClassifier


def predict_and_submit(
    model_path=Config.MODEL_PATH,
    metadata_path=Config.TEST_METADATA,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    device=Config.device,
    threshold=0.5,
):
    """
    Generates predictions for the test set and creates a submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        metadata_path (str): Path to the test metadata CSV.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run inference on ('cuda' or 'cpu').
        threshold (float): Probability threshold for multi-label classification.
    """
    # 1. Setup
    set_seed(Config.seed)
    print(f"Inference device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # 2. Data Loading
    print(f"Loading test metadata from {metadata_path}...")
    test_dataset = ArtworkDataset(
        metadata_path=metadata_path,
        input_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="test", image_size=Config.image_size),
        mode="test",
        num_classes=Config.num_classes,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model structure: {Config.model_name}")
    # We use pretrained=False because we are loading our own weights
    model = ArtworkClassifier(
        model_name=Config.model_name, num_classes=Config.num_classes, pretrained=False
    )

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []
    print("Starting inference...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            # Mixed precision inference
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            # Move to CPU for processing
            probs = probs.cpu()

            # Apply threshold to get binary predictions
            binary_preds = probs > threshold

            # Process batch
            for i in range(len(image_ids)):
                img_id = image_ids[i]

                # Get indices of predicted classes
                # torch.nonzero returns indices where condition is True
                pred_indices = torch.nonzero(binary_preds[i]).flatten().numpy()

                # Convert to space-separated string
                if len(pred_indices) > 0:
                    pred_str = " ".join(map(str, pred_indices))
                else:
                    pred_str = ""  # No attributes predicted

                results.append({"id": img_id, "attribute_ids": pred_str})

    # 5. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions generated: {len(submission_df)}")
