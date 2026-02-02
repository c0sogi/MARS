import os
import torch
import pandas as pd
import numpy as np
from library.utils import Config, get_device, set_seed
from library.model import LightweightMetricModel
from library.dataset import get_dataloaders


def generate_submission(
    checkpoint_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): Whether to load processed metadata from cache.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Inference using device: {device}")

    # 2. Data Loading
    # We need the test_loader and the encoder_classes to decode predictions
    print("Loading test data...")
    _, _, test_loader, encoder_classes = get_dataloaders(
        load_cached_data=load_cached_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    num_classes = len(encoder_classes)
    print(f"Number of classes: {num_classes}")

    # 3. Model Initialization
    print("Initializing model...")
    model = LightweightMetricModel(
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        backbone_name=Config.BACKBONE,
        pretrained=False,  # No need to download weights, we are loading a checkpoint
    )
    model = model.to(device)

    # 4. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint saving formats (e.g., if wrapped in 'state_dict')
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    print("Generating predictions...")
    image_ids = []
    predictions = []

    with torch.no_grad():
        for i, (images, filenames) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass
            # The model's forward method with label=None returns the scaled cosine similarities
            # between the input embeddings and the ArcFace class centers.
            # Shape: (Batch_Size, Num_Classes)
            outputs = model(images)

            # Get top 5 indices
            # We want the classes with the highest cosine similarity
            _, top_indices = outputs.topk(5, dim=1, largest=True, sorted=True)

            # Move to CPU for processing
            top_indices = top_indices.cpu().numpy()

            # Map indices to original hotel IDs
            # encoder_classes is a numpy array where index corresponds to the label_idx
            batch_preds = encoder_classes[top_indices]

            # Format predictions
            for pred_row in batch_preds:
                # pred_row is an array of 5 hotel IDs (integers or strings)
                # Join them with spaces
                pred_str = " ".join(map(str, pred_row))
                predictions.append(pred_str)

            image_ids.extend(filenames)

            if (i + 1) % 10 == 0:
                print(f"Processed batch {i + 1}/{len(test_loader)}")

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({"image": image_ids, "hotel_id": predictions})

    # 7. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions generated: {len(submission_df)}")
