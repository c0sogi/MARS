import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import seed_everything, get_label_encoder
from library.dataset import HotelDataset, get_transforms
from library.model import HotelResNet


def generate_submission(
    checkpoint_path=Config.MODEL_CHECKPOINT,
    output_file=Config.SUBMISSION_FILE,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Generates predictions for the test set and saves them to a submission CSV.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_file (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a small subset of the test data.
        load_cached_data (bool): Whether to load processed dataset from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 2. Load Label Encoder
    # We need the encoder to know num_classes and to map predictions back to IDs
    # This relies on the training metadata to establish the class mapping
    encoder = get_label_encoder(Config.TRAIN_CSV, load_cached_data=load_cached_data)
    num_classes = len(encoder.id_to_class)
    print(f"Label Encoder loaded. Number of classes: {num_classes}")

    # 3. Prepare Test Data
    # Using 'val' transforms for test (deterministic resize/crop, no augmentation)
    test_dataset = HotelDataset(
        csv_path=Config.TEST_CSV,
        root_dir=Config.INPUT_DIR,
        label_encoder=None,  # Not needed for test set as we don't have targets
        transform=get_transforms("val"),
        is_test=True,
        load_cached_data=load_cached_data,
    )

    # Handle Debug Mode
    if debug:
        print(
            f"Debug mode enabled. Subsetting test set to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        indices = list(range(min(len(test_dataset), Config.DEBUG_SAMPLE_SIZE)))
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to maintain order, though we map by ID anyway
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # 4. Load Model
    # Initialize with pretrained=False to avoid downloading ImageNet weights
    # since we are immediately loading our own trained weights.
    model = HotelResNet(num_classes=num_classes, pretrained=False)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please ensure the model has been trained."
        )

    print(f"Loading model weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Inference Loop
    all_image_ids = []
    all_top5_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get top 5 indices
            # outputs shape: (Batch, NumClasses)
            _, top5_indices = torch.topk(outputs, k=5, dim=1)

            # Move to CPU and numpy
            top5_indices = top5_indices.cpu().numpy()

            all_image_ids.extend(image_ids)
            all_top5_preds.append(top5_indices)

    # Concatenate all predictions
    if len(all_top5_preds) > 0:
        all_top5_preds = np.vstack(all_top5_preds)
    else:
        all_top5_preds = np.array([])

    # 6. Post-processing and Formatting
    print("Processing predictions...")
    submission_rows = []

    for i, img_id in enumerate(all_image_ids):
        # Get the 5 class indices for this image
        indices = all_top5_preds[i]

        # Map indices back to hotel_ids using the encoder
        hotel_ids = encoder.inverse_transform(indices)

        # Format as space-delimited string
        hotel_ids_str = " ".join(map(str, hotel_ids))

        submission_rows.append({"image": img_id, "hotel_id": hotel_ids_str})

    # 7. Save Submission
    submission_df = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print("First 5 rows of submission:")
    print(submission_df.head())
