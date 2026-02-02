import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from library.config import Config
from library.dataset import get_dataloader
from library.model import HierarchicalEfficientNet
from library.utils import set_seed


def inference(checkpoint_path=None, batch_size=None):
    """
    Performs inference on the test dataset and generates a submission file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CHECKPOINT_STAGE_2.
        batch_size (int, optional): Batch size for inference.
                                    Defaults to Config.STAGE_2_BATCH_SIZE.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    if checkpoint_path is None:
        checkpoint_path = Config.CHECKPOINT_STAGE_2

    if batch_size is None:
        batch_size = Config.STAGE_2_BATCH_SIZE

    print(f"Starting inference using checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"TTA Enabled: {Config.TTA_FLIP}")

    # 2. Load Data
    # Ensure test metadata exists
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Use the resolution from Stage 2 for inference
    test_loader = get_dataloader(
        df_test,
        mode="test",
        batch_size=batch_size,
        image_size=Config.STAGE_2_RES,
        shuffle=False,
    )

    # 3. Initialize Model
    model = HierarchicalEfficientNet(pretrained=False)

    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Ensure training is complete."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Forward pass (Original)
            outputs = model(images)
            # We only care about the species head for the final submission
            logits = outputs["species"]
            probs = F.softmax(logits, dim=1)

            # Test Time Augmentation (Horizontal Flip)
            if Config.TTA_FLIP:
                # Flip images horizontally (dim 3 is width in NCHW)
                images_flipped = torch.flip(images, dims=[3])

                outputs_flip = model(images_flipped)
                logits_flip = outputs_flip["species"]
                probs_flip = F.softmax(logits_flip, dim=1)

                # Average probabilities
                probs = (probs + probs_flip) / 2.0

            # Get predictions
            preds = torch.argmax(probs, dim=1)

            # Store results
            all_ids.extend(
                image_ids.tolist()
                if isinstance(image_ids, torch.Tensor)
                else list(image_ids)
            )
            all_preds.extend(preds.cpu().numpy().tolist())

    # 5. Generate Submission
    submission_df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(submission_df)}")
