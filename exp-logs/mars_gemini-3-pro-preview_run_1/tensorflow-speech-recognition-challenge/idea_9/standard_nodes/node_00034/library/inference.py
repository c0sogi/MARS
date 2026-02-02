import os
import torch
import pandas as pd
from library.config import Config, set_seed
from library.utils import get_logger, map_prediction_to_label
from library.dataset import get_dataloaders
from library.model import DilatedEfficientNet


def generate_submission(
    load_cached_data: bool = True, batch_size: int = Config.BATCH_SIZE
):
    """
    Generates predictions for the test set using the best saved model.
    Saves the result to ./submission/submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        batch_size (int): Batch size for inference.
    """
    logger = get_logger("inference")
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Load Data
    # We use get_dataloaders to ensure consistent preprocessing (Dual-Channel Spectrograms).
    # We only need the test_loader for inference.
    logger.info("Initializing data loaders...")
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
        debug_subset_size=None,  # Always predict on the full test set
    )

    # 2. Load Model
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        logger.error(
            f"Model file not found at {model_path}. Cannot generate submission."
        )
        return

    logger.info(f"Loading model from {model_path}...")
    # Initialize model structure (pretrained=False because we load our own weights)
    model = DilatedEfficientNet(num_classes=Config.NUM_CLASSES, pretrained=False)

    # Load trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load state dict: {e}")
        return

    model.to(device)
    model.eval()

    # 3. Inference Loop
    predictions = []
    fnames = []

    # Access the underlying dataframe to retrieve filenames
    # The test loader is not shuffled, so order is preserved.
    test_df = test_loader.dataset.df
    total_files = len(test_df)

    logger.info(f"Starting inference on {total_files} test files...")

    with torch.no_grad():
        batch_start_idx = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)  # (B, NumClasses)

            # Get predicted class indices (Fine-grained IDs)
            batch_preds_indices = torch.argmax(outputs, dim=1).cpu().numpy()

            # Map indices to submission labels and match with filenames
            current_batch_size = inputs.size(0)

            for i in range(current_batch_size):
                global_idx = batch_start_idx + i

                # Safety check to avoid index errors if loader/df mismatch (should not happen)
                if global_idx >= total_files:
                    break

                # Get filename from dataframe
                # filepath in metadata is like "test/audio/clip_xxxx.wav"
                # Submission format requires just filename "clip_xxxx.wav"
                full_path = test_df.iloc[global_idx]["filepath"]
                fname = os.path.basename(full_path)

                # Get prediction index
                pred_idx = batch_preds_indices[i]

                # Map to 12-class string (handles 31->12 mapping)
                label_str = map_prediction_to_label(pred_idx)

                fnames.append(fname)
                predictions.append(label_str)

            batch_start_idx += current_batch_size

    # 4. Save Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame({"fname": fnames, "label": predictions})
    submission_df.to_csv(submission_path, index=False)

    logger.info(
        f"Submission saved to {submission_path} with {len(submission_df)} rows."
    )
