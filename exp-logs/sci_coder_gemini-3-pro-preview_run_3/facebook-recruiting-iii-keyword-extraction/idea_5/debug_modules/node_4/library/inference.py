import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, set_seed, load_checkpoint
from library.model import DualStreamTextCNN
from library.data import get_dataloaders

# Initialize Logger
logger = get_logger("inference_module")


def predict_probs(model, loader, device):
    """
    Runs inference on a dataloader and returns predicted probabilities.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The data loader.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: Predicted probabilities in float16 (Num_Samples, Num_Classes).
    """
    model.eval()
    probs_list = []

    logger.info(f"Starting inference on {len(loader)} batches...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            title = batch["title"].to(device, non_blocking=True)
            body = batch["body"].to(device, non_blocking=True)

            # Use autocast for mixed precision inference (speed + memory)
            with autocast():
                logits = model(title, body)
                probs = torch.sigmoid(logits)

            # Move to CPU and convert to float16 to save memory
            probs_list.append(probs.cpu().numpy().astype(np.float16))

    # Concatenate all batches
    all_probs = np.vstack(probs_list)
    return all_probs


def optimize_threshold(val_probs, val_targets_sparse):
    """
    Finds the optimal probability threshold that maximizes the F1-score (samples).

    Args:
        val_probs (np.ndarray): Predicted probabilities (float16).
        val_targets_sparse (scipy.sparse.csr_matrix): Ground truth labels.

    Returns:
        float: The optimal threshold.
    """
    logger.info("Optimizing probability threshold...")

    thresholds = np.arange(0.1, 0.55, 0.05)
    best_threshold = 0.5
    best_score = 0.0

    # Ensure targets are in a compatible format for fast scoring if possible
    # scikit-learn f1_score works with sparse matrices for targets

    for thresh in thresholds:
        # Binarize predictions
        # Note: This creates a boolean matrix which is relatively memory efficient
        val_preds = val_probs > thresh

        # Calculate F1 Score (Samples Average)
        # We use the sparse targets directly
        score = f1_score(
            val_targets_sparse, val_preds, average="samples", zero_division=0
        )

        logger.info(f"Threshold: {thresh:.2f} | F1-Score: {score}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    logger.info(f"Best Threshold: {best_threshold:.2f} with F1-Score: {best_score}")
    return best_threshold


def generate_submission(test_probs, test_ids, threshold, mlb, output_path):
    """
    Generates the submission CSV file.

    Args:
        test_probs (np.ndarray): Predicted probabilities for test set.
        test_ids (np.ndarray): IDs for the test set.
        threshold (float): Optimal threshold for binarization.
        mlb (MultiLabelBinarizer): Fitted binarizer to convert indices to tags.
        output_path (str): Path to save the CSV.
    """
    logger.info(f"Generating submission file with threshold {threshold:.2f}...")

    predicted_tags = []
    num_samples = len(test_probs)
    chunk_size = 10000  # Process in chunks to manage memory during inverse_transform

    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        chunk_probs = test_probs[i:end]

        # Binarize
        chunk_preds = chunk_probs > threshold

        # Inverse transform to get tags
        # Returns a list of tuples of tags
        chunk_tags_tuples = mlb.inverse_transform(chunk_preds)

        # Join tuples into space-delimited strings
        chunk_tags_strings = [" ".join(tags) for tags in chunk_tags_tuples]
        predicted_tags.extend(chunk_tags_strings)

    # Create DataFrame
    df_submission = pd.DataFrame({"Id": test_ids, "Tags": predicted_tags})

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # quoting=2 is csv.QUOTE_NONNUMERIC (quotes around non-numeric fields)
    # However, pandas usually handles strings correctly.
    # The sample format is: 1,"tag1 tag2"
    df_submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Head of submission:\n{df_submission.head()}")


def run_inference():
    """
    Main function to run the inference pipeline.
    """
    set_seed(Config.SEED)
    device = Config.get_device()
    logger.info(f"Using device: {device}")

    # 1. Load Data Loaders and Artifacts
    # We rely on cached data if available
    _, val_loader, test_loader, vocab, mlb = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    num_classes = len(mlb.classes_)
    logger.info(f"Number of classes: {num_classes}")

    # 2. Load Model
    model = DualStreamTextCNN(num_classes=num_classes)
    model.to(device)

    # Load weights
    try:
        start_epoch, best_metric = load_checkpoint(
            Config.MODEL_SAVE_PATH, model, device=device
        )
        logger.info(
            f"Loaded checkpoint from epoch {start_epoch-1} with metric {best_metric}"
        )
    except FileNotFoundError:
        logger.error(
            f"Checkpoint not found at {Config.MODEL_SAVE_PATH}. Please train the model first."
        )
        return

    # 3. Optimize Threshold using Validation Set
    # We process validation first, then clear memory before processing test
    logger.info("--- Processing Validation Set ---")
    val_probs = predict_probs(model, val_loader, device)

    # Retrieve sparse targets from dataset
    # val_loader.dataset is StackExchangeDataset
    # val_loader.dataset.labels is the sparse matrix (or we can access the file if not stored in dataset object)
    # Based on data.py, self.labels is stored in the dataset instance.
    val_targets = val_loader.dataset.labels

    best_threshold = optimize_threshold(val_probs, val_targets)

    # Clean up validation data to free memory
    del val_probs
    del val_targets
    gc.collect()

    # 4. Predict on Test Set
    logger.info("--- Processing Test Set ---")
    test_probs = predict_probs(model, test_loader, device)

    # Get Test IDs
    test_ids = test_loader.dataset.ids

    # 5. Generate Submission
    generate_submission(
        test_probs, test_ids, best_threshold, mlb, Config.SUBMISSION_FILE
    )

    # Clean up
    del test_probs
    gc.collect()
    logger.info("Inference completed successfully.")


if __name__ == "__main__":
    run_inference()
