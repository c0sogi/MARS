import os
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_processing import prepare_data, get_dataloaders, TagEncoder
from library.model import WideAndDeepModel
from library.trainer import Trainer


def find_optimal_threshold(trainer, val_loader, val_labels_sparse):
    """
    Evaluates the model on the validation set and searches for the best
    probability threshold based on percentiles to maximize the F1 score.

    Args:
        trainer: Instance of library.trainer.Trainer
        val_loader: DataLoader for validation set
        val_labels_sparse: Sparse matrix (CSR) of validation labels

    Returns:
        float: The optimal threshold value.
    """
    print("Predicting on validation set for threshold tuning...")
    val_probs = trainer.predict(val_loader)

    print("Optimizing threshold based on validation probabilities...")

    # Flatten probabilities to find distribution percentiles
    flat_probs = val_probs.flatten()

    # Define search range based on percentiles
    # We focus on the upper tail because tags are sparse (mostly zeros)
    percentiles = np.concatenate(
        [
            np.arange(90, 99, 1),  # 90, 91, ..., 98
            np.arange(99, 99.9, 0.1),  # 99.0, 99.1, ..., 99.8
            [99.9, 99.95, 99.99],  # Extreme tail
        ]
    )

    thresholds = np.percentile(flat_probs, percentiles)
    thresholds = np.unique(thresholds)  # Remove duplicates
    thresholds = thresholds[thresholds > 0.01]  # Sanity check

    best_th = 0.5
    best_f1 = 0.0

    # Convert sparse targets to dense for consistent scoring with sklearn
    # Given 220GB RAM, converting validation targets to dense is safe.
    if hasattr(val_labels_sparse, "toarray"):
        targets = val_labels_sparse.toarray()
    else:
        targets = val_labels_sparse

    for th in thresholds:
        preds = (val_probs > th).astype(int)

        # Calculate F1 score with 'samples' average
        score = f1_score(targets, preds, average="samples", zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_th = th

    print(f"Optimal Threshold: {best_th} with Validation F1: {best_f1}")
    return best_th


def generate_submission(trainer, test_loader, threshold):
    """
    Runs inference on the test set using the optimized threshold and formats the output CSV.

    Args:
        trainer: Instance of library.trainer.Trainer
        test_loader: DataLoader for test set
        threshold: Float threshold for binarizing probabilities
    """
    print("Generating predictions for test set...")

    # 1. Predict Probabilities
    probs = trainer.predict(test_loader)

    # 2. Load Tag Encoder
    tag_encoder = TagEncoder()
    tag_encoder.load(Config.TAG_ENCODER_PATH)

    # 3. Convert to Tags
    print(f"Converting probabilities to tags using threshold {threshold}...")
    pred_tags = tag_encoder.inverse_transform(probs, threshold=threshold)

    # 4. Load Test IDs
    test_ids = np.load(Config.TEST_IDS_PATH)

    # 5. Create DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Tags": pred_tags})

    # 6. Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    # quoting=1 corresponds to csv.QUOTE_ALL, ensuring format matches sample
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False, quoting=1)

    print("Submission generated successfully.")


def run_inference(load_cached_data=True, debug=False):
    """
    Main entry point for inference pipeline.

    Args:
        load_cached_data (bool): Whether to use existing processed data.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    set_seed(Config.SEED)

    # 1. Prepare Data
    # Ensure processed data exists
    prepare_data(load_cached_data=load_cached_data, debug=debug)

    # Get DataLoaders (only need val and test for inference)
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=debug
    )

    # 2. Initialize Model
    device = torch.device(Config.DEVICE)
    print(f"Initializing WideAndDeepModel on {device}...")

    model = WideAndDeepModel(
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        wide_dim=Config.TFIDF_MAX_FEATURES,
        num_classes=Config.NUM_CLASSES,
        filter_sizes=Config.FILTER_SIZES,
        num_filters=Config.NUM_FILTERS,
        attention_dim=Config.ATTENTION_DIM,
        dropout=Config.DROPOUT,
    ).to(device)

    # 3. Load Best Model Checkpoint
    load_checkpoint(model, filename=Config.MODEL_SAVE_PATH)

    # 4. Initialize Trainer
    # We use Trainer for its predict method and mixed precision handling
    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=val_loader,
        criterion=None,
        optimizer=None,
        device=device,
        use_amp=Config.USE_AMP,
    )

    # 5. Dynamic Thresholding
    print("Loading validation labels for threshold tuning...")
    val_labels_sparse = scipy.sparse.load_npz(Config.VAL_LABELS_PATH)

    if debug:
        # Slice labels to match debug loader size
        val_labels_sparse = val_labels_sparse[: Config.DEBUG_SIZE]

    best_threshold = find_optimal_threshold(trainer, val_loader, val_labels_sparse)

    # 6. Generate Submission
    generate_submission(trainer, test_loader, best_threshold)
