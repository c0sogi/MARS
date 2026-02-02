import os
import torch
from library.config import Config
from library.model import run_training, predict_and_submit


def train(debug=False, epochs=None):
    """
    Executes the training pipeline for the Pyramid Symmetry-Difference Siamese Network.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
        epochs (int, optional): Override the number of training epochs defined in Config.

    Returns:
        float: The best Probabilistic F1 (pF1) score achieved on the validation set.
    """
    # Override Config hyperparameters if provided
    if epochs is not None:
        Config.EPOCHS = epochs

    # Delegate to the library's training orchestration function
    # This handles:
    # - Dataset initialization
    # - Model setup (EfficientNet-B2 Siamese)
    # - Optimizer (AdamW) and Scheduler (CosineAnnealing)
    # - Loss (BCEWithLogitsLoss with pos_weight=47.0)
    # - Training loop with no gradient clipping
    # - Validation with pF1 tracking
    # - Checkpointing
    best_pf1 = run_training(debug=debug)

    return best_pf1


def inference(debug=False):
    """
    Executes the inference pipeline and generates the submission file.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
    """
    # Delegate to the library's inference function
    # This handles:
    # - Loading the best saved model
    # - Processing the test set
    # - Generating predictions
    # - Aggregating by prediction_id (max probability)
    # - Saving to submission.csv
    predict_and_submit(debug=debug)
