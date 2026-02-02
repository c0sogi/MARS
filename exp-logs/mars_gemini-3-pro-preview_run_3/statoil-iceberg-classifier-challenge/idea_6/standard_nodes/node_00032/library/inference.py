import os
import torch
from library.utils import set_seed
from library.dataset import get_dataset
from library.model import predict_with_tta
from library.trainer import generate_submission


def run_inference(
    batch_size: int = 32,
    folds: int = 5,
    limit_samples: int = None,
    model_dir: str = "./working/idea_6",
    output_path: str = "./submission/submission.csv",
):
    """
    Orchestrates the inference process using the trained ensemble.

    This function loads the test dataset, applies optional debugging limits,
    and calls the submission generator which utilizes Test-Time Augmentation (TTA)
    and model ensembling.

    Args:
        batch_size (int): The batch size for inference. Defaults to 32.
        folds (int): The number of folds to use for the ensemble. Defaults to 5.
        limit_samples (int, optional): If provided, limits the test set size for debugging.
        model_dir (str): Directory containing the fold subdirectories with checkpoints.
        output_path (str): File path to save the submission CSV.
    """
    # Ensure reproducibility
    set_seed(42)

    # Load the test dataset
    # We load cached data to save time, as preprocessing is deterministic
    test_dataset = get_dataset("test", load_cached_data=True)

    # Debugging: limit dataset size if requested
    if limit_samples is not None:
        print(f"Limiting test dataset to first {limit_samples} samples for debugging.")
        # Slice the numpy arrays in the dataset object
        test_dataset.X = test_dataset.X[:limit_samples]
        test_dataset.angles = test_dataset.angles[:limit_samples]
        test_dataset.ids = test_dataset.ids[:limit_samples]

    # Generate submission
    # This function handles model loading, TTA (via predict_with_tta),
    # ensembling across folds, and saving to CSV.
    generate_submission(
        test_dataset=test_dataset,
        folds=folds,
        batch_size=batch_size,
        model_dir=model_dir,
        output_path=output_path,
    )
