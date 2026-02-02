import os
import torch
from library.config import Config
from library.data import get_dataloaders
from library.trainer import Trainer


class DebugLoader:
    """
    A wrapper around a DataLoader to limit the number of batches yielded.
    Used for debugging purposes to verify the pipeline without running on the full dataset.
    """

    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


def run_inference(output_path=Config.SUBMISSION_PATH, max_batches=None):
    """
    Executes the inference pipeline to generate the submission file.

    Args:
        output_path (str): The file path where the submission CSV will be saved.
                           Defaults to Config.SUBMISSION_PATH.
        max_batches (int, optional): If set, limits the number of batches processed.
                                     Useful for debugging. Defaults to None (process all).
    """
    # 1. Load DataLoaders
    # We retrieve the dictionary of loaders and select the 'test' loader.
    dataloaders = get_dataloaders()
    test_loader = dataloaders["test"]

    # 2. Handle Debugging
    # If max_batches is specified, wrap the loader to limit iteration.
    if max_batches is not None:
        print(f"Debugging mode enabled: processing only {max_batches} batches.")
        test_loader = DebugLoader(test_loader, max_batches)

    # 3. Initialize Trainer
    # The Trainer class handles model initialization (ResNet18Baseline),
    # device configuration, and seeding.
    trainer = Trainer()

    # 4. Generate Submission
    # This method loads the best checkpoint defined in Config.MODEL_CHECKPOINT_PATH,
    # runs the forward pass on the test_loader, applies Max Pooling aggregation
    # for Naive MIL, and saves the results to output_path.
    trainer.generate_submission(test_loader, output_path=output_path)
