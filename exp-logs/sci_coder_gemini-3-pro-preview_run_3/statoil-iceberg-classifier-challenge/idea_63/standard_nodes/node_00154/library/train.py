import os
import library.config
import library.data
import library.model


class Trainer:
    """
    Orchestrates the training and submission process for the RTI-CNN model.
    Wraps the functional logic provided in library.model to allow for
    configurable execution.
    """

    def __init__(self, epochs=None, debug=False):
        """
        Initialize the Trainer with optional configuration overrides.

        Args:
            epochs (int, optional): Number of training epochs. Overrides config default.
            debug (bool): If True, runs on a small subset of data for debugging.
        """
        self.epochs = epochs
        self.debug = debug

        # Apply configuration overrides to the imported modules
        # We must update the specific module attributes because they were imported
        # into those namespaces (e.g., 'from library.config import EPOCHS')
        if self.epochs is not None:
            library.model.EPOCHS = self.epochs
            # Also update config for consistency, though model uses its own import
            library.config.EPOCHS = self.epochs

        if self.debug:
            library.data.DEBUG = True
            library.config.DEBUG = True
            print(f"Debug mode enabled. Using {library.config.DEBUG_SAMPLES} samples.")

    def train(self):
        """
        Executes the 5-fold cross-validation training pipeline.
        """
        print(f"Starting training for {library.model.EPOCHS} epochs...")
        library.model.train_all_folds()

    def generate_submission(self):
        """
        Generates predictions for the test set using the trained models
        and saves the submission file.
        """
        print("Generating submission...")
        library.model.generate_submission()

    def run(self):
        """
        Convenience method to run the full pipeline: training followed by submission.
        """
        self.train()
        self.generate_submission()
