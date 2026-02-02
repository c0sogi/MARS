import os
from library.model import predict_and_submit


def predict(debug=False):
    """
    Executes the inference pipeline for the Pyramid Symmetry-Difference Siamese Network.

    This function generates predictions for the test set defined in the metadata.
    It performs the following steps (delegated to library.model.predict_and_submit):
    1. Sets fixed random seeds for reproducibility.
    2. Loads the trained EfficientNet-B2 Siamese model from the configured path.
    3. Iterates through the test dataset, constructing pairs of (Target, Contralateral) images.
       - Uses zero-tensors for missing contralateral views.
    4. Computes cancer probabilities using the model.
    5. Aggregates probabilities by 'prediction_id' (taking the maximum across views).
    6. Saves the final predictions to the submission CSV file.

    Args:
        debug (bool): If True, runs inference on a small subset of the test data
                      for debugging purposes. Defaults to False.
    """
    # Delegate to the robust implementation in the library to avoid code duplication
    # and ensure consistency with the training/validation logic.
    predict_and_submit(debug=debug)
