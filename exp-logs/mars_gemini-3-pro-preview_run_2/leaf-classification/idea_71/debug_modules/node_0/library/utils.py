import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import set_seed, SUBMISSION_FILE


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility by wrapping the configuration
    library's set_seed function.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def clipped_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss with specific rescaling and clipping
    as defined in the competition metric.

    The scoring mechanism requires:
    1. Rescaling probabilities so each row sums to 1.
    2. Clipping probabilities to the range [1e-15, 1 - 1e-15].
    3. Calculating the negative log likelihood.

    Args:
        y_true (array-like): True labels. Can be label indices (n_samples,)
                             or one-hot encoded (n_samples, n_classes).
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are float64 for precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale rows to sum to 1
    # Calculate row sums
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums (though unlikely in valid probability outputs)
    row_sums[row_sums == 0] = 1.0
    # Normalize
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities to avoid log(0) and strict 0/1 predictions
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 3. Compute Log Loss
    # sklearn.metrics.log_loss handles the log calculation and averaging.
    # We pass eps=epsilon to align with the clipping, although we manually clipped above.
    return log_loss(y_true, y_pred, eps=epsilon)


def save_submission(ids, probabilities, class_names, output_path=SUBMISSION_FILE):
    """
    Formats the predictions and saves them to a CSV file in the required submission format.

    Args:
        ids (array-like): Sequence of image IDs corresponding to the predictions.
        probabilities (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of species names corresponding to the columns of the probability matrix.
        output_path (str): File path where the submission CSV will be saved.
                           Defaults to the path defined in library.config.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a DataFrame with the probabilities and class names
    submission_df = pd.DataFrame(probabilities, columns=class_names)

    # Insert the 'id' column at the very beginning
    submission_df.insert(0, "id", ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
