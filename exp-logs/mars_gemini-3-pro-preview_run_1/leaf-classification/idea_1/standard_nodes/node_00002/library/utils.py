import os
import random
import numpy as np
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and environment variables to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(
    predictions, test_ids, label_encoder, output_path=Config.SUBMISSION_FILE
):
    """
    Formats and saves the submission file based on the sample submission structure.

    Args:
        predictions (np.ndarray): The predicted probabilities for the test set.
                                  Shape should be (n_samples, n_classes).
        test_ids (list or np.array or pd.Series): The IDs corresponding to the test samples.
        label_encoder (LabelEncoder): The fitted LabelEncoder used to encode the target variable.
        output_path (str): Path to save the submission CSV. Defaults to Config.SUBMISSION_FILE.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load sample submission to ensure correct column order
    # The competition requires specific column ordering
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Create DataFrame from predictions
    # label_encoder.classes_ provides the class names corresponding to prediction indices
    submission_df = pd.DataFrame(predictions, columns=label_encoder.classes_)

    # Insert the ID column at the beginning
    submission_df.insert(0, Config.ID_COL, test_ids)

    # Ensure the columns match the sample submission order exactly
    # This handles potential differences in sorting between LabelEncoder and the sample file
    submission_df = submission_df[sample_sub.columns]

    # Clip probabilities to avoid log loss extremes as per task description
    # Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15)
    epsilon = Config.PROB_CLIP_EPSILON
    feature_cols = [col for col in submission_df.columns if col != Config.ID_COL]
    submission_df[feature_cols] = submission_df[feature_cols].clip(
        lower=epsilon, upper=1.0 - epsilon
    )

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
