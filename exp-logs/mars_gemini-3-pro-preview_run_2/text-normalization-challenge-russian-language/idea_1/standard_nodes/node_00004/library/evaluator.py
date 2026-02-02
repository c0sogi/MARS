import pandas as pd
from library.config import COL_AFTER, COL_ID, SUBMISSION_FILE_PATH
from library.utils import setup_logger, save_submission
from library.data_loader import load_and_process_data

# Initialize logger
logger = setup_logger("evaluator")


def calculate_accuracy(model, data):
    """
    Calculates the prediction accuracy of the model on the provided data.

    Args:
        model: The trained model instance with a predict method.
        data (pd.DataFrame): The dataset containing features and the target column 'after'.

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    # Validate input data
    if COL_AFTER not in data.columns:
        raise ValueError(
            f"Dataset must contain '{COL_AFTER}' column for accuracy calculation."
        )

    logger.info(f"Predicting on {len(data)} samples...")

    # Generate predictions
    predictions = model.predict(data)

    # Ensure strict string comparison
    # Fill NaNs with empty strings to avoid type errors during comparison if any
    y_pred = predictions.fillna("").astype(str)
    y_true = data[COL_AFTER].fillna("").astype(str)

    # Calculate exact match accuracy
    correct_count = (y_pred == y_true).sum()
    total_count = len(data)

    if total_count == 0:
        accuracy = 0.0
    else:
        accuracy = correct_count / total_count

    # Print full precision as requested
    print(f"Validation Accuracy: {accuracy}")

    return accuracy


def evaluate_on_split(model, split="val", limit=None, load_cached_data=True):
    """
    Loads the specified data split and evaluates the model.

    Args:
        model: The trained model instance.
        split (str): The split to load ('train', 'val').
        limit (int, optional): Max rows to load for debugging.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        float: The accuracy score.
    """
    logger.info(f"Loading data for split: {split}")
    df = load_and_process_data(
        split=split, load_cached_data=load_cached_data, limit=limit
    )

    return calculate_accuracy(model, df)


def generate_submission(
    model,
    split="test",
    limit=None,
    load_cached_data=True,
    output_path=SUBMISSION_FILE_PATH,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained model instance.
        split (str): The split to load (usually 'test').
        limit (int, optional): Max rows to load for debugging.
        load_cached_data (bool): Whether to use cached data.
        output_path (str): Path to save the submission file.
    """
    logger.info(f"Loading data for split: {split}")
    df = load_and_process_data(
        split=split, load_cached_data=load_cached_data, limit=limit
    )

    logger.info("Generating predictions for submission...")
    predictions = model.predict(df)

    # Verify ID column exists
    if COL_ID not in df.columns:
        raise ValueError(
            f"Dataset must contain '{COL_ID}' column for submission generation."
        )

    # Prepare submission DataFrame
    submission_df = pd.DataFrame({COL_ID: df[COL_ID], COL_AFTER: predictions})

    logger.info(f"Saving submission to {output_path}...")
    save_submission(submission_df, filepath=output_path)
