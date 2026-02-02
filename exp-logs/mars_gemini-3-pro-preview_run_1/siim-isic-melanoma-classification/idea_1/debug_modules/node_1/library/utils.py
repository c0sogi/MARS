import os
import pandas as pd
from library.config import seed_everything


def save_submission(image_names, predictions, save_path):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        image_names (list-like): A list or array of image identifiers (strings).
        predictions (list-like): A list or array of predicted probabilities (floats).
        save_path (str): The full path where the CSV file should be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Construct the DataFrame
    submission_df = pd.DataFrame({"image_name": image_names, "target": predictions})

    # Save to CSV without the index
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
