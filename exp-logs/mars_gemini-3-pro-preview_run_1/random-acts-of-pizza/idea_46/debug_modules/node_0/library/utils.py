import os
import pandas as pd
from library.config import Config, set_seed


def save_submission(request_ids, predictions, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the predictions to a CSV file conforming to the competition requirements.

    Args:
        request_ids (list or np.ndarray): A sequence of request identifiers.
        predictions (list or np.ndarray): A sequence of predicted probabilities (real-valued).
        output_path (str): The file path where the submission CSV will be saved.
                           Defaults to the path defined in Config.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a DataFrame with the required columns
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
