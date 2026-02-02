import os
import pandas as pd
import library.config as config
import library.trainer as trainer


def predict_and_submit(models, X_test, test_ids):
    """
    Generates predictions for the test set using the trained models and saves the submission file.

    Args:
        models (list): List of trained LightGBM Booster objects.
        X_test (pd.DataFrame): Feature matrix for the test set.
        test_ids (pd.Series): Series containing the segment IDs for the test set.

    Returns:
        None
    """
    print(
        f"Generating predictions for {len(X_test)} test segments using {len(models)} models..."
    )

    # Generate predictions using the ensemble of models
    # The trainer.predict function handles iterating through models and averaging the results
    predictions = trainer.predict(models, X_test)

    # Create the submission DataFrame
    # The format requires 'segment_id' and 'time_to_eruption'
    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": predictions}
    )

    # Ensure segment_id is treated as an integer
    submission_df["segment_id"] = submission_df["segment_id"].astype("int64")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)

    # Save to CSV without the index, as per the sample submission format
    print(f"Saving submission file to {config.SUBMISSION_FILE}...")
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)

    print("Submission generation completed.")
