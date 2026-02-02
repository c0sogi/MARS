import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, load_object
from library.data_loader import DataLoader


def generate_submission(
    working_dir=Config.WORKING_DIR, submission_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set using the ensemble of saved fold-models.
    Loads test data, iterates through saved pipelines, averages predictions, and saves to CSV.

    Args:
        working_dir (str): Directory containing saved model artifacts (default: Config.WORKING_DIR).
        submission_path (str): Path to save the output CSV (default: Config.SUBMISSION_PATH).
    """
    # Ensure reproducibility
    set_seed(Config.SEED)
    print("\nStarting Submission Generation...")

    # 1. Load Test Data
    # DataLoader handles metadata reading, raw JSON merging, and basic cleaning
    loader = DataLoader()
    # load_merged_data checks cache first, then processes if needed
    df_test = loader.load_merged_data(split="test")

    # The pipeline is designed to accept the full DataFrame and select columns internally
    X_test = df_test

    # 2. Iterate through saved fold models
    fold_preds = []

    print(f"Loading {Config.N_FOLDS} fold models from {working_dir}...")

    for fold in range(Config.N_FOLDS):
        # Construct path to the saved pipeline artifact
        model_filename = f"fold_{fold}_pipeline.joblib"
        model_path = os.path.join(working_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Ensure training has been run."
            )

        print(
            f"Processing Fold {fold + 1}/{Config.N_FOLDS}: Loading {model_filename}..."
        )

        try:
            # Load the full pipeline (Preprocessing + Classifier)
            pipeline = load_object(model_path)

            # Generate probabilities for the positive class (1)
            # predict_proba returns [prob_0, prob_1]
            preds = pipeline.predict_proba(X_test)[:, 1]
            fold_preds.append(preds)

        except Exception as e:
            print(f"Error processing fold {fold}: {e}")
            raise e

    # 3. Ensemble Aggregation
    # Average predictions across all folds (Ensemble of Ensembles)
    if not fold_preds:
        raise RuntimeError("No predictions were generated.")

    avg_preds = np.mean(fold_preds, axis=0)

    # 4. Create Submission DataFrame
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": avg_preds}
    )

    # 5. Save Submission
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Submission Head:")
    print(submission.head())
