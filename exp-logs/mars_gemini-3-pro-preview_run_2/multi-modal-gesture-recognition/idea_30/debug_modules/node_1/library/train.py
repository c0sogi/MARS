import os
import shutil
from library.config import NUM_EPOCHS, BATCH_SIZE, SUBMISSION_DIR
from library.model import run_pipeline


def train_and_predict(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    Orchestrates the training and prediction pipeline.

    This function utilizes the pre-implemented GestureRecognitionModel in library.model,
    which contains the logic for:
    - train_epoch: Computing aggregated Multi-Stage Deep Supervision loss.
    - validate: Evaluating the model using Levenshtein distance.
    - fit: Managing the training loop with Early Stopping.
    - predict: Generating sequence predictions with Median Filtering.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
    """
    print(
        f"Starting training pipeline with epochs={epochs}, batch_size={batch_size}..."
    )

    # Execute the end-to-end pipeline defined in the library
    # This handles data loading (with caching), training, validation, and prediction generation.
    run_pipeline(epochs=epochs, batch_size=batch_size)

    # The pipeline saves the submission to library.config.SUBMISSION_DIR.
    # We ensure a copy is available at ./submission/submission.csv as per specific task requirements.
    source_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    target_dir = "./submission"
    target_path = os.path.join(target_dir, "submission.csv")

    if os.path.exists(source_path):
        try:
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(source_path, target_path)
            print(f"Final submission successfully copied to {target_path}")
        except Exception as e:
            print(f"Warning: Failed to copy submission to {target_path}. Error: {e}")
            print(f"Submission is available at {source_path}")
    else:
        print(f"Error: Submission file not found at {source_path}")
