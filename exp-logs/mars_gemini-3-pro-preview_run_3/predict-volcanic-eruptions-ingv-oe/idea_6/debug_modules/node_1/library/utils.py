import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import METADATA_DIR, SUBMISSION_DIR, SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata(metadata_dir=METADATA_DIR):
    """
    Loads the train, validation, and test metadata CSV files.

    Args:
        metadata_dir (str): Directory containing the metadata CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(f"One or more metadata files missing in {metadata_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def save_submission(
    predictions, test_ids, output_dir=SUBMISSION_DIR, filename="submission.csv"
):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        predictions (array-like): The predicted time_to_eruption values.
        test_ids (array-like): The corresponding segment_ids.
        output_dir (str): Directory to save the submission file.
        filename (str): Name of the submission file.
    """
    os.makedirs(output_dir, exist_ok=True)

    submission_df = pd.DataFrame(
        {"segment_id": test_ids, "time_to_eruption": predictions}
    )

    # Ensure segment_id is integer
    submission_df["segment_id"] = submission_df["segment_id"].astype(int)

    output_path = os.path.join(output_dir, filename)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
