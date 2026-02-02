import pandas as pd
import os
from library.config import *
from library.data_loader import get_regression_data, get_inference_data
from library.model import RankPredictor, predict_notebook_order
from library.metrics import score_dataset


def assemble_ordering(model, notebook_data):
    """
    Reconstructs the full notebook order by assigning fixed, evenly spaced ranks
    to the known sequence of code cells, combining them with the predicted ranks
    of markdown cells, and sorting the union.

    Args:
        model: The trained RankPredictor model instance.
        notebook_data (dict): Dictionary containing 'code_cells' (list of ids)
                              and 'markdown_cells' (list of (id, text) tuples).

    Returns:
        str: Space-delimited string of ordered cell IDs.
    """
    # Utilize the provided library implementation which matches the logic
    # for the Interleaving Sort strategy described in the task.
    return predict_notebook_order(model, notebook_data)


def run_inference(max_train_samples=None, max_test_samples=None):
    """
    Executes the full inference pipeline: Training, Validation, and Submission.

    Args:
        max_train_samples (int, optional): Limit the number of training samples for debugging.
        max_test_samples (int, optional): Limit the number of validation/test notebooks for debugging.
    """

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("Loading training data...")
    # The data loader handles caching of the processed regression data
    df_train = get_regression_data(data_type="train", max_samples=max_train_samples)

    print(f"Training model on {len(df_train)} markdown cells...")
    model = RankPredictor()
    model.fit(df_train)

    # -------------------------------------------------------------------------
    # 2. Validation Phase
    # -------------------------------------------------------------------------
    print("Loading validation data...")
    val_notebooks = get_inference_data(data_type="val", max_samples=max_test_samples)

    if val_notebooks:
        print(f"Predicting on {len(val_notebooks)} validation notebooks...")
        val_preds = []
        for nb in val_notebooks:
            # Reconstruct order using the assemble_ordering logic
            order_str = assemble_ordering(model, nb)
            val_preds.append({"id": nb["id"], "cell_order": order_str})

        df_val_pred = pd.DataFrame(val_preds)

        # Load Ground Truth for validation
        # We filter the metadata to match only the notebooks we predicted (in case of subsampling)
        df_val_meta = pd.read_csv(VAL_METADATA_PATH)
        val_ids = set(df_val_pred["id"])
        df_val_meta = df_val_meta[df_val_meta["id"].isin(val_ids)]

        # Compute and print metric with full precision
        score = score_dataset(df_val_meta, df_val_pred)
        print(f"Validation Kendall Tau: {score}")
    else:
        print("No validation data found or loaded.")

    # -------------------------------------------------------------------------
    # 3. Submission Phase
    # -------------------------------------------------------------------------
    print("Loading test data...")
    test_notebooks = get_inference_data(data_type="test", max_samples=max_test_samples)

    if test_notebooks:
        print(f"Generating submission for {len(test_notebooks)} notebooks...")
        test_preds = []
        for nb in test_notebooks:
            order_str = assemble_ordering(model, nb)
            test_preds.append({"id": nb["id"], "cell_order": order_str})

        df_submission = pd.DataFrame(test_preds)

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        df_submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print("No test data found.")
