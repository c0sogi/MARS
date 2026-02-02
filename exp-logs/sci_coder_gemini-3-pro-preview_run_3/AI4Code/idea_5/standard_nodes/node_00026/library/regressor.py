import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.config import Config
from library.utils import set_seed
from library.dataset import get_notebook_data


def train_lgbm(train_df, val_df):
    """
    Trains a LightGBM regressor on the provided training features and evaluates on validation features.

    Args:
        train_df (pd.DataFrame): Training features including target.
        val_df (pd.DataFrame): Validation features including target.

    Returns:
        lgb.Booster: Trained LightGBM model.
    """
    set_seed(Config.SEED)

    # Define feature columns (exclude identifiers and target)
    exclude_cols = ["id", "notebook_id", "target"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training LightGBM with {len(feature_cols)} features.")

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    # Initialize dataset for LightGBM
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Configure callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=Config.LGBM_PARAMS["early_stopping_rounds"]),
        lgb.log_evaluation(period=100),
    ]

    # Train the model
    print("Starting LightGBM training...")
    model = lgb.train(
        params=Config.LGBM_PARAMS,
        train_set=train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        num_boost_round=Config.LGBM_PARAMS["n_estimators"],
        callbacks=callbacks,
    )

    # Print final validation metric with full precision
    if model.best_score:
        val_rmse = model.best_score["valid"]["rmse"]
        print(f"Final Validation RMSE: {val_rmse}")

    # Save model
    model_save_path = os.path.join(Config.WORKING_DIR, "lgbm_model.txt")
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model


def predict_ranks(model, test_features_df):
    """
    Generates rank predictions for the test set features.

    Args:
        model (lgb.Booster): Trained model.
        test_features_df (pd.DataFrame): Test features.

    Returns:
        pd.DataFrame: DataFrame with 'id', 'notebook_id', and 'pred_rank'.
    """
    exclude_cols = ["id", "notebook_id", "target"]
    feature_cols = [c for c in test_features_df.columns if c not in exclude_cols]

    # Handle case where test set might be empty or features missing
    if test_features_df.empty:
        return pd.DataFrame(columns=["id", "notebook_id", "pred_rank"])

    X_test = test_features_df[feature_cols]
    preds = model.predict(X_test)

    result_df = test_features_df[["id", "notebook_id"]].copy()
    result_df["pred_rank"] = preds

    return result_df


def generate_submission(model, test_features_df, test_metadata_path):
    """
    Generates the final submission file by sorting cells based on predictions.

    Args:
        model (lgb.Booster): Trained model.
        test_features_df (pd.DataFrame): Test features.
        test_metadata_path (str): Path to test metadata CSV.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    print("Generating predictions for test set...")
    pred_df = predict_ranks(model, test_features_df)

    print("Loading test metadata...")
    test_meta = pd.read_csv(test_metadata_path)

    submission_rows = []

    # Group predictions by notebook_id for efficient lookup
    if not pred_df.empty:
        preds_grouped = dict(tuple(pred_df.groupby("notebook_id")))
    else:
        preds_grouped = {}

    print("Reconstructing cell orders...")
    for _, row in test_meta.iterrows():
        notebook_id = row["id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Get all cells from the notebook file
        # code_cells: ordered list of dicts {'id': ..., 'source': ...}
        # md_cells: unordered list of dicts
        code_cells, md_cells = get_notebook_data(file_path)

        n_code = len(code_cells)

        # Prepare sorting keys
        cell_ordering = []

        # 1. Add Code Cells with fixed positions
        # Logic: Code cell i is at position i + 0.5.
        # This creates "slots" between code cells:
        # [0.0-0.5) -> Before Code 0
        # (0.5-1.5) -> Between Code 0 and Code 1
        # ...
        for i, cell in enumerate(code_cells):
            cell_ordering.append({"id": cell["id"], "pos": i + 0.5})

        # 2. Add Markdown Cells with predicted positions
        if notebook_id in preds_grouped:
            nb_preds = preds_grouped[notebook_id]
            # Map cell_id to predicted rank
            md_pred_map = dict(zip(nb_preds["id"], nb_preds["pred_rank"]))

            for cell in md_cells:
                cell_id = cell["id"]
                # Default to end if prediction missing
                pred_norm_rank = md_pred_map.get(cell_id, 1.0)

                # Convert normalized rank [0, 1] to absolute position relative to code cells
                # pos = rank * n_code
                pos = pred_norm_rank * n_code

                cell_ordering.append({"id": cell_id, "pos": pos})
        else:
            # Fallback if no predictions found (e.g. no markdown cells in features)
            # Append markdown cells at the end
            for cell in md_cells:
                cell_ordering.append({"id": cell["id"], "pos": n_code + 1.0})

        # 3. Sort all cells by position
        cell_ordering.sort(key=lambda x: x["pos"])

        # 4. Extract IDs into space-delimited string
        sorted_ids = [x["id"] for x in cell_ordering]

        submission_rows.append({"id": notebook_id, "cell_order": " ".join(sorted_ids)})

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    return submission_df
