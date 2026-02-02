import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from bisect import bisect
from library.config import Config


def train_regressor(train_features, val_features):
    """
    Trains a LightGBM regressor to predict the normalized rank of markdown cells.

    Args:
        train_features (pd.DataFrame): Training features.
        val_features (pd.DataFrame): Validation features.

    Returns:
        lgb.Booster: The trained model.
    """
    print("Preparing LightGBM datasets...")

    feature_cols = ["best_match_loc", "center_of_mass", "sim_max", "n_code", "md_len"]
    target_col = "target"

    X_train = train_features[feature_cols]
    y_train = train_features[target_col]

    X_val = val_features[feature_cols]
    y_val = val_features[target_col]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    print("Starting LightGBM training...")

    callbacks = [
        lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        Config.LGBM_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Save model
    os.makedirs(os.path.dirname(Config.LGBM_MODEL_PATH), exist_ok=True)
    model.save_model(Config.LGBM_MODEL_PATH)
    print(f"Model saved to {Config.LGBM_MODEL_PATH}")

    return model


def predict_rank(model, features_df):
    """
    Generates predictions using the trained model.

    Args:
        model (lgb.Booster): Trained LightGBM model.
        features_df (pd.DataFrame): Features to predict on.

    Returns:
        np.array: Predicted normalized ranks.
    """
    feature_cols = ["best_match_loc", "center_of_mass", "sim_max", "n_code", "md_len"]

    # Handle empty dataframe edge case
    if features_df.empty:
        return np.array([])

    return model.predict(features_df[feature_cols])


def compute_kendall_tau(ground_truth, predicted):
    """
    Computes the Kendall Tau correlation between two lists of cell IDs.
    Formula: 1 - 4 * (number of swaps) / (n * (n - 1))
    """
    n = len(ground_truth)
    if n <= 1:
        return 1.0

    # Map cell_id to its index in ground_truth
    gt_map = {cid: i for i, cid in enumerate(ground_truth)}

    # Create a list of ranks for the predicted order based on ground truth
    # If a cell in predicted is not in gt (should not happen), ignore or handle
    pred_ranks = [gt_map[cid] for cid in predicted if cid in gt_map]

    # Count inversions (swaps)
    swaps = 0
    seen = []
    for rank in pred_ranks:
        # bisect_right returns the insertion point to maintain order
        # The number of elements to the right in 'seen' that are greater than 'rank'
        # is effectively counted by len(seen) - insertion_index
        # But for inversion counting (how many elements to the left are greater),
        # we actually want to know how many elements seen so far are greater than current.
        # Standard inversion count:
        # For array A, pair (i, j) is inversion if i < j and A[i] > A[j].
        # Here we iterate j. We want count of i < j such that seen[i] > rank.
        # bisect gives index where rank fits. Elements > rank are at indices [idx, len).
        idx = bisect(seen, rank)
        swaps += len(seen) - idx
        seen.insert(idx, rank)

    total_pairs = n * (n - 1) // 2
    # The formula in the prompt is 1 - 4 * S / (n * (n-1))
    # Note: n * (n - 1) = 2 * total_pairs.
    # So 4 * S / (2 * total_pairs) = 2 * S / total_pairs.
    # Standard Kendall Tau is 1 - 2 * S / total_pairs.
    # The prompt formula is equivalent to the standard definition.

    score = 1 - 4 * swaps / (n * (n - 1))
    return score


def validate_model(model, val_df, val_features):
    """
    Validates the model by computing the average Kendall Tau score on the validation set.

    Args:
        model (lgb.Booster): Trained model.
        val_df (pd.DataFrame): Validation metadata (contains ground truth cell_order).
        val_features (pd.DataFrame): Validation features (markdown cells).

    Returns:
        float: Mean Kendall Tau score.
    """
    print("Validating model...")

    # Predict
    preds = predict_rank(model, val_features)
    val_features = val_features.copy()
    val_features["pred_rank"] = preds

    # Group predictions by notebook ID
    pred_groups = val_features.groupby("id")

    scores = []

    for _, row in val_df.iterrows():
        nb_id = row["id"]
        gt_order = row["cell_order"].split()

        # Identify code cells in ground truth (their relative order is fixed)
        # Note: We don't have cell_type in val_df, but we can infer from features
        # or we assume we need to read the notebook.
        # However, to be efficient, we rely on the fact that 'val_features' only contains Markdown.
        # Code cells are those in gt_order NOT in val_features for this notebook.

        if nb_id in pred_groups.groups:
            nb_preds = pred_groups.get_group(nb_id)
            md_preds = dict(zip(nb_preds["cell_id"], nb_preds["pred_rank"]))
            n_code = nb_preds["n_code"].iloc[0]
        else:
            md_preds = {}
            n_code = 0  # Fallback, though rare if processed correctly

        # Separate code and markdown from GT
        code_cells = [c for c in gt_order if c not in md_preds]

        # If n_code derived from features differs from actual count in GT, rely on GT
        # (Feature extraction ensures n_code is accurate per notebook)

        # Construct positions
        cells_with_pos = []

        # Code cells get fixed positions: 0.5, 1.5, 2.5, ...
        for i, cid in enumerate(code_cells):
            cells_with_pos.append((cid, i + 0.5))

        # Markdown cells get predicted positions: pred * n_code
        # n_code used in scaling should match the one used in training features
        # If n_code is 0, just use raw prediction (though usually means no anchors)
        scaling_factor = len(code_cells) if len(code_cells) > 0 else 1.0

        for cid, pred in md_preds.items():
            pos = pred * scaling_factor
            cells_with_pos.append((cid, pos))

        # Sort by position
        cells_with_pos.sort(key=lambda x: x[1])
        predicted_order = [c for c, _ in cells_with_pos]

        # Compute Metric
        score = compute_kendall_tau(gt_order, predicted_order)
        scores.append(score)

    mean_score = np.mean(scores)
    print(f"Validation Kendall Tau: {mean_score}")
    return mean_score


def generate_submission(test_df, test_features, model):
    """
    Generates the submission file.

    Args:
        test_df (pd.DataFrame): Test metadata.
        test_features (pd.DataFrame): Test features.
        model (lgb.Booster): Trained model.
    """
    print("Generating submission...")

    preds = predict_rank(model, test_features)
    test_features = test_features.copy()
    test_features["pred_rank"] = preds

    pred_groups = test_features.groupby("id")

    submission_data = []

    # We need to read test notebooks to get the code cells (anchors)
    # since test_df only has file paths.
    from library.data_utils import read_notebook

    for _, row in test_df.iterrows():
        nb_id = row["id"]

        # Read notebook to get all cells and types
        nb_data = read_notebook(row["file_path"])
        if nb_data is None:
            # Fallback: empty prediction
            submission_data.append({"id": nb_id, "cell_order": ""})
            continue

        cell_types = nb_data.get("cell_type", {})
        # In test set, we assume the keys in the JSON are the cell IDs.
        # The task description says "The code cells are in their original (correct) order."
        # So we can extract code cells in the order they appear in the JSON (or keys).
        # Wait, JSON keys are unordered in standard dicts before Py3.7, but usually
        # provided data implies an order or we just filter types.
        # The prompt says: "The code cells are in their original (correct) order. The markdown cells have been shuffled and placed after the code cells."
        # This implies the file content might have code first then markdown?
        # Actually, standard approach: Filter for code cells, keep their relative order as found in the file/keys.

        all_cells = list(cell_types.keys())
        code_cells = [c for c in all_cells if cell_types[c] == "code"]

        # Get predictions for markdown
        if nb_id in pred_groups.groups:
            nb_preds = pred_groups.get_group(nb_id)
            md_preds = dict(zip(nb_preds["cell_id"], nb_preds["pred_rank"]))
        else:
            md_preds = {}
            # If no features found (e.g. no markdown), check for markdown cells in file
            md_in_file = [c for c in all_cells if cell_types[c] == "markdown"]
            for m in md_in_file:
                md_preds[m] = 0.0  # Default to start if no prediction

        # Construct positions
        cells_with_pos = []

        # Code cells: 0.5, 1.5, ...
        for i, cid in enumerate(code_cells):
            cells_with_pos.append((cid, i + 0.5))

        # Markdown cells
        scaling_factor = len(code_cells) if len(code_cells) > 0 else 1.0

        for cid, pred in md_preds.items():
            pos = pred * scaling_factor
            cells_with_pos.append((cid, pos))

        # Sort
        cells_with_pos.sort(key=lambda x: x[1])
        final_order = [c for c, _ in cells_with_pos]

        submission_data.append({"id": nb_id, "cell_order": " ".join(final_order)})

    sub_df = pd.DataFrame(submission_data)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
