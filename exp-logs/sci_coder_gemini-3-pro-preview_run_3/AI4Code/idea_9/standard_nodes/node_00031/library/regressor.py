import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from library.config import Config
from library.data_loader import get_data_splits, read_notebook
from library.feature_engineering import generate_features_pipeline
from library.metrics import kendall_tau_metric


def reconstruct_orders(
    df_preds: pd.DataFrame, df_metadata: pd.DataFrame, mode: str
) -> pd.DataFrame:
    """
    Reconstructs the full cell order string from markdown predictions and code skeletons.

    Args:
        df_preds: DataFrame containing ['id', 'cell_id', 'pred'] for markdown cells.
        df_metadata: DataFrame containing metadata for the notebooks.
        mode: 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: DataFrame with ['id', 'cell_order'].
    """
    # Create a dictionary for fast lookup of predictions: {nb_id: {md_id: pred_score}}
    pred_dict = {}
    if not df_preds.empty:
        for nb_id, group in df_preds.groupby("id"):
            pred_dict[nb_id] = dict(zip(group["cell_id"], group["pred"]))

    final_orders = []

    print(f"Reconstructing orders for {mode} set...")
    # Iterate over metadata to ensure we generate an entry for every notebook
    for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata)):
        nb_id = row["id"]
        file_path = row["file_path"]

        # Read notebook to get the code cell skeleton
        # Code cells in the JSON are guaranteed to be in the correct relative order
        nb_data = read_notebook(file_path)
        cell_types = nb_data.get("cell_type", {})

        if not cell_types:
            # Fallback for empty/corrupt files
            final_orders.append({"id": nb_id, "cell_order": ""})
            continue

        # Extract code cells in order
        # JSON keys in Python 3.7+ preserve insertion order.
        # The prompt states code cells are in original order in the JSON.
        code_cells = [cid for cid in cell_types if cell_types[cid] == "code"]

        # Get predictions for markdown cells in this notebook
        nb_preds = pred_dict.get(nb_id, {})

        # We will assign a "position score" to every cell and sort by it.
        # Code cells are anchors at indices 0, 1, 2...
        # We assign them scores like 0.5, 1.5, 2.5 to represent "slots".
        cells_with_scores = []

        for i, cid in enumerate(code_cells):
            cells_with_scores.append((cid, i + 0.5))

        # Markdown cells have a predicted normalized rank (0.0 to 1.0).
        # We convert this to an estimated index: rank * n_code.
        n_code = len(code_cells)

        # If there are no code cells, n_code is 0. All MDs get score 0.
        # If there are code cells, MDs get a score relative to them.
        for cid, pred in nb_preds.items():
            score = pred * n_code
            cells_with_scores.append((cid, score))

        # Sort all cells by score
        # Stable sort ensures that if multiple MD cells map to the same slot,
        # their relative order is preserved (or arbitrary but consistent).
        cells_with_scores.sort(key=lambda x: x[1])

        # Extract IDs
        sorted_ids = [x[0] for x in cells_with_scores]

        final_orders.append({"id": nb_id, "cell_order": " ".join(sorted_ids)})

    return pd.DataFrame(final_orders)


def train_lgbm_regressor(load_cached_data: bool = True):
    """
    Trains the LightGBM regressor.

    Args:
        load_cached_data: Whether to load features from cache.

    Returns:
        bst: The trained Booster model.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Metadata
    df_train_meta, df_val_meta, _ = get_data_splits()

    # 2. Load/Generate Features
    print("Loading Training Features...")
    df_train_feats = generate_features_pipeline(
        df_train_meta, mode="train", load_cached_data=load_cached_data
    )

    print("Loading Validation Features...")
    df_val_feats = generate_features_pipeline(
        df_val_meta, mode="val", load_cached_data=load_cached_data
    )

    if df_train_feats.empty or df_val_feats.empty:
        raise ValueError("Feature extraction failed. Dataframes are empty.")

    # 3. Prepare Datasets
    # Filter columns to only include features (exclude metadata)
    # Features are 'n_code', 'md_len', and the kernel features starting with 'k'
    feature_cols = [
        c for c in df_train_feats.columns if c not in ["id", "cell_id", "target"]
    ]
    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    X_train = df_train_feats[feature_cols].values
    y_train = df_train_feats["target"].values

    X_val = df_val_feats[feature_cols].values
    y_val = df_val_feats["target"].values

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 4. Train Model
    print("Starting LightGBM training...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=Config.LGBM_EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=100),
    ]

    bst = lgb.train(
        Config.LGBM_PARAMS,
        train_data,
        num_boost_round=Config.LGBM_NUM_BOOST_ROUND,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    # 5. Save Model
    print(f"Saving model to {Config.LGBM_MODEL_PATH}")
    bst.save_model(Config.LGBM_MODEL_PATH)

    # 6. Validate with Metric
    print("Evaluating on Validation Set (Kendall Tau)...")
    val_preds = bst.predict(X_val)
    df_val_feats["pred"] = val_preds

    # Reconstruct full orders
    df_val_pred_orders = reconstruct_orders(df_val_feats, df_val_meta, mode="val")

    # Compute Kendall Tau
    score = kendall_tau_metric(df_val_pred_orders, df_val_meta)
    print(f"Validation Kendall Tau: {score}")

    return bst


def predict_test_set(load_cached_data: bool = True):
    """
    Loads the model and generates the submission file.

    Args:
        load_cached_data: Whether to load features from cache.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 1. Load Metadata
    _, _, df_test_meta = get_data_splits()

    # 2. Load/Generate Features
    print("Loading Test Features...")
    df_test_feats = generate_features_pipeline(
        df_test_meta, mode="test", load_cached_data=load_cached_data
    )

    # 3. Load Model
    if not os.path.exists(Config.LGBM_MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {Config.LGBM_MODEL_PATH}. Train the model first."
        )

    print(f"Loading model from {Config.LGBM_MODEL_PATH}")
    bst = lgb.Booster(model_file=Config.LGBM_MODEL_PATH)

    # 4. Predict
    if not df_test_feats.empty:
        feature_cols = [
            c for c in df_test_feats.columns if c not in ["id", "cell_id", "target"]
        ]
        X_test = df_test_feats[feature_cols].values

        print("Predicting on Test Set...")
        preds = bst.predict(X_test)
        df_test_feats["pred"] = preds
    else:
        print("Warning: Test features empty. Submission will contain default orders.")
        df_test_feats = pd.DataFrame(columns=["id", "cell_id", "pred"])

    # 5. Reconstruct Orders
    df_submission = reconstruct_orders(df_test_feats, df_test_meta, mode="test")

    # 6. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
