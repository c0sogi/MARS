import os
import pandas as pd
import numpy as np
from library.config import Config
from library.data_utils import get_metadata, get_notebook_cells
from library.feature_extraction import FeatureExtractor
from library.ranker import LGBMRanker


def predict_order(
    notebook_id, ranker, feature_extractor, notebook_features=None, notebook_data=None
):
    """
    Predicts the cell order for a single notebook.

    Args:
        notebook_id (str): The notebook ID.
        ranker (LGBMRanker): The ranker instance.
        feature_extractor (FeatureExtractor): The feature extractor instance.
        notebook_features (pd.DataFrame, optional): Pre-computed features for this notebook.
                                                    If None, features will be extracted on the fly.
        notebook_data (dict, optional): Pre-loaded notebook data. If None, loaded from disk.

    Returns:
        str: The predicted cell order (space-delimited).
    """
    # 1. Get Notebook Content
    if notebook_data is None:
        # Resolve path from metadata
        # We primarily check test metadata, but fallback logic can be added if needed
        df_test = get_metadata("test")
        row = df_test[df_test["id"] == notebook_id]

        if not row.empty:
            rel_path = row.iloc[0]["file_path"]
        else:
            # Fallback assumption for path structure
            rel_path = f"test/{notebook_id}.json"

        try:
            notebook_data = get_notebook_cells(notebook_id, rel_path)
        except Exception:
            # If file cannot be read, return empty string
            return ""

    code_cells = notebook_data["code_cells"]
    markdown_cells = notebook_data["markdown_cells"]

    code_ids = [c["id"] for c in code_cells]
    md_ids = [m["id"] for m in markdown_cells]

    # 2. Handle Edge Cases
    # If no markdown cells, order is just code cells
    if not markdown_cells:
        return " ".join(code_ids)

    # If no code cells, order is just markdown cells (arbitrary or original order)
    if not code_cells:
        return " ".join(md_ids)

    # 3. Get Features and Predictions
    # If features not provided (single inference mode), extract them
    if notebook_features is None:
        # Construct a metadata row for the extractor
        row = pd.Series({"id": notebook_id, "file_path": f"test/{notebook_id}.json"})

        # Ensure the extractor has a loaded model
        # FeatureExtractor doesn't persist model in self usually, so we load it
        model = feature_extractor._load_model()

        # Extract features
        feats_list = feature_extractor._process_notebook(row, model)
        notebook_features = pd.DataFrame(feats_list)

        # Optimize types to match training schema
        if not notebook_features.empty:
            float_cols = [
                c
                for c in notebook_features.columns
                if notebook_features[c].dtype == "float64"
            ]
            notebook_features[float_cols] = notebook_features[float_cols].astype(
                "float32"
            )

    if notebook_features.empty:
        # Fallback if feature extraction failed
        return " ".join(code_ids + md_ids)

    # 4. Get Ranks
    # Check if 'pred_rank' is already in features (batch mode optimization)
    if "pred_rank" in notebook_features.columns:
        preds = notebook_features["pred_rank"].values
    else:
        # Predict using ranker (loads model from disk)
        preds = ranker.predict(notebook_features)

    # 5. Reconstruct Order
    # Map markdown_id to prediction
    md_scores = {}
    # Reset index to ensure alignment between dataframe rows and prediction array
    nb_feats_reset = notebook_features.reset_index(drop=True)

    for idx, row in nb_feats_reset.iterrows():
        mid = row["markdown_id"]
        score = preds[idx]
        md_scores[mid] = score

    cells_with_scores = []

    # Code cells are anchors at fixed positions: 0.5, 1.5, 2.5...
    for i, cid in enumerate(code_ids):
        cells_with_scores.append((cid, i + 0.5))

    # Markdown cells are placed based on predicted relative rank
    # Prediction is in [0, 1], so we scale by n_code to get position
    n_code = len(code_ids)
    for mid in md_ids:
        pred = md_scores.get(mid, 0.0)  # Default to 0 if missing
        pos = pred * n_code
        cells_with_scores.append((mid, pos))

    # Sort all cells by their score/position
    cells_with_scores.sort(key=lambda x: x[1])

    # Extract IDs
    final_order = [x[0] for x in cells_with_scores]
    return " ".join(final_order)


def save_submission(submission_rows, output_path):
    """
    Saves the submission rows to a CSV file.

    Args:
        submission_rows (list): List of dicts with 'id' and 'cell_order'.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame(submission_rows)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference():
    """
    Main driver for generating the submission file.
    Executes the pipeline in batch mode for efficiency.
    """
    print("Initializing Inference Pipeline...")

    # 1. Setup Components
    ranker = LGBMRanker()
    feature_extractor = FeatureExtractor()

    # Ensure model is trained
    if not os.path.exists(ranker.model_path):
        print("Model not found. Training model first...")
        ranker.train()

    # 2. Batch Feature Extraction
    print("Extracting features for test set...")
    # Leverages caching mechanism in FeatureExtractor
    df_test_features = feature_extractor.extract_features("test", load_cached_data=True)

    # 3. Batch Prediction
    # We predict all test samples at once to avoid reloading the model 20k times
    features_by_nb = None
    if not df_test_features.empty:
        print(f"Running batch prediction on {len(df_test_features)} samples...")
        preds = ranker.predict(df_test_features)
        df_test_features["pred_rank"] = preds

        # Group by notebook_id for O(1) access during reconstruction
        features_by_nb = df_test_features.groupby("notebook_id")

    # 4. Generate Orders
    df_test_meta = get_metadata("test")
    submission_rows = []

    print(f"Processing {len(df_test_meta)} notebooks...")

    for i, row in df_test_meta.iterrows():
        nb_id = row["id"]

        # Retrieve pre-computed features if available
        nb_feats = None
        if features_by_nb is not None and nb_id in features_by_nb.groups:
            nb_feats = features_by_nb.get_group(nb_id)

        # Predict order
        # We pass the pre-computed features to avoid re-extraction/re-prediction
        order = predict_order(
            notebook_id=nb_id,
            ranker=ranker,
            feature_extractor=feature_extractor,
            notebook_features=nb_feats,
        )

        submission_rows.append({"id": nb_id, "cell_order": order})

        # Logging
        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{len(df_test_meta)} notebooks")

    # 5. Save Submission
    save_submission(submission_rows, Config.Paths.SUBMISSION_PATH)
