import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    BATCH_SIZE,
    ALPHA,
    SEED,
    VECTORIZER_PATH,
    RIDGE_MODEL_PATH,
    TRANSFORMER_MODEL_PATH,
    SUBMISSION_DIR,
    TEST_METADATA_PATH,
    INPUT_DIR,
)
from library.utils import seed_everything
from library.data_processing import load_notebook_data
from library.feature_extraction import SparseVectorizer, DenseInputProcessor
from library.model_definitions import RidgeRegressorWrapper, TransformerRegressor
from library.training_engine import MarkdownDataset, create_collate_fn

# Ensure reproducibility
seed_everything(SEED)


def predict_ranks(partition="test", load_cached_data=True):
    """
    Generates rank predictions for the specified partition using the ensemble
    of Sparse (Ridge) and Dense (Transformer) models.

    Args:
        partition (str): 'test', 'val', or 'train'.
        load_cached_data (bool): Whether to load processed data from cache.

    Returns:
        pd.DataFrame: The dataframe with an added 'pred_rank' column.
    """
    print(f"--- Starting Inference on {partition} set ---")

    # 1. Load Data
    df = load_notebook_data(partition, load_cached_data=load_cached_data)

    if df.empty:
        print("Warning: Loaded dataframe is empty.")
        df["pred_rank"] = []
        return df

    # 2. Sparse Stream Inference
    print("Running Sparse Stream (Ridge) Inference...")

    # Load Vectorizer
    vectorizer = SparseVectorizer()
    vectorizer.load(VECTORIZER_PATH)

    # Transform Text
    # Ensure text is string (handle NaNs if any)
    texts = df["text"].fillna("").astype(str).tolist()
    X_sparse = vectorizer.transform(texts)

    # Load Ridge Model
    ridge_model = RidgeRegressorWrapper()
    ridge_model.load(RIDGE_MODEL_PATH)

    # Predict
    preds_sparse = ridge_model.predict(X_sparse)

    # 3. Dense Stream Inference
    print("Running Dense Stream (Transformer) Inference...")

    # Initialize Processor and Dataset
    processor = DenseInputProcessor()
    dataset = MarkdownDataset(df)
    collate_fn = create_collate_fn(processor)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load Transformer Model
    transformer_model = TransformerRegressor()
    transformer_model.load(TRANSFORMER_MODEL_PATH, device=DEVICE)
    transformer_model.to(DEVICE)
    transformer_model.eval()

    preds_dense = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            # Forward pass (using mixed precision for consistency/speed)
            with torch.cuda.amp.autocast():
                outputs = transformer_model(input_ids, attention_mask)

            preds_dense.extend(outputs.cpu().numpy().tolist())

    preds_dense = np.array(preds_dense)

    # 4. Ensemble
    print(f"Ensembling predictions with ALPHA={ALPHA} (Sparse) ...")

    # Combine predictions
    # Final Rank = ALPHA * Ridge + (1 - ALPHA) * Transformer
    final_preds = ALPHA * preds_sparse + (1 - ALPHA) * preds_dense

    # Add to dataframe
    df["pred_rank"] = final_preds

    return df


def anchor_sort(df_preds):
    """
    Sorts cells for each notebook by combining ordered code cells (anchors)
    with predicted markdown cells.

    Args:
        df_preds (pd.DataFrame): Dataframe containing 'id', 'cell_id', 'pred_rank',
                                 and 'code_cell_ids'.

    Returns:
        pd.DataFrame: Dataframe with 'id' and 'cell_order' columns ready for submission.
    """
    print("Sorting cells using Anchor-Based Sorting...")

    submission_rows = []

    # Group by notebook ID
    if not df_preds.empty:
        grouped = df_preds.groupby("id")

        for nb_id, group in grouped:
            # Get code cells (anchors)
            # The 'code_cell_ids' column contains a space-delimited string of code cell IDs
            # We take the first value as they are identical for the group
            code_ids_str = group["code_cell_ids"].iloc[0]
            code_ids = code_ids_str.split() if code_ids_str else []

            # Get markdown cells and their predicted ranks
            md_cells = group[["cell_id", "pred_rank"]].values.tolist()

            # Assign ranks to code cells
            # Strategy: Equidistant ranks from 0.0 to 1.0
            n_code = len(code_ids)
            if n_code == 0:
                # No code cells, just sort markdown by rank
                combined_cells = md_cells
            else:
                if n_code == 1:
                    # Single code cell at 0.0 (start of notebook logic from data processing)
                    code_ranks = [0.0]
                else:
                    code_ranks = np.linspace(0.0, 1.0, n_code).tolist()

                # Create (cell_id, rank) pairs for code cells
                code_cells_with_rank = list(zip(code_ids, code_ranks))

                # Combine
                combined_cells = code_cells_with_rank + md_cells

            # Sort by rank (x[1])
            combined_cells.sort(key=lambda x: x[1])

            # Extract ordered cell IDs
            ordered_ids = [x[0] for x in combined_cells]
            cell_order_str = " ".join(ordered_ids)

            submission_rows.append({"id": nb_id, "cell_order": cell_order_str})

    # Handle notebooks that might have ONLY code cells (no markdown)
    # These notebooks do not appear in df_preds because data_processing only returns markdown rows.

    # Load test metadata to get all IDs
    if os.path.exists(TEST_METADATA_PATH):
        df_meta = pd.read_csv(TEST_METADATA_PATH)
        all_test_ids = set(df_meta["id"].unique())
        processed_ids = set([row["id"] for row in submission_rows])
        missing_ids = all_test_ids - processed_ids

        if missing_ids:
            print(f"Processing {len(missing_ids)} notebooks with no markdown cells...")
            import json

            for missing_id in missing_ids:
                # Find filepath
                meta_row = df_meta[df_meta["id"] == missing_id]
                if meta_row.empty:
                    continue
                filepath = os.path.join(INPUT_DIR, meta_row.iloc[0]["filepath"])

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        nb_json = json.load(f)
                    # Infer code order from JSON structure
                    cell_types = nb_json.get("cell_type", {})
                    # In Python 3.7+, dict order is insertion order
                    code_ids = [
                        cid for cid, ctype in cell_types.items() if ctype == "code"
                    ]
                    submission_rows.append(
                        {"id": missing_id, "cell_order": " ".join(code_ids)}
                    )
                except Exception as e:
                    print(f"Error processing missing notebook {missing_id}: {e}")
                    submission_rows.append({"id": missing_id, "cell_order": ""})

    return pd.DataFrame(submission_rows)


def generate_submission_file(
    output_path=os.path.join(SUBMISSION_DIR, "submission.csv")
):
    """
    Orchestrates the inference pipeline and generates the submission file.
    """
    # 1. Predict Ranks
    df_preds = predict_ranks(partition="test", load_cached_data=True)

    # 2. Sort and Format
    df_submission = anchor_sort(df_preds)

    # 3. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    print(f"Total notebooks in submission: {len(df_submission)}")
