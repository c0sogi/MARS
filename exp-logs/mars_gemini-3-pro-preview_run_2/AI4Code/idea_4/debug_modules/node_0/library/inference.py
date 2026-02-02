import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything
from library.data_processing import load_data, NotebookDataset
from library.feature_engineering import create_sparse_features
from library.models import TransformerRanker, RidgeRanker


def assemble_submission(
    df_pred: pd.DataFrame, metadata_df: pd.DataFrame, input_dir: str = Config.INPUT_DIR
) -> pd.DataFrame:
    """
    Assembles the final submission dataframe by combining predicted markdown ranks
    with fixed code cell anchors.

    Args:
        df_pred (pd.DataFrame): DataFrame containing ['id', 'cell_id', 'pred_rank'].
        metadata_df (pd.DataFrame): DataFrame containing ['id', 'filepath'].
        input_dir (str): Root directory for input files.

    Returns:
        pd.DataFrame: DataFrame containing ['id', 'cell_order'].
    """
    # Ensure IDs are strings for consistent matching
    df_pred = df_pred.copy()
    df_pred["id"] = df_pred["id"].astype(str)
    df_pred["cell_id"] = df_pred["cell_id"].astype(str)

    # Create a lookup dictionary for predictions: notebook_id -> {cell_id: pred_rank}
    # Grouping by ID first is significantly faster than filtering inside the loop
    pred_lookup = {}
    for nid, group in df_pred.groupby("id"):
        pred_lookup[nid] = dict(zip(group["cell_id"], group["pred_rank"]))

    submission_data = []

    # Iterate through test notebooks using metadata
    for _, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Assembling Submission"
    ):
        nb_id = str(row["id"])
        filepath = row["filepath"]
        full_path = os.path.join(input_dir, filepath)

        # 1. Read Notebook JSON to get Code Cells (Structure)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                nb_json = json.load(f)
        except Exception as e:
            # Fallback for corrupted files (unlikely in competition data)
            submission_data.append({"id": nb_id, "cell_order": ""})
            continue

        cell_types = nb_json.get("cell_type", {})

        # Extract code cells; these form the fixed skeleton of the notebook
        code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]
        n_code = len(code_cells)

        cells_with_ranks = []

        # 2. Assign Anchor Ranks to Code Cells
        # Strategy: Distribute code cells equidistantly from 0.0 to 1.0
        if n_code == 0:
            pass
        elif n_code == 1:
            # If only one code cell, place it in the middle
            cells_with_ranks.append((code_cells[0], 0.5))
        else:
            for i, cid in enumerate(code_cells):
                # Rank = index / (N - 1)
                r = i / (n_code - 1)
                cells_with_ranks.append((cid, r))

        # 3. Retrieve Predicted Ranks for Markdown Cells
        md_preds = pred_lookup.get(nb_id, {})

        for cid, rank in md_preds.items():
            cells_with_ranks.append((cid, rank))

        # 4. Sort all cells by Rank
        # This interleaves the markdown cells into the code cell skeleton
        cells_with_ranks.sort(key=lambda x: x[1])

        # 5. Format Output
        sorted_order = " ".join([x[0] for x in cells_with_ranks])
        submission_data.append({"id": nb_id, "cell_order": sorted_order})

    return pd.DataFrame(submission_data)


def run_inference(debug: bool = False):
    """
    Main function to run the inference pipeline:
    1. Load Test Data
    2. Load Models
    3. Generate Predictions (Ensemble)
    4. Assemble and Save Submission
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting Inference (Debug={debug})...")

    # --- 1. Load Data ---
    # Load test dataframe (markdown cells)
    test_df = load_data("test", debug=debug)

    # Load test metadata (for file paths)
    test_metadata = pd.read_csv(Config.TEST_METADATA_PATH)
    if debug:
        test_metadata = test_metadata[test_metadata["id"].isin(test_df["id"].unique())]

    # Generate sparse features (uses saved vectorizer)
    print("Generating sparse features for test set...")
    test_sparse = create_sparse_features(test_df, "test")

    # Prepare Dense Data Loader
    print("Preparing dense data loader...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    test_dataset = NotebookDataset(test_df, tokenizer, max_len=Config.MAX_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 2. Load Models ---
    print("Loading models...")

    # Ridge Model
    ridge_model = RidgeRanker()
    ridge_model.load()  # Automatically loads from Config.RIDGE_MODEL_PATH

    # Transformer Model
    transformer_model = TransformerRanker(
        model_name=Config.MODEL_NAME, pretrained=False
    )
    transformer_model.load(
        device=device
    )  # Automatically loads from Config.TRANSFORMER_MODEL_PATH
    transformer_model.to(device)
    transformer_model.eval()

    # --- 3. Generate Predictions ---
    print("Running inference...")

    # A. Transformer Predictions
    trans_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Transformer Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = transformer_model(input_ids, attention_mask)
            trans_preds.append(outputs.cpu().numpy())

    trans_preds = np.concatenate(trans_preds)

    # B. Ridge Predictions
    print("Ridge Inference...")
    ridge_preds = ridge_model.predict(test_sparse)

    # C. Ensemble
    alpha = Config.ENSEMBLE_ALPHA
    print(f"Ensembling predictions (Alpha={alpha})...")
    final_preds = alpha * ridge_preds + (1 - alpha) * trans_preds

    # Assign predictions back to dataframe
    test_df["pred_rank"] = final_preds

    # --- 4. Assemble Submission ---
    print("Assembling submission...")
    df_pred_subset = test_df[["id", "cell_id", "pred_rank"]]

    df_submission = assemble_submission(df_pred_subset, test_metadata)

    # --- 5. Save ---
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
