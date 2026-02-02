import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, read_notebook
from library.model import CorrectedDCAN
from library.dataset import NotebookDataset, custom_collate_fn


def predict_and_rank(model, loader, device):
    """
    Runs inference on the test set and computes soft ranks (Expected Index)
    for markdown cells.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        pd.DataFrame: DataFrame containing ['id', 'cell_id', 'rank_score'].
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"]
            md_ids_batch = batch["md_ids"]

            code_emb = batch["code_emb"].to(device)
            md_emb = batch["md_emb"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["md_mask"].to(device)
            code_lens = batch["code_lens"].to(device)

            # Forward pass: logits (B, M, L+1)
            # L+1 accounts for the L code cells + the EOS position
            logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)

            # Compute probabilities over the possible positions
            probs = torch.softmax(logits, dim=-1)

            # Compute Expected Index (Soft Rank)
            # The indices represent positions relative to code cells (0 to L)
            # We calculate the weighted average index
            L_plus_1 = probs.shape[-1]
            indices = torch.arange(L_plus_1, device=device).float()

            # expected_pos shape: (B, M)
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

            for i, nb_id in enumerate(ids):
                curr_md_ids = md_ids_batch[i]
                # Extract scores corresponding to the valid markdown cells in this notebook
                curr_scores = expected_pos[i, : len(curr_md_ids)]

                for m_id, score in zip(curr_md_ids, curr_scores):
                    # We store the raw expected index as the rank score.
                    # Code cells will be assigned integer ranks 0.0, 1.0, 2.0...
                    # A score of 0.5 means the markdown cell is predicted to be between code cell 0 and 1.
                    preds_list.append(
                        {"id": nb_id, "cell_id": m_id, "rank_score": float(score)}
                    )

    return pd.DataFrame(preds_list)


def generate_submission_dataframe(df_scores, df_meta):
    """
    Reconstructs the cell order for each notebook by interleaving markdown cells
    (sorted by predicted rank) into the fixed sequence of code cells.

    Args:
        df_scores (pd.DataFrame): Predictions with ['id', 'cell_id', 'rank_score'].
        df_meta (pd.DataFrame): Metadata containing ['id', 'filepath'].

    Returns:
        pd.DataFrame: Submission DataFrame with ['id', 'cell_order'].
    """
    # Create a map for fast lookup: (id, cell_id) -> score
    if df_scores.empty:
        pred_scores_map = {}
    else:
        pred_scores_map = df_scores.set_index(["id", "cell_id"])["rank_score"].to_dict()

    submission_rows = []

    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        try:
            # We read the notebook to identify code cells and their original order
            nb = read_notebook(filepath)
        except Exception:
            # Fallback if file read fails
            submission_rows.append({"id": nb_id, "cell_order": ""})
            continue

        cell_types = nb.get("cell_type", {})

        # Identify Code and Markdown cells
        # The prompt states: "The code cells are in their original (correct) order."
        # We rely on the insertion order of keys in the JSON (Python 3.7+ guarantees this).
        code_cells = [c for c in cell_types if cell_types[c] == "code"]
        md_cells = [c for c in cell_types if cell_types[c] == "markdown"]

        cell_ranks = []

        # Assign integer ranks to code cells: 0.0, 1.0, 2.0, ...
        # This fixes the skeleton of the notebook.
        for i, cid in enumerate(code_cells):
            cell_ranks.append((cid, float(i)))

        # Assign predicted ranks to markdown cells
        for cid in md_cells:
            # Default to 0.0 (top) if prediction is missing for some reason
            score = pred_scores_map.get((nb_id, cid), 0.0)
            cell_ranks.append((cid, score))

        # Sort all cells by rank
        # Stable sort is preferred if ranks are identical
        cell_ranks.sort(key=lambda x: x[1])

        # Create space-delimited string
        pred_order = " ".join([x[0] for x in cell_ranks])
        submission_rows.append({"id": nb_id, "cell_order": pred_order})

    return pd.DataFrame(submission_rows)


def run_inference(
    model_path=Config.MODEL_PATH,
    features_path=Config.TEST_FEATURES_PATH,
    metadata_path=Config.TEST_METADATA_PATH,
    submission_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device_name=Config.DEVICE,
):
    """
    Main inference pipeline function.

    Args:
        model_path (str): Path to the saved model checkpoint.
        features_path (str): Path to the precomputed test features (parquet).
        metadata_path (str): Path to the test metadata CSV.
        submission_path (str): Path to save the final submission CSV.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        device_name (str): Device to use ('cuda' or 'cpu').
    """
    set_seed(Config.SEED)
    device = torch.device(device_name)

    print(f"Initializing Inference on {device}...")

    # 1. Load Dataset
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Test features not found at {features_path}. Please run preprocessing first."
        )

    print(f"Loading test features from {features_path}...")
    test_ds = NotebookDataset(features_path, is_test=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = CorrectedDCAN().to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random initialization (for debugging/fallback)."
        )

    # 3. Predict
    print("Running prediction loop...")
    df_scores = predict_and_rank(model, test_loader, device)

    # 4. Generate Submission
    print("Generating submission file...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {metadata_path}.")

    df_test_meta = pd.read_csv(metadata_path)
    df_submission = generate_submission_dataframe(df_scores, df_test_meta)

    # 5. Save
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
