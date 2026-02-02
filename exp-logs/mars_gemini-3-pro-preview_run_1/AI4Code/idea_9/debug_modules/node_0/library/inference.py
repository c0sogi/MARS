import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import CachedNotebookDataset, custom_collate_fn
from library.model import DualContextAnchorNetwork


def predict_and_rank(debug=False):
    """
    Runs inference on the test set, computes the optimal cell order,
    and saves the results to submission.csv.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
    """
    device = Config.DEVICE

    # 1. Load Test Features DataFrame to retrieve Cell IDs
    # We must replicate the sorting logic of CachedNotebookDataset exactly
    # to ensure the mapping between embeddings (from DataLoader) and Cell IDs (from DataFrame) is correct.
    print(f"Loading test features from {Config.TEST_FEATURES_PATH} for ID mapping...")

    # Load parquet
    df_test = pd.read_parquet(Config.TEST_FEATURES_PATH)

    if debug:
        print("Debug mode: limiting inference to first 100 notebooks.")
        unique_ids = df_test["id"].unique()[:100]
        df_test = df_test[df_test["id"].isin(unique_ids)].copy()

    # Replicate dataset.py sorting logic for test split
    # This ensures df_test rows align with the CachedNotebookDataset internal structure
    df_test = df_test.sort_values(by=["id"])

    # Create an index map for fast retrieval of cell IDs by notebook ID
    # groupby(...).indices returns {group_key: np.array(indices_into_df)}
    print("Indexing test metadata...")
    notebook_indices = df_test.groupby("id", sort=False).indices

    # 2. Setup Dataset and DataLoader
    test_dataset = CachedNotebookDataset(split="test", debug=debug)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = DualContextAnchorNetwork()
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    results = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Move inputs to device
            code_emb = batch["code_embeddings"].to(device)
            code_lens = batch["code_lens"].to(device)
            code_mask = batch["code_padding_mask"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            md_lens = batch["md_lens"].to(device)
            md_mask = batch["md_padding_mask"].to(device)
            ids = batch["id"]

            # Forward Pass
            # logits: (Batch, Max_MD, Max_Code + 1)
            logits = model(code_emb, code_lens, code_mask, md_emb, md_lens, md_mask)

            # Compute Probabilities
            probs = torch.softmax(logits, dim=-1)

            # Compute Expected Index (Soft Rank)
            # The model predicts the index k (0..N) where the MD cell should be inserted.
            # k=0: Before first code cell. k=N: After last code cell.
            max_c = probs.size(-1)
            indices = torch.arange(max_c, device=device).float()

            # Expected position: Sum(prob * index) -> (Batch, Max_MD)
            pred_scores = torch.sum(probs * indices, dim=-1)
            pred_scores_np = pred_scores.cpu().numpy()

            # Process each notebook in the batch
            for i, nb_id in enumerate(ids):
                # Retrieve the number of valid markdown cells for this notebook
                n_md = md_lens[i].item()

                # Get predictions for the valid markdown cells
                scores = pred_scores_np[i, :n_md]

                # Retrieve Cell IDs from DataFrame using the index map
                if nb_id in notebook_indices:
                    row_indices = notebook_indices[nb_id]

                    # Extract the dataframe chunk for this notebook
                    nb_df = df_test.iloc[row_indices]

                    # Separate Code and Markdown IDs
                    # Note: The order here must match the order in dataset.py's __getitem__
                    # dataset.py separates by: nb_types == "code" and nb_types == "markdown"
                    # Since we are using the same dataframe source and indices, this is consistent.
                    code_cells = nb_df[nb_df["cell_type"] == "code"]["cell_id"].values
                    md_cells = nb_df[nb_df["cell_type"] == "markdown"]["cell_id"].values

                    # Sanity check
                    if len(md_cells) != n_md:
                        # This should theoretically not happen if the parquet file is consistent
                        pass

                    # --- Ranking Logic ---
                    # 1. Assign "fixed" positions to code cells.
                    # Code cell i is at position i + 0.5 (between slot i and slot i+1)
                    code_positions = np.arange(len(code_cells)) + 0.5

                    # 2. Use predicted scores for markdown cells.
                    # A score of S implies the MD cell is at index S in the code sequence.

                    # 3. Combine both sets of cells
                    all_ids = np.concatenate([code_cells, md_cells])
                    all_scores = np.concatenate([code_positions, scores])

                    # 4. Sort based on scores
                    # We use mergesort for stability, though scores are floats so it rarely matters
                    sort_order = np.argsort(all_scores, kind="mergesort")
                    sorted_cell_ids = all_ids[sort_order]

                    # 5. Format output
                    cell_order_str = " ".join(sorted_cell_ids)

                    results.append({"id": nb_id, "cell_order": cell_order_str})
                else:
                    print(f"Warning: ID {nb_id} not found in features DataFrame.")

    # 4. Save Submission
    submission_df = pd.DataFrame(results)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
