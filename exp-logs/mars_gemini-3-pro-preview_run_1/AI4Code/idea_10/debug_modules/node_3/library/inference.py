import os
import json
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import DCAN
from library.dataset import get_dataloaders
from library.utils import save_submission


def get_test_code_cells(test_ids):
    """
    Reads test notebook JSON files to extract code cell IDs in their original order.
    This is necessary because the parquet features store embeddings but not the
    explicit list of code anchor IDs required for reconstruction.

    Args:
        test_ids (list): List of notebook IDs to process.

    Returns:
        dict: Mapping from notebook_id to list of code cell IDs.
    """
    code_maps = {}

    # Load test metadata to get filepaths
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Metadata file not found at {Config.TEST_METADATA_PATH}")
        return {}

    df_meta = pd.read_csv(Config.TEST_METADATA_PATH).set_index("id")

    # Filter for requested IDs that exist in metadata
    ids_to_process = [tid for tid in test_ids if tid in df_meta.index]

    print(f"Extracting code anchors for {len(ids_to_process)} notebooks...")

    for nb_id in ids_to_process:
        filepath = df_meta.loc[nb_id, "filepath"]
        full_path = os.path.join(Config.INPUT_DIR, filepath)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source = data.get("source", {})
            cell_type = data.get("cell_type", {})

            # In Python 3.7+, dictionary insertion order is preserved.
            # This matches the order used during feature extraction.
            code_ids = [cid for cid in source if cell_type.get(cid) == "code"]
            code_maps[nb_id] = code_ids

        except Exception as e:
            print(f"Error reading notebook {nb_id}: {e}")
            code_maps[nb_id] = []

    return code_maps


def predict(
    model_path=Config.MODEL_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
):
    """
    Main inference function.
    Loads the model, generates predictions for the test set, reconstructs the cell order,
    and saves the submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (str): Computation device ('cuda' or 'cpu').
    """
    Config.set_seed(Config.SEED)

    # 1. Load Data
    # We use get_dataloaders to ensure consistent preprocessing and caching logic.
    # We only need the test_loader.
    print("Initializing test dataloader...")
    _, _, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # Access the underlying dataframe to map IDs to markdown_ids later
    test_df = test_loader.dataset.df.set_index("id")

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = DCAN().to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint not found at {model_path}. Using random initialization (debug mode)."
        )

    model.eval()

    # 3. Inference Loop
    preds_map = {}  # Map: nb_id -> {md_cell_id: expected_rank}
    all_test_ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["id"]
            all_test_ids.extend(ids)

            # Move data to device
            code_emb = batch["code_embeddings"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["markdown_mask"].to(device)
            code_lens = batch["code_lens"].to(device)
            md_lens = batch["markdown_lens"].to(device)

            # Forward Pass
            # Logits: (Batch, MaxMD, MaxCode + 1)
            logits = model(code_emb, md_emb, code_mask, md_mask, code_lens, md_lens)

            # Compute Soft Ranks (Expected Index)
            probs = torch.softmax(logits, dim=-1)
            max_cls = probs.size(2)
            indices = torch.arange(max_cls, device=device).float()

            # Expected rank = Sum(prob_i * i)
            expected_ranks = torch.sum(probs * indices, dim=-1).cpu().numpy()

            # Map predictions back to specific markdown cell IDs
            for i, nb_id in enumerate(ids):
                if nb_id not in test_df.index:
                    continue

                # Retrieve the list of markdown IDs for this notebook
                md_ids = test_df.loc[nb_id]["markdown_ids"]

                # Slice the padded output to the valid number of markdown cells
                valid_len = len(md_ids)
                nb_ranks = expected_ranks[i][:valid_len]

                preds_map[nb_id] = dict(zip(md_ids, nb_ranks))

    # 4. Retrieve Code Anchors
    # We need the ordered list of code cell IDs to interleave the markdown cells correctly.
    unique_ids = list(set(all_test_ids))
    code_maps = get_test_code_cells(unique_ids)

    # 5. Reconstruct Final Order
    print("Reconstructing cell orders...")
    submission_ids = []
    submission_orders = []

    for nb_id in all_test_ids:
        md_ranks = preds_map.get(nb_id, {})
        code_ids = code_maps.get(nb_id, [])

        cells_with_scores = []

        # Assign Code cells fixed ranks: 0.5, 1.5, 2.5 ...
        # This places them firmly at integer intervals, allowing markdown cells
        # (with continuous expected ranks) to fall in between.
        for r, cid in enumerate(code_ids):
            cells_with_scores.append((cid, r + 0.5))

        # Assign Markdown cells their predicted expected rank
        for mid, rank in md_ranks.items():
            cells_with_scores.append((mid, rank))

        # Sort all cells by their rank
        cells_with_scores.sort(key=lambda x: x[1])

        # Extract just the cell IDs
        final_order = [x[0] for x in cells_with_scores]

        submission_ids.append(nb_id)
        submission_orders.append(final_order)

    # 6. Save Submission
    save_submission(submission_ids, submission_orders)
    print("Inference completed successfully.")
