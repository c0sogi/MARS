import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import HAPSModel


def compute_global_sort(anchor_logits, code_cell_ids, md_cell_ids):
    """
    Computes the initial global sort based on Anchor Head predictions.

    Args:
        anchor_logits (torch.Tensor): Shape (1, N_md, N_code + 1).
        code_cell_ids (list): List of code cell IDs (strings).
        md_cell_ids (list): List of markdown cell IDs (strings).

    Returns:
        list: A list of dictionaries representing the sorted cells.
    """
    # Remove batch dim
    logits = anchor_logits.squeeze(0)  # (N_md, N_code + 1)

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # (N_md, N_code + 1)

    # Compute Expected Index
    # We map the class indices 0..N_code to positions.
    # Code cell i is at position i + 0.5.
    # Class k corresponds to the interval around integer k.
    # We use the expected value of the class index as the continuous position.
    n_classes = probs.shape[1]
    class_indices = np.arange(n_classes)
    expected_positions = np.sum(probs * class_indices, axis=1)

    all_cells = []

    # Add Code Cells
    for i, cid in enumerate(code_cell_ids):
        all_cells.append(
            {"cell_id": cid, "type": "code", "position": i + 0.5, "original_idx": i}
        )

    # Add Markdown Cells
    for i, cid in enumerate(md_cell_ids):
        all_cells.append(
            {
                "cell_id": cid,
                "type": "markdown",
                "position": expected_positions[i],
                "original_idx": i,  # Index into the md_embeddings tensor
            }
        )

    # Sort
    all_cells.sort(key=lambda x: x["position"])

    return all_cells


def refine_order(model, all_cells, md_embeddings, device, passes=2):
    """
    Applies Sliding Window Refinement using the Pairwise Head.

    Args:
        model (HAPSModel): The trained model.
        all_cells (list): List of cell dicts from compute_global_sort.
        md_embeddings (torch.Tensor): Shape (1, N_md, Dim).
        device (torch.device): Device.
        passes (int): Number of refinement passes.

    Returns:
        list: List of cell IDs in refined order.
    """
    # We only refine if we have MD cells
    if md_embeddings.size(1) < 2:
        return [c["cell_id"] for c in all_cells]

    # Dummy inputs for the model forward pass (we only need pairwise output)
    # The model computes projections every time.
    # We pass minimal dummy code input to satisfy the forward signature.
    dummy_code = torch.zeros((1, 1, Config.input_dim), device=device)
    dummy_code_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
    md_mask = torch.ones((1, md_embeddings.size(1)), dtype=torch.bool, device=device)

    # Refinement Loop
    for _ in range(passes):
        pairs = []
        list_indices = []

        # Identify adjacent markdown cells
        for i in range(len(all_cells) - 1):
            c1 = all_cells[i]
            c2 = all_cells[i + 1]

            if c1["type"] == "markdown" and c2["type"] == "markdown":
                # We want to check P(c1 < c2).
                # The model expects [batch_idx, idx1, idx2].
                # idx1 and idx2 are indices into md_embeddings (stored as original_idx).
                pairs.append([0, c1["original_idx"], c2["original_idx"]])
                list_indices.append(i)

        if not pairs:
            break

        # Batch Inference
        pairwise_indices = torch.tensor(pairs, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(
                dummy_code,
                dummy_code_mask,
                md_embeddings,
                md_mask,
                pairwise_indices=pairwise_indices,
            )

        logits = outputs["pairwise_logits"]  # (NumPairs,)
        probs = torch.sigmoid(logits).cpu().numpy()

        # Apply Swaps
        # We process swaps. To avoid conflict in a single pass, we skip overlapping swaps.
        swapped = False
        skip_next = False

        for k, idx_in_list in enumerate(list_indices):
            if skip_next:
                skip_next = False
                continue

            # If P(c1 < c2) < 0.5, it means c2 should precede c1. Swap.
            if probs[k] < 0.5:
                all_cells[idx_in_list], all_cells[idx_in_list + 1] = (
                    all_cells[idx_in_list + 1],
                    all_cells[idx_in_list],
                )
                swapped = True
                skip_next = True  # Indices changed, skip next pair involving these

        if not swapped:
            break

    return [c["cell_id"] for c in all_cells]


def generate_predictions(model, dataset, device):
    """
    Generates predictions for the test set.

    Args:
        model (HAPSModel): Trained model.
        dataset (HAPSDataset): Test dataset.
        device (torch.device): Device.

    Returns:
        pd.DataFrame: Submission dataframe.
    """
    model.eval()
    results = []

    print(f"Inference on {len(dataset)} notebooks...")

    # We iterate manually to handle the reconstruction logic
    for i in range(len(dataset)):
        sample = dataset[i]
        nb_id = sample["notebook_id"]

        # Prepare Tensors (Add batch dim)
        code_emb = sample["code_embeddings"].unsqueeze(0).to(device)
        md_emb = sample["md_embeddings"].unsqueeze(0).to(device)

        n_code = code_emb.size(1)
        n_md = md_emb.size(1)

        code_mask = torch.ones((1, n_code), dtype=torch.bool, device=device)
        md_mask = torch.ones((1, n_md), dtype=torch.bool, device=device)

        # Get Cell IDs from the dataset's internal dataframe
        # Note: This relies on HAPSDataset structure
        nb_df = dataset.grouped.get_group(nb_id)
        code_ids = nb_df[nb_df["cell_type"] == "code"]["cell_id"].tolist()
        md_ids = nb_df[nb_df["cell_type"] == "markdown"]["cell_id"].tolist()

        # 1. Anchor Head Prediction
        with torch.no_grad():
            outputs = model(code_emb, code_mask, md_emb, md_mask, pairwise_indices=None)

        all_cells = compute_global_sort(outputs["anchor_logits"], code_ids, md_ids)

        # 2. Pairwise Refinement
        if n_md > 1:
            final_order = refine_order(model, all_cells, md_emb, device)
        else:
            final_order = [c["cell_id"] for c in all_cells]

        results.append({"id": nb_id, "cell_order": " ".join(final_order)})

    return pd.DataFrame(results)
