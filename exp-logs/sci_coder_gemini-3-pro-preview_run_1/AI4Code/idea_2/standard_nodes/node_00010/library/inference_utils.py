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


def refine_order(model, all_cells, md_emb, device, passes=1):
    """
    Refines the order of cells using the Pairwise Head of the model.
    It iterates through the sequence and swaps adjacent markdown cells if the
    pairwise head predicts they are out of order.

    Args:
        model (HAPSModel): The trained model.
        all_cells (list): List of cell dicts (from compute_global_sort).
        md_emb (torch.Tensor): Raw markdown embeddings (1, N_md, InputDim).
        device (torch.device): Device to run computation on.
        passes (int): Number of bubble-sort-like passes.

    Returns:
        list: A list of cell IDs in the refined order.
    """
    # 1. Project MD embeddings once for efficiency
    # We need to access the internal projector or run a partial forward pass.
    # Since HAPSModel.md_projector is a public attribute, we use it directly.
    model.eval()
    with torch.no_grad():
        # md_emb is (1, N_md, InputDim)
        # proj_md will be (1, N_md, ProjDim)
        proj_md = model.md_projector(md_emb)
        proj_md = proj_md.squeeze(0)  # (N_md, ProjDim)

    seq = all_cells[:]  # Shallow copy

    for _ in range(passes):
        swapped = False
        # Iterate through the sequence to find adjacent markdown cells
        for i in range(len(seq) - 1):
            c1 = seq[i]
            c2 = seq[i + 1]

            # We only refine the relative order of adjacent markdown cells.
            # If a code cell separates them, the Anchor head determines their bin.
            if c1["type"] == "markdown" and c2["type"] == "markdown":
                idx1 = c1["original_idx"]
                idx2 = c2["original_idx"]

                vec1 = proj_md[idx1]
                vec2 = proj_md[idx2]

                # Predict pairwise order
                # The model's pairwise_bilinear outputs a score s.
                # If s > 0, it implies vec1 precedes vec2.
                # If s < 0, it implies vec2 precedes vec1.
                # Currently c1 is before c2. If score < 0, we should swap.
                score = model.pairwise_bilinear(vec1, vec2).item()

                if score < 0:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
                    swapped = True

        if not swapped:
            break

    return [c["cell_id"] for c in seq]


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
            outputs = model(code_emb, code_mask, md_emb, md_mask)

        all_cells = compute_global_sort(outputs["anchor_logits"], code_ids, md_ids)

        # 2. Pairwise Refinement
        # Only needed if there are multiple markdown cells
        if n_md > 1:
            final_order = refine_order(model, all_cells, md_emb, device, passes=2)
        else:
            final_order = [c["cell_id"] for c in all_cells]

        results.append({"id": nb_id, "cell_order": " ".join(final_order)})

    return pd.DataFrame(results)
