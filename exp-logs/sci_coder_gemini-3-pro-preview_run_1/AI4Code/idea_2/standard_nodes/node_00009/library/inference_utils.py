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

        final_order = [c["cell_id"] for c in all_cells]

        results.append({"id": nb_id, "cell_order": " ".join(final_order)})

    return pd.DataFrame(results)
