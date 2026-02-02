import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.model import DC_AN
from library.dataset import CachedNotebookDataset, collate_fn
from library.train import get_ordered_cells
from library.preprocess import generate_embeddings


def predict_and_sort(model, dataloader, device):
    """
    Runs inference on the dataloader using the provided model.
    Calculates the expected position for each markdown cell and reconstructs
    the cell order.

    Args:
        model (nn.Module): The trained DC_AN model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        list: A list of dictionaries containing 'id' and 'cell_order'.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            code_emb = batch["code_emb"].to(device)
            code_lens = batch["code_lens"].to(device)
            md_emb = batch["md_emb"].to(device)
            md_mask = batch["md_mask"].to(device)

            ids = batch["ids"]
            batch_code_ids = batch["code_ids"]
            batch_md_ids = batch["md_ids"]

            # Forward pass
            # logits: [B, M, L+1]
            logits = model(code_emb, code_lens, md_emb, md_mask)

            # Calculate Expected Position (Soft Ranking)
            # We compute the center of mass of the probability distribution
            probs = torch.softmax(logits, dim=-1)
            L_plus_1 = probs.size(-1)
            indices = torch.arange(L_plus_1, device=device).float()

            # expected_pos: [B, M]
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

            # Reconstruct order for each notebook in the batch
            for i, nb_id in enumerate(ids):
                c_ids = batch_code_ids[i]
                m_ids = batch_md_ids[i]
                num_md = len(m_ids)

                if num_md == 0:
                    # If no markdown cells, order is just code cells
                    pred_order = c_ids
                else:
                    # Get scores for the valid markdown cells in this sample
                    m_scores = expected_pos[i, :num_md]
                    # Interleave markdown cells into code anchors based on scores
                    pred_order = get_ordered_cells(c_ids, m_ids, m_scores)

                predictions.append({"id": nb_id, "cell_order": " ".join(pred_order)})

    return predictions


def run_inference(
    config=None, load_cached_data=True, output_path=None, model_path=None
):
    """
    Main driver function to generate the submission file.

    Args:
        config (Config, optional): Configuration object. Defaults to Config.
        load_cached_data (bool): Whether to use existing cached embeddings.
        output_path (str, optional): Path to save the submission CSV.
        model_path (str, optional): Path to the trained model weights.
    """
    if config is None:
        config = Config

    # Determine paths
    test_features_path = config.TEST_FEATURES_PATH
    if output_path is None:
        output_path = config.SUBMISSION_PATH
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH

    # 1. Ensure Embeddings Exist
    # If the parquet file doesn't exist, we must generate it regardless of load_cached_data flag
    if not os.path.exists(test_features_path) or not load_cached_data:
        print("Generating test embeddings...")
        generate_embeddings(load_cached_data=load_cached_data)
    else:
        print(f"Using cached test features at {test_features_path}")

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # 3. Load Model
    print(f"Loading model from {model_path}...")
    model = DC_AN(config).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 4. Setup DataLoader
    print("Loading test dataset...")
    test_ds = CachedNotebookDataset(test_features_path, config)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Run Prediction
    print(f"Predicting for {len(test_ds)} notebooks...")
    predictions = predict_and_sort(model, test_loader, device)

    # 6. Save Submission
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_submission = pd.DataFrame(predictions)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
