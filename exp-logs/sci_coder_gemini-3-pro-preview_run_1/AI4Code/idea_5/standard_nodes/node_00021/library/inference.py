import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloader
from library.model import DCAN
from library.utils import set_seed, get_ordered_cell_ids


def predict(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set and creates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached feature files.
        batch_size (int): Batch size for inference.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Inference running on device: {device}")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Data
    # get_dataloader handles feature precomputation if cache is missing/invalid
    print("Loading test dataloader...")
    test_loader = get_dataloader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        load_cached_data=load_cached_data,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = DCAN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    results = []
    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            code_features = batch["code_features"].to(device)
            code_mask = batch["code_mask"].to(device)
            markdown_features = batch["markdown_features"].to(device)
            markdown_mask = batch["markdown_mask"].to(device)

            # Metadata for reconstruction
            ids = batch["ids"]
            batch_code_ids = batch["code_ids"]
            batch_markdown_ids = batch["markdown_ids"]

            # Forward Pass
            # logits: (Batch, Num_Markdown, Num_Code + 1)
            logits = model(code_features, code_mask, markdown_features, markdown_mask)

            # Compute Soft Ranking (Expected Index)
            # Probabilities: (B, M, L+1)
            probs = torch.softmax(logits, dim=-1)

            # Create indices tensor [0, 1, ..., L]
            # L_plus_1 is the size of the last dimension (number of code cells + sink)
            L_plus_1 = probs.size(-1)
            indices = torch.arange(L_plus_1, device=device, dtype=torch.float32)

            # Expected Index: sum(p * i) -> (B, M)
            # This gives a continuous score representing the predicted index in the code sequence
            expected_indices = torch.sum(probs * indices, dim=-1)
            expected_indices = expected_indices.cpu().numpy()

            # Reconstruct Order for each notebook in the batch
            for i, nb_id in enumerate(ids):
                c_ids = batch_code_ids[i]
                m_ids = batch_markdown_ids[i]

                # Get scores for this notebook's markdown cells
                # The batch is padded, so we slice to the actual number of markdown cells
                num_md = len(m_ids)
                scores = expected_indices[i][:num_md]

                # Generate the ordered string string
                # Logic: Code cell i is at position i + 0.5.
                # Markdown cell with score S is placed relative to these fixed anchors.
                pred_order_str = get_ordered_cell_ids(c_ids, m_ids, scores)

                results.append({"id": nb_id, "cell_order": pred_order_str})

    # 5. Save Submission
    print(f"Generating submission file with {len(results)} entries...")
    df_submission = pd.DataFrame(results)

    # Ensure columns are in correct order
    df_submission = df_submission[["id", "cell_order"]]

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
