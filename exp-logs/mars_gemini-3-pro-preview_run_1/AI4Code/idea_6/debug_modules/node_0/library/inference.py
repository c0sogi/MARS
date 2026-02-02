import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import NotebookEmbeddingDataset
from library.model import DualContextAnchorNetwork


def predict_and_sort():
    """
    Performs inference on the test set, reconstructs cell orders, and saves the submission file.
    """
    # 1. Setup
    device = Config.DEVICE
    print(f"Inference running on device: {device}")

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # 2. Data Loading
    print("Loading test dataset...")
    # We use the test split. max_size can be used if debugging is enabled in Config.
    max_size = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None
    test_dataset = NotebookEmbeddingDataset(split="test", max_size=max_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Order doesn't strictly matter for inference, but False is safer for debugging
        num_workers=Config.NUM_WORKERS,
        collate_fn=NotebookEmbeddingDataset.collate_fn,
    )

    # 3. Model Loading
    print("Loading trained model...")
    model = DualContextAnchorNetwork().to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train the model first."
        )

    # Load weights
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    submission_data = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            code_embs = batch["code_embeddings"].to(device)
            md_embs = batch["markdown_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["markdown_mask"].to(device)

            # Metadata for reconstruction
            ids = batch["ids"]
            code_cell_ids_batch = batch["code_cell_ids"]
            md_cell_ids_batch = batch["markdown_cell_ids"]

            # Forward Pass
            # Logits: (Batch, N_md, N_code + 1)
            logits = model(code_embs, md_embs, code_mask, md_mask)

            # Compute Probabilities
            probs = torch.softmax(logits, dim=-1)  # (B, N_md, N_code + 1)

            # Compute Expected Index (Soft Ranking)
            # Create index tensor: [0, 1, 2, ..., N_code]
            n_classes = logits.size(-1)
            indices = torch.arange(n_classes, device=device).float()

            # Expected Rank: (B, N_md)
            # Value E means the markdown cell is predicted to be before code cell at index E.
            expected_ranks = torch.sum(probs * indices, dim=-1)
            expected_ranks_np = expected_ranks.cpu().numpy()

            # Reconstruct Order for each notebook in the batch
            batch_size = len(ids)
            for i in range(batch_size):
                nb_id = ids[i]
                code_ids = code_cell_ids_batch[i]
                md_ids = md_cell_ids_batch[i]

                # Get ranks for valid markdown cells (ignore padding)
                num_md = len(md_ids)
                nb_ranks = expected_ranks_np[i, :num_md]

                # Create a list of (position, cell_id) tuples
                cells_with_pos = []

                # 1. Code Cells: Fixed anchors at integer positions 0.0, 1.0, 2.0, ...
                for idx, cid in enumerate(code_ids):
                    cells_with_pos.append((float(idx), cid))

                # 2. Markdown Cells: Placed relative to anchors
                # If rank is E, it means "before code cell E".
                # To sort correctly, we place it at E - 0.5.
                # Example: E=0 -> -0.5 (Before Code 0)
                # Example: E=1 -> 0.5 (Between Code 0 and Code 1)
                # Example: E=N -> N - 0.5 (After Code N-1)
                for idx, cid in enumerate(md_ids):
                    rank = nb_ranks[idx]
                    cells_with_pos.append((rank - 0.5, cid))

                # Sort by position
                cells_with_pos.sort(key=lambda x: x[0])

                # Extract ordered IDs
                ordered_ids = [cid for _, cid in cells_with_pos]
                cell_order_str = " ".join(ordered_ids)

                submission_data.append({"id": nb_id, "cell_order": cell_order_str})

    # 4. Save Submission
    print("Saving submission file...")
    df_submission = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total notebooks processed: {len(df_submission)}")
