import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.model import SemanticAnchorClassifier
from library.data import EmbeddingManager, NotebookDataset, collate_fn


def generate_submission(config: Config = None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        config (Config, optional): Configuration object. If None, defaults are used.
    """
    if config is None:
        config = Config()

    # Ensure reproducibility
    set_seed(config.seed)

    print(f"Initializing inference on device: {config.device}")

    # ---------------------------------------------------------
    # 1. Data Loading
    # ---------------------------------------------------------
    # The EmbeddingManager handles caching automatically.
    # It will process test notebooks and save/load features from test_features.parquet
    manager = EmbeddingManager(config)
    df_test = manager.process_data("test", load_cached_data=True)

    # Create Dataset and DataLoader
    test_dataset = NotebookDataset(df_test)

    # We use a larger batch size for inference as we don't need to store gradients
    inference_batch_size = config.val_batch_size * 2

    test_loader = DataLoader(
        test_dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 2. Model Loading
    # ---------------------------------------------------------
    model = SemanticAnchorClassifier(config)

    if os.path.exists(config.model_save_path):
        print(f"Loading model weights from {config.model_save_path}")
        state_dict = torch.load(config.model_save_path, map_location=config.device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {config.model_save_path}. Using random weights."
        )

    model.to(config.device)
    model.eval()

    # ---------------------------------------------------------
    # 3. Inference Loop
    # ---------------------------------------------------------
    predictions = []

    # Map to retrieve code IDs for reconstruction
    # The dataframe contains JSON strings for IDs
    test_code_map = {
        row["id"]: json.loads(row["code_ids"]) for _, row in df_test.iterrows()
    }

    print(f"Running inference on {len(df_test)} notebooks...")

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            code_emb = batch["code_embeddings"].to(config.device)
            md_emb = batch["markdown_embeddings"].to(config.device)
            ids = batch["ids"]
            batch_md_ids = batch["markdown_ids"]

            # Calculate Expected Positions
            if md_emb.size(1) > 0:
                # Forward pass: (Batch, Num_MD, Num_Code + 1)
                logits = model(md_emb, code_emb)

                # Calculate probabilities: (Batch, Num_MD, Num_Code + 1)
                probs = torch.softmax(logits, dim=-1)

                # Calculate Expected Index (Soft Rank)
                # We sum (prob * class_index) to get a continuous position estimate
                num_classes = probs.size(-1)
                class_indices = torch.arange(num_classes, device=config.device).float()

                # Shape: (Batch, Num_MD)
                expected_positions = torch.sum(probs * class_indices, dim=-1)
                expected_positions = expected_positions.cpu().numpy()
            else:
                # Handle notebooks with no markdown cells
                expected_positions = np.array([])

            # Reconstruct Order for each notebook in the batch
            for i, nb_id in enumerate(ids):
                # Retrieve original code cell IDs
                nb_code_ids = test_code_map.get(nb_id, [])

                # Retrieve markdown cell IDs for this notebook
                nb_md_ids = batch_md_ids[i]

                # List to store tuples of (position, cell_id)
                cells_with_pos = []

                # Assign fixed positions to code cells
                # Code cells are anchors at 0.5, 1.5, 2.5, etc.
                # This corresponds to the intervals defined by the indices 0, 1, 2...
                for idx, cid in enumerate(nb_code_ids):
                    cells_with_pos.append((idx + 0.5, cid))

                # Assign predicted positions to markdown cells
                if len(nb_md_ids) > 0:
                    # Valid length (ignoring padding in batch)
                    valid_len = len(nb_md_ids)

                    # Extract predictions for this specific notebook
                    # Note: expected_positions[i] might be padded, but we slice by valid_len
                    if expected_positions.size > 0:
                        nb_preds = expected_positions[i][:valid_len]

                        for md_id, pos in zip(nb_md_ids, nb_preds):
                            cells_with_pos.append((pos, md_id))
                    else:
                        # Fallback if logic fails (shouldn't happen with valid_len check)
                        for md_id in nb_md_ids:
                            cells_with_pos.append((0.0, md_id))

                # Sort all cells by their position value
                cells_with_pos.sort(key=lambda x: x[0])

                # Extract the ordered list of IDs
                final_order = [x[1] for x in cells_with_pos]

                # Convert to space-delimited string
                cell_order_str = " ".join(final_order)

                predictions.append({"id": nb_id, "cell_order": cell_order_str})

    # ---------------------------------------------------------
    # 4. Save Submission
    # ---------------------------------------------------------
    df_submission = pd.DataFrame(predictions)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    df_submission.to_csv(config.submission_path, index=False)
    print(
        f"Submission saved to {config.submission_path} with {len(df_submission)} rows."
    )
