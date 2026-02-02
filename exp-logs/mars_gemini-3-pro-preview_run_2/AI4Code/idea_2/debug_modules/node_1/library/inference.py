import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import (
    MODEL_NAME,
    WORKING_DIR,
    SUBMISSION_DIR,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    MAX_LEN,
    DEVICE,
    SEED,
)
from library.utils import seed_everything, read_notebook
from library.preprocess import transform_notebook_features
from library.dataset import MarkdownRankDataset
from library.model import ContextAwareRanker


def predict_ranks(model, dataloader, device):
    """
    Generates rank predictions for the markdown cells using the trained model.

    Args:
        model (nn.Module): The trained ContextAwareRanker model.
        dataloader (DataLoader): DataLoader containing test dataset batches.
        device (torch.device): The hardware device for inference.

    Returns:
        np.array: A flat array of predicted rank scores (floats).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            all_preds.append(logits.cpu().numpy())

    if not all_preds:
        return np.array([])

    return np.concatenate(all_preds)


def assemble_submission(df_preds, test_metadata_path, output_path):
    """
    Reconstructs the full cell order for test notebooks by interleaving code cells
    (fixed order) and markdown cells (predicted order).

    Args:
        df_preds (pd.DataFrame): DataFrame containing ['id', 'cell_id', 'pred_rank'].
        test_metadata_path (str): Path to the test set metadata CSV.
        output_path (str): Path to save the final submission CSV.
    """
    # Create a fast lookup dictionary for predicted ranks: (notebook_id, cell_id) -> rank
    pred_lookup = {}
    for nb_id, cell_id, rank in df_preds[["id", "cell_id", "pred_rank"]].itertuples(
        index=False
    ):
        pred_lookup[(nb_id, cell_id)] = rank

    # Load the list of test notebooks
    df_meta = pd.read_csv(test_metadata_path)

    submission_data = []

    # Process each notebook to reconstruct order
    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        nb = read_notebook(filepath)
        if not nb:
            # Handle missing/corrupt files gracefully by submitting empty string
            submission_data.append({"id": nb_id, "cell_order": ""})
            continue

        cell_types = nb.get("cell_type", {})

        # 1. Identify Code Cells (Preserve original relative order)
        # Note: In Python 3.7+, dictionary insertion order is preserved.
        # The prompt states code cells are in the correct order in the JSON.
        code_cells = [cid for cid, ctype in cell_types.items() if ctype == "code"]
        markdown_cells = [
            cid for cid, ctype in cell_types.items() if ctype == "markdown"
        ]

        cells_with_ranks = []

        # 2. Assign Fixed Ranks to Code Cells (0.0 to 1.0)
        n_code = len(code_cells)
        if n_code > 0:
            if n_code == 1:
                code_ranks = [0.0]
            else:
                code_ranks = np.linspace(0, 1, n_code)

            for cid, r in zip(code_cells, code_ranks):
                cells_with_ranks.append((cid, r))

        # 3. Retrieve Predicted Ranks for Markdown Cells
        for cid in markdown_cells:
            # Fallback to 1.0 (end of notebook) if prediction is missing
            rank = pred_lookup.get((nb_id, cid), 1.0)
            cells_with_ranks.append((cid, rank))

        # 4. Sort All Cells by Rank
        cells_with_ranks.sort(key=lambda x: x[1])

        # 5. Extract IDs
        ordered_ids = [x[0] for x in cells_with_ranks]
        cell_order_str = " ".join(ordered_ids)

        submission_data.append({"id": nb_id, "cell_order": cell_order_str})

    # Save Submission
    df_submission = pd.DataFrame(submission_data)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def generate_submission(load_cached_data=True, debug=False):
    """
    Main function to execute the inference pipeline.

    Args:
        load_cached_data (bool): Whether to use cached feature dataframes.
        debug (bool): If True, processes a small subset of data for testing.
    """
    seed_everything(SEED)

    # 1. Prepare Test Data Features
    print("Preparing test features...")
    df_test_meta = pd.read_csv(TEST_METADATA_PATH)

    if debug:
        print("Debug mode: Processing first 100 test notebooks.")
        df_test_meta = df_test_meta.iloc[:100]

    # This function handles caching internally via parquet
    df_test = transform_notebook_features(
        df_test_meta, load_cached_data=load_cached_data
    )

    # 2. Load Model
    print(f"Loading model from {WORKING_DIR}...")
    model = ContextAwareRanker(model_name=MODEL_NAME).to(DEVICE)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(
            f"WARNING: Model checkpoint not found at {model_path}. Using initialized weights."
        )

    # 3. Run Inference
    print("Running inference...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = MarkdownRankDataset(df_test, tokenizer, max_len=MAX_LEN)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    preds = predict_ranks(model, test_loader, DEVICE)

    # Assign predictions back to the dataframe
    df_test["pred_rank"] = preds

    # 4. Assemble and Save Submission
    print("Assembling final submission...")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # If debugging, we still use the full metadata path for assembly,
    # but the lookup will only contain predictions for the debug subset.
    # Non-predicted notebooks will have empty/default entries, which is acceptable for debug.
    assemble_submission(df_test, TEST_METADATA_PATH, submission_path)
