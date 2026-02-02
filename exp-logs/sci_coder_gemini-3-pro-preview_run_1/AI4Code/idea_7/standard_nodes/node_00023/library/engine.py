import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, compute_kendall_tau, read_notebook
from library.model import CorrectedDCAN
from library.dataset import NotebookDataset, custom_collate_fn
from library.preprocessor import Preprocessor


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        # Move batch to device
        code_emb = batch["code_emb"].to(device)
        md_emb = batch["md_emb"].to(device)
        code_mask = batch["code_mask"].to(device)
        md_mask = batch["md_mask"].to(device)
        code_lens = batch["code_lens"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # logits shape: (B, M, L+1)
        logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)

        # Flatten for CrossEntropyLoss
        # logits: (B*M, L+1), labels: (B*M)
        B, M, L_plus_1 = logits.shape
        loss = criterion(logits.view(-1, L_plus_1), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / count if count > 0 else 0.0


def validate(model, loader, df_val_meta, device):
    """
    Evaluates the model on the validation set using Kendall Tau.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"]
            md_ids_batch = batch["md_ids"]

            code_emb = batch["code_emb"].to(device)
            md_emb = batch["md_emb"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["md_mask"].to(device)
            code_lens = batch["code_lens"].to(device)

            # Forward pass
            logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)
            probs = torch.softmax(logits, dim=-1)  # (B, M, L+1)

            # Compute Expected Index (Soft Rank)
            L_plus_1 = probs.shape[-1]
            indices = torch.arange(L_plus_1, device=device).float()
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()  # (B, M)

            # Collect predictions
            for i, nb_id in enumerate(ids):
                curr_md_ids = md_ids_batch[i]
                # We interpret the score as the rank relative to code cells (0, 1, 2...)
                # Subtracting 0.5 centers the "insertion" logic, but raw score is fine for sorting.
                curr_scores = expected_pos[i, : len(curr_md_ids)]

                for m_id, score in zip(curr_md_ids, curr_scores):
                    preds_list.append(
                        {"id": nb_id, "cell_id": m_id, "rank_score": float(score)}
                    )

    if not preds_list:
        return 0.0

    df_pred_scores = pd.DataFrame(preds_list)

    # Map predictions for fast lookup
    # (nb_id, cell_id) -> score
    pred_scores_map = df_pred_scores.set_index(["id", "cell_id"])[
        "rank_score"
    ].to_dict()

    # Reconstruct orders for notebooks present in the predictions
    # We only validate on notebooks that were successfully processed (not empty)
    valid_ids = df_pred_scores["id"].unique()
    df_val_meta_subset = df_val_meta[df_val_meta["id"].isin(valid_ids)].copy()

    submission_rows = []

    for _, row in df_val_meta_subset.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        # Read notebook to identify code cells and their relative order
        # Note: We assume the code cells in the file are in the correct relative order
        try:
            nb = read_notebook(filepath)
        except Exception:
            continue

        cell_types = nb.get("cell_type", {})

        # Identify Code and Markdown cells
        # We use the order of keys in cell_types as the source order.
        # For Code cells, this is the correct relative order.
        code_cells = [c for c in cell_types if cell_types[c] == "code"]
        md_cells = [c for c in cell_types if cell_types[c] == "markdown"]

        # Assign ranks
        cell_ranks = []

        # Code cells get integer ranks: 0.0, 1.0, 2.0, ...
        for i, cid in enumerate(code_cells):
            cell_ranks.append((cid, float(i)))

        # Markdown cells get predicted ranks
        for cid in md_cells:
            # If a MD cell was truncated or missing in prediction, default to 0.0 (top)
            score = pred_scores_map.get((nb_id, cid), 0.0)
            cell_ranks.append((cid, score))

        # Sort all cells by rank
        cell_ranks.sort(key=lambda x: x[1])

        # Create order string
        pred_order = " ".join([x[0] for x in cell_ranks])
        submission_rows.append({"id": nb_id, "cell_order": pred_order})

    df_pred = pd.DataFrame(submission_rows)

    # Compute Metric
    score = compute_kendall_tau(df_val_meta_subset, df_pred)
    return score


def predict(model, loader, df_test_meta, device):
    """
    Generates predictions for the test set and returns a submission DataFrame.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"]
            md_ids_batch = batch["md_ids"]

            code_emb = batch["code_emb"].to(device)
            md_emb = batch["md_emb"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["md_mask"].to(device)
            code_lens = batch["code_lens"].to(device)

            logits = model(code_emb, md_emb, code_mask, md_mask, code_lens)
            probs = torch.softmax(logits, dim=-1)

            L_plus_1 = probs.shape[-1]
            indices = torch.arange(L_plus_1, device=device).float()
            expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

            for i, nb_id in enumerate(ids):
                curr_md_ids = md_ids_batch[i]
                curr_scores = expected_pos[i, : len(curr_md_ids)]

                for m_id, score in zip(curr_md_ids, curr_scores):
                    preds_list.append(
                        {"id": nb_id, "cell_id": m_id, "rank_score": float(score)}
                    )

    df_pred_scores = pd.DataFrame(preds_list)
    pred_scores_map = df_pred_scores.set_index(["id", "cell_id"])[
        "rank_score"
    ].to_dict()

    submission_rows = []

    for _, row in df_test_meta.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        try:
            nb = read_notebook(filepath)
        except Exception:
            # Fallback for missing files
            submission_rows.append({"id": nb_id, "cell_order": ""})
            continue

        cell_types = nb.get("cell_type", {})
        code_cells = [c for c in cell_types if cell_types[c] == "code"]
        md_cells = [c for c in cell_types if cell_types[c] == "markdown"]

        cell_ranks = []
        for i, cid in enumerate(code_cells):
            cell_ranks.append((cid, float(i)))

        for cid in md_cells:
            score = pred_scores_map.get((nb_id, cid), 0.0)
            cell_ranks.append((cid, score))

        cell_ranks.sort(key=lambda x: x[1])
        pred_order = " ".join([x[0] for x in cell_ranks])

        submission_rows.append({"id": nb_id, "cell_order": pred_order})

    return pd.DataFrame(submission_rows)


def run_pipeline():
    """
    Main execution pipeline: Preprocessing -> Training -> Inference.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Preprocessing
    print("Running Preprocessor...")
    preprocessor = Preprocessor()
    preprocessor.process_all(load_cached_data=True)

    # 2. Datasets & Loaders
    print("Loading Datasets...")
    train_ds = NotebookDataset(Config.TRAIN_FEATURES_PATH, is_test=False)
    val_ds = NotebookDataset(Config.VAL_FEATURES_PATH, is_test=False)
    test_ds = NotebookDataset(Config.TEST_FEATURES_PATH, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = CorrectedDCAN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop
    best_score = -1.0
    patience_counter = 0
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, df_val_meta, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Kendall Tau: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved with score {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # 5. Inference
    print("Running Inference on Test Set...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    df_submission = predict(model, test_loader, df_test_meta, device)

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
