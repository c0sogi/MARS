import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import kendall_tau_metric, save_submission
from library.dataset import get_dataloaders
from library.model import DCAN


class Trainer:
    def __init__(self):
        Config.set_seed(Config.SEED)
        self.device = Config.DEVICE

        # Initialize Model
        self.model = DCAN().to(self.device)

        # Optimizer & Loss
        # We disable warmup and use constant LR as per instructions
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # CrossEntropyLoss with ignore_index for padded labels (-100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_score = -float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in train_loader:
            # Move data to device
            code_emb = batch["code_embeddings"].to(self.device)
            md_emb = batch["markdown_embeddings"].to(self.device)
            code_mask = batch["code_mask"].to(self.device)
            md_mask = batch["markdown_mask"].to(self.device)
            code_lens = batch["code_lens"].to(self.device)
            md_lens = batch["markdown_lens"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Logits shape: (Batch, MaxMD, MaxCode + 1)
            logits = self.model(
                code_emb, md_emb, code_mask, md_mask, code_lens, md_lens
            )

            # Reshape for CrossEntropyLoss
            # Flatten Batch and MaxMD dimensions
            B, Lm, NumClasses = logits.shape
            logits_flat = logits.view(-1, NumClasses)
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            if torch.isnan(loss):
                continue

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * B
            count += B

        epoch_loss = running_loss / count if count > 0 else 0
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.6f}")

    def validate(self, val_loader):
        self.model.eval()

        # Store predictions: id -> {md_id: expected_rank}
        preds_map = {}

        # Access dataset dataframe to retrieve markdown_ids (which are not in the batch tensor)
        val_df = val_loader.dataset.df.set_index("id")

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["id"]
                code_emb = batch["code_embeddings"].to(self.device)
                md_emb = batch["markdown_embeddings"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                md_mask = batch["markdown_mask"].to(self.device)
                code_lens = batch["code_lens"].to(self.device)
                md_lens = batch["markdown_lens"].to(self.device)

                logits = self.model(
                    code_emb, md_emb, code_mask, md_mask, code_lens, md_lens
                )

                # Compute Expected Index (Soft Rank)
                probs = torch.softmax(logits, dim=-1)  # (B, Lm, Lc+1)
                max_cls = probs.size(2)
                indices = torch.arange(max_cls, device=self.device).float()

                # expected_pos = sum(p_i * i)
                expected_ranks = torch.sum(probs * indices, dim=-1)  # (B, Lm)
                expected_ranks = expected_ranks.cpu().numpy()

                # Map back to IDs
                for i, nb_id in enumerate(ids):
                    if nb_id not in val_df.index:
                        continue

                    md_ids = val_df.loc[nb_id]["markdown_ids"]
                    nb_ranks = expected_ranks[i]

                    # Slice to valid length
                    valid_len = len(md_ids)
                    valid_ranks = nb_ranks[:valid_len]

                    preds_map[nb_id] = dict(zip(md_ids, valid_ranks))

        # Reconstruct Orders for Metric Calculation
        df_gt = pd.read_csv(Config.VAL_METADATA_PATH)
        pred_rows = []

        for idx, row in df_gt.iterrows():
            nb_id = row["id"]
            gt_order = row["cell_order"].split()

            if nb_id not in preds_map:
                pred_rows.append({"id": nb_id, "cell_order": row["cell_order"]})
                continue

            md_ranks = preds_map[nb_id]

            # Identify Code Cells (Anchors) from GT
            # Code cell at index i (relative to code sequence) gets rank i + 0.5
            code_cells = [cid for cid in gt_order if cid not in md_ranks]

            cells_with_scores = []
            for i, cid in enumerate(code_cells):
                cells_with_scores.append((cid, i + 0.5))

            for mid, score in md_ranks.items():
                cells_with_scores.append((mid, score))

            # Sort by rank
            cells_with_scores.sort(key=lambda x: x[1])

            pred_order = [x[0] for x in cells_with_scores]
            pred_rows.append({"id": nb_id, "cell_order": " ".join(pred_order)})

        df_pred = pd.DataFrame(pred_rows)

        score = kendall_tau_metric(df_pred, df_gt)
        print(f"Validation Kendall Tau: {score}")
        return score

    def run(self):
        train_loader, val_loader, _ = get_dataloaders()

        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

        for epoch in range(Config.NUM_EPOCHS):
            self.train_one_epoch(train_loader, epoch)
            val_score = self.validate(val_loader)

            if val_score > self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved with score {self.best_score}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Score: {self.best_score}")


def get_test_code_cells(test_ids):
    """
    Reads test JSONs to extract code cell IDs in order.
    Necessary because parquet features don't store the ordered code IDs explicitly.
    """
    print("Extracting code cell anchors from Test JSONs...")
    code_maps = {}

    df_meta = pd.read_csv(Config.TEST_METADATA_PATH).set_index("id")

    # Filter for requested IDs to avoid unnecessary reads
    ids_to_process = [tid for tid in test_ids if tid in df_meta.index]

    for nb_id in ids_to_process:
        filepath = df_meta.loc[nb_id, "filepath"]
        full_path = os.path.join(Config.INPUT_DIR, filepath)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source = data.get("source", {})
            cell_type = data.get("cell_type", {})

            # Python 3.7+ preserves insertion order, which matches preprocess.py logic
            code_ids = []
            for cid in source.keys():
                if cell_type.get(cid) == "code":
                    code_ids.append(cid)

            code_maps[nb_id] = code_ids

        except Exception:
            code_maps[nb_id] = []

    return code_maps


def predict_and_submit():
    device = Config.DEVICE

    # Load Model
    model = DCAN().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No model checkpoint found. Using random initialization.")

    model.eval()

    _, _, test_loader = get_dataloaders()
    test_df = test_loader.dataset.df.set_index("id")

    preds_map = {}
    all_test_ids = []

    print("Running inference on Test set...")
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["id"]
            all_test_ids.extend(ids)

            code_emb = batch["code_embeddings"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_mask = batch["markdown_mask"].to(device)
            code_lens = batch["code_lens"].to(device)
            md_lens = batch["markdown_lens"].to(device)

            logits = model(code_emb, md_emb, code_mask, md_mask, code_lens, md_lens)

            probs = torch.softmax(logits, dim=-1)
            max_cls = probs.size(2)
            indices = torch.arange(max_cls, device=device).float()
            expected_ranks = torch.sum(probs * indices, dim=-1).cpu().numpy()

            for i, nb_id in enumerate(ids):
                if nb_id not in test_df.index:
                    continue
                md_ids = test_df.loc[nb_id]["markdown_ids"]
                nb_ranks = expected_ranks[i][: len(md_ids)]
                preds_map[nb_id] = dict(zip(md_ids, nb_ranks))

    # Retrieve Code Anchors
    code_maps = get_test_code_cells(list(set(all_test_ids)))

    submission_ids = []
    submission_orders = []

    print("Reconstructing final cell orders...")
    for nb_id in all_test_ids:
        md_ranks = preds_map.get(nb_id, {})
        code_ids = code_maps.get(nb_id, [])

        cells_with_scores = []

        # Assign Code cells fixed ranks: 0.5, 1.5, 2.5...
        for i, cid in enumerate(code_ids):
            cells_with_scores.append((cid, i + 0.5))

        # Assign Markdown cells their predicted expected rank
        for mid, score in md_ranks.items():
            cells_with_scores.append((mid, score))

        # Sort all cells
        cells_with_scores.sort(key=lambda x: x[1])

        final_order = [x[0] for x in cells_with_scores]

        submission_ids.append(nb_id)
        submission_orders.append(final_order)

    save_submission(submission_ids, submission_orders)


def main():
    trainer = Trainer()
    trainer.run()
    predict_and_submit()


if __name__ == "__main__":
    main()
