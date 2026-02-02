import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, compute_kendall_tau, save_checkpoint
from library.model import DCCodeBERT
from library.data_loader import get_dataloader


class Trainer:
    """
    Trainer class for the DC-CodeBERT model.
    Handles training, validation, and inference/submission generation.
    """

    def __init__(self):
        set_seed(Config.SEED)
        self.device = Config.DEVICE
        print(f"Initializing Trainer on {self.device}...")

        # Initialize Model
        self.model = DCCodeBERT().to(self.device)

        # Initialize Optimizer
        # Using AdamW with constant learning rate as per design
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        # ignore_index=-100 handles the padding in labels
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def _load_code_cells(self, features_path):
        """
        Helper to load code cell IDs for order reconstruction.
        Reads the parquet file and extracts code cells for each notebook.

        Args:
            features_path (str): Path to the features parquet file.

        Returns:
            dict: {nb_id: [code_cell_id_1, code_cell_id_2, ...]}
        """
        print(f"Loading code cell map from {features_path}...")
        try:
            # Read only necessary columns to save memory
            df = pd.read_parquet(
                features_path, columns=["id", "cell_id", "cell_type", "rank"]
            )
        except Exception as e:
            print(f"Error reading parquet: {e}")
            return {}

        # Filter for code cells
        df_code = df[df["cell_type"] == "code"].copy()

        # For validation/train, sort by rank. For test, rank is -1 but file order is preserved.
        # Since FeatureExtractor processes sequentially, the dataframe index order is correct for test.
        # We explicitly sort by rank if available (rank != -1).
        # If rank is -1 (test), we rely on stable sort or existing order.

        # To be safe for Train/Val:
        if df_code["rank"].max() > -1:
            df_code = df_code.sort_values(["id", "rank"])

        # Group by ID and collect cell_ids
        code_map = df_code.groupby("id")["cell_id"].apply(list).to_dict()
        return code_map

    def _load_ground_truth(self, metadata_path):
        """
        Loads ground truth cell orders for validation.

        Args:
            metadata_path (str): Path to metadata CSV.

        Returns:
            dict: {nb_id: [correct_cell_order_list]}
        """
        df = pd.read_csv(metadata_path)
        gt_map = {}
        for _, row in df.iterrows():
            if "cell_order" in row and pd.notna(row["cell_order"]):
                gt_map[row["id"]] = row["cell_order"].split()
        return gt_map

    def train_epoch(self, dataloader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            # Move batch to device
            code_emb = batch["code_embeddings"].to(self.device)
            code_mask = batch["code_mask"].to(self.device)
            md_emb = batch["md_embeddings"].to(self.device)
            md_mask = batch["md_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # logits: (B, L_md, L_code + 1)
            logits = self.model(code_emb, code_mask, md_emb, md_mask)

            # Flatten for CrossEntropy
            # logits -> (B * L_md, L_code + 1)
            # labels -> (B * L_md)
            # Note: The number of classes (L_code + 1) varies per batch due to padding?
            # Actually, in a batch, L_code is fixed to max_len of that batch.
            # The logits shape is (B, Max_Md, Max_Code + 1).
            # Labels are indices into the code sequence.

            # Reshape
            B, L_md, n_classes = logits.shape
            logits_flat = logits.view(-1, n_classes)
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * B
            count += B

        avg_loss = total_loss / count if count > 0 else 0.0
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self, dataloader, code_map, gt_map):
        """
        Runs validation and computes Kendall Tau score.
        """
        self.model.eval()
        predictions = []
        ground_truths = []

        with torch.no_grad():
            for batch in dataloader:
                code_emb = batch["code_embeddings"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                md_emb = batch["md_embeddings"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)

                nb_ids = batch["id"]
                batch_md_ids = batch["md_ids"]  # list of lists

                # Forward pass
                logits = self.model(code_emb, code_mask, md_emb, md_mask)
                # logits: (B, L_md, L_code + 1)

                # Softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (B, L_md, L_code + 1)

                # Compute Expected Index for each MD cell
                # Indices are 0, 1, ..., L_code
                # Create index tensor: [0, 1, ..., L_code]
                max_code_len = probs.size(2)
                indices = torch.arange(max_code_len, device=self.device).float()

                # expected_pos = sum(p_i * i)
                expected_pos = torch.sum(probs * indices, dim=-1)  # (B, L_md)

                # Move to CPU for processing
                expected_pos = expected_pos.cpu().numpy()

                # Reconstruct orders for each notebook in batch
                for i, nb_id in enumerate(nb_ids):
                    if nb_id not in gt_map:
                        continue

                    # Get Code Cells (Anchors)
                    code_cells = code_map.get(nb_id, [])

                    # Get Markdown Cells (Queries)
                    # batch_md_ids[i] contains md IDs for this notebook
                    # expected_pos[i] contains scores. We need to slice valid length.
                    # The collator pads md_ids? No, md_ids is a list of lists, so valid length is len(batch_md_ids[i])
                    curr_md_ids = batch_md_ids[i]
                    curr_scores = expected_pos[i][: len(curr_md_ids)]

                    # Reconstruct
                    pred_order = self._reconstruct_order(
                        code_cells, curr_md_ids, curr_scores
                    )

                    predictions.append(pred_order)
                    ground_truths.append(gt_map[nb_id])

        score = compute_kendall_tau(predictions, ground_truths)
        print(f"Validation Kendall Tau: {score:.6f}")
        return score

    def _reconstruct_order(self, code_cells, md_cells, md_scores):
        """
        Merges code and markdown cells into a single ordered list.

        Strategy:
        - Code cell at index i (0-based) is assigned position i + 0.5.
        - Markdown cell is assigned its predicted Expected Index.
        - Sort all cells by position.
        """
        cells = []

        # Add Code Cells
        for i, cid in enumerate(code_cells):
            cells.append((cid, i + 0.5))

        # Add Markdown Cells
        for cid, score in zip(md_cells, md_scores):
            cells.append((cid, score))

        # Sort by score
        cells.sort(key=lambda x: x[1])

        return [c[0] for c in cells]

    def predict(self, dataloader, code_map):
        """
        Generates predictions for the test set.
        Returns a DataFrame suitable for submission.
        """
        self.model.eval()
        results = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in dataloader:
                code_emb = batch["code_embeddings"].to(self.device)
                code_mask = batch["code_mask"].to(self.device)
                md_emb = batch["md_embeddings"].to(self.device)
                md_mask = batch["md_mask"].to(self.device)

                nb_ids = batch["id"]
                batch_md_ids = batch["md_ids"]

                logits = self.model(code_emb, code_mask, md_emb, md_mask)
                probs = torch.softmax(logits, dim=-1)

                max_code_len = probs.size(2)
                indices = torch.arange(max_code_len, device=self.device).float()
                expected_pos = torch.sum(probs * indices, dim=-1).cpu().numpy()

                for i, nb_id in enumerate(nb_ids):
                    code_cells = code_map.get(nb_id, [])
                    curr_md_ids = batch_md_ids[i]
                    curr_scores = expected_pos[i][: len(curr_md_ids)]

                    pred_order = self._reconstruct_order(
                        code_cells, curr_md_ids, curr_scores
                    )
                    pred_string = " ".join(pred_order)

                    results.append({"id": nb_id, "cell_order": pred_string})

        return pd.DataFrame(results)

    def fit(self):
        """
        Main execution method.
        """
        # 1. Load DataLoaders
        print("Loading DataLoaders...")
        train_loader = get_dataloader(
            Config.TRAIN_FEATURES_PATH, mode="train", shuffle=True
        )
        val_loader = get_dataloader(Config.VAL_FEATURES_PATH, mode="val", shuffle=False)

        # 2. Load Metadata for Validation
        val_code_map = self._load_code_cells(Config.VAL_FEATURES_PATH)
        val_gt_map = self._load_ground_truth(Config.VAL_METADATA_PATH)

        best_score = -1.0

        # 3. Training Loop
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            self.train_epoch(train_loader, epoch)

            # Validation
            score = self.validate(val_loader, val_code_map, val_gt_map)

            # Checkpoint
            if score > best_score:
                print(
                    f"New best score: {score:.6f} (was {best_score:.6f}). Saving model."
                )
                best_score = score
                save_checkpoint(
                    self.model, self.optimizer, epoch, score, filename="best_model.pth"
                )

        print(f"Training complete. Best Validation Score: {best_score:.6f}")

        # 4. Prediction on Test Set
        print("Starting inference on Test Set...")
        # Load best model
        from library.utils import load_checkpoint

        load_checkpoint(self.model, filename="best_model.pth", device=self.device)

        test_loader = get_dataloader(
            Config.TEST_FEATURES_PATH, mode="test", shuffle=False
        )
        test_code_map = self._load_code_cells(Config.TEST_FEATURES_PATH)

        df_submission = self.predict(test_loader, test_code_map)

        # Save Submission
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Done.")


def run_training():
    trainer = Trainer()
    trainer.fit()
