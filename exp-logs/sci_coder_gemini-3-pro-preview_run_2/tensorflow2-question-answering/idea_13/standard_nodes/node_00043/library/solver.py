import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config


class Solver:
    """
    Solver class to handle training, validation, and inference for the
    Window-Based Max-Pooling Network.
    """

    def __init__(self, model, config: Config, device=None):
        self.model = model
        self.config = config
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )

        # Loss functions
        self.criterion_window = nn.BCEWithLogitsLoss()
        self.criterion_span = nn.CrossEntropyLoss()
        self.criterion_yesno = nn.CrossEntropyLoss()

    def _calculate_loss(self, batch):
        """
        Calculates the composite loss: Window Score + Span + Yes/No.
        Handles masking for span and yes/no losses (only for positive windows).
        """
        input_ids = batch["input_ids"].to(self.device)
        question_ids = batch["question_ids"].to(self.device)

        label_window = batch["label_window"].to(self.device)  # Float (B,)
        label_start = batch["label_start"].to(self.device)  # Long (B,)
        label_end = batch["label_end"].to(self.device)  # Long (B,)
        label_yes_no = batch["label_yes_no"].to(self.device)  # Long (B,)

        # Forward pass
        window_score, start_logits, end_logits, yes_no_logits = self.model(
            input_ids, question_ids
        )

        # 1. Window Relevance Loss (Binary Classification)
        # window_score is (B, 1), label_window is (B,)
        loss_window = self.criterion_window(window_score.squeeze(-1), label_window)

        # 2. Span Loss (Only for positive windows)
        # Create mask for positive windows
        pos_mask = label_window == 1.0

        if pos_mask.sum() > 0:
            # Filter logits and labels
            loss_start = self.criterion_span(
                start_logits[pos_mask], label_start[pos_mask]
            )
            loss_end = self.criterion_span(end_logits[pos_mask], label_end[pos_mask])
            loss_span = loss_start + loss_end

            # 3. Yes/No Loss (Only for positive windows)
            loss_yesno = self.criterion_yesno(
                yes_no_logits[pos_mask], label_yes_no[pos_mask]
            )
        else:
            loss_span = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_yesno = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Weighted Sum
        total_loss = (
            self.config.LOSS_WEIGHT_WINDOW * loss_window
            + self.config.LOSS_WEIGHT_SPAN * loss_span
            + self.config.LOSS_WEIGHT_YESNO * loss_yesno
        )

        return total_loss, loss_window.item(), loss_span.item(), loss_yesno.item()

    def _train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            self.optimizer.zero_grad()
            loss, _, _, _ = self._calculate_loss(batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    def _validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                loss, _, _, _ = self._calculate_loss(batch)
                running_loss += loss.item()

        return running_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        print(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            print(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss}, Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_CHECKPOINT_PATH)
                print(f"  New best model saved to {self.config.MODEL_CHECKPOINT_PATH}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    def inference(self, test_loader):
        """
        Runs inference on the test set and generates the submission file.
        """
        print("Starting inference...")
        # Load best model
        if os.path.exists(self.config.MODEL_CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(self.config.MODEL_CHECKPOINT_PATH, map_location=self.device)
            )
            print(f"Loaded model from {self.config.MODEL_CHECKPOINT_PATH}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        results = []

        # 1. Collect predictions for all windows
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                question_ids = batch["question_ids"].to(self.device)

                # Forward pass
                window_score, start_logits, end_logits, yes_no_logits = self.model(
                    input_ids, question_ids
                )

                # Apply sigmoid to window score
                window_probs = torch.sigmoid(window_score).squeeze(-1).cpu().numpy()

                # Get span predictions (argmax)
                pred_starts = torch.argmax(start_logits, dim=1).cpu().numpy()
                pred_ends = torch.argmax(end_logits, dim=1).cpu().numpy()

                # Get Yes/No predictions
                pred_yes_no = torch.argmax(yes_no_logits, dim=1).cpu().numpy()

                # Collect metadata
                example_ids = batch["example_id"]  # List of strings
                candidate_indices = batch["candidate_index"].numpy()
                global_starts = batch["global_start"].numpy()

                for i in range(len(example_ids)):
                    results.append(
                        {
                            "example_id": example_ids[i],
                            "candidate_index": candidate_indices[i],
                            "window_score": window_probs[i],
                            "rel_start": pred_starts[i],
                            "rel_end": pred_ends[i],
                            "yes_no_class": pred_yes_no[i],
                            "global_w_start": global_starts[i],
                        }
                    )

        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("No predictions generated.")
            return

        # 2. Load Test Data to map candidate indices to long answer spans
        # We need to know the start/end tokens of the candidate to output the long answer format.
        # WindowProcessor doesn't pass this explicitly, so we look it up.
        print("Loading test data for candidate mapping...")
        candidates_map = {}  # (example_id, candidate_index) -> (start_token, end_token)

        try:
            with open(self.config.TEST_DATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    eid = str(entry["example_id"])
                    cands = entry.get("long_answer_candidates", [])
                    for idx, cand in enumerate(cands):
                        candidates_map[(eid, idx)] = (
                            cand["start_token"],
                            cand["end_token"],
                        )
        except FileNotFoundError:
            print(f"Error: Test data file not found at {self.config.TEST_DATA_PATH}")
            return

        # 3. Aggregate results per example
        final_preds = []

        # Group by example_id
        grouped = results_df.groupby("example_id")

        # We also need to ensure we output for every ID in the sample submission
        sample_sub = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)
        # Extract unique IDs from sample submission (format: ID_long, ID_short)
        required_ids = set()
        for row_id in sample_sub["example_id"]:
            if "_long" in row_id:
                required_ids.add(row_id.replace("_long", ""))
            elif "_short" in row_id:
                required_ids.add(row_id.replace("_short", ""))

        print(f"Processing {len(required_ids)} examples for submission...")

        for eid in required_ids:
            long_pred_str = ""
            short_pred_str = ""

            if eid in grouped.groups:
                group = grouped.get_group(eid)

                # Find the window with the maximum score
                best_row_idx = group["window_score"].idxmax()
                best_row = group.loc[best_row_idx]

                max_score = best_row["window_score"]

                # Threshold Check
                if max_score >= self.config.LONG_ANSWER_CONFIDENCE_THRESHOLD:
                    cand_idx = best_row["candidate_index"]

                    # --- Long Answer ---
                    if (eid, cand_idx) in candidates_map:
                        c_start, c_end = candidates_map[(eid, cand_idx)]
                        long_pred_str = f"{c_start}:{c_end}"

                    # --- Short Answer ---
                    # Calculate global token indices
                    # global_start is the start of the window in the doc
                    s_start_global = best_row["global_w_start"] + best_row["rel_start"]
                    # Convert inclusive prediction back to exclusive end index
                    s_end_global = best_row["global_w_start"] + best_row["rel_end"] + 1

                    # Basic validity check: end >= start
                    if s_end_global >= s_start_global:
                        # Check if Yes/No takes precedence?
                        # In NQ, usually Yes/No is mutually exclusive with span, or span is empty.
                        # Here we output YES/NO if class is 1 or 2, else span.
                        yn_class = best_row["yes_no_class"]
                        if yn_class == 1:
                            short_pred_str = "YES"
                        elif yn_class == 2:
                            short_pred_str = "NO"
                        else:
                            short_pred_str = f"{s_start_global}:{s_end_global}"

            # Append to list
            final_preds.append(
                {"example_id": f"{eid}_long", "PredictionString": long_pred_str}
            )
            final_preds.append(
                {"example_id": f"{eid}_short", "PredictionString": short_pred_str}
            )

        # 4. Save Submission
        submission_df = pd.DataFrame(final_preds)

        # Ensure order matches sample submission if possible, or just save
        # Kaggle usually matches by ID, so sorting isn't strictly required but good practice

        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(self.config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.FINAL_SUBMISSION_PATH}")
