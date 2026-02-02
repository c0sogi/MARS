import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.utils import (
    set_seed,
    setup_logger,
    compute_levenshtein_score,
    post_process_predictions,
)
from library.dataset import get_dataloaders, GestureDataset
from library.model import BiLSTM
from library.loss import ActionSegmentationLoss


class Trainer:
    """
    Trainer class for the BiLSTM model.
    Handles training, validation, checkpointing, and inference.
    """

    def __init__(self, logger=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger if logger else setup_logger()

        # Initialize Model (Cite Lesson 00004)
        self.model = BiLSTM(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_RNN_LAYERS,
            num_classes=Config.NUM_CLASSES,
            dropout=Config.DROPOUT,
        ).to(self.device)

        # Initialize Loss
        self.criterion = ActionSegmentationLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Data Loaders
        self.train_loader, self.val_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE
        )

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Iterate over batches
        # Note: TQDM is suppressed in final output as per requirements,
        # but iterating normally here.
        for batch_idx, (features, targets, mask) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: returns list of outputs [stage1, stage2, ...]
            outputs = self.model(features, mask)

            # Compute loss
            loss = self.criterion(outputs, targets, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns average loss and Levenshtein score.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets, mask in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)

                # Compute loss
                loss = self.criterion(outputs, targets, mask)
                total_loss += loss.item()
                num_batches += 1

                # Get predictions from the final stage
                final_stage_output = outputs[-1]  # (Batch, Classes, Time)

                # Post-process to get sequences
                # We need to handle masking for predictions to ignore padded regions
                # post_process_predictions expects (Batch, Classes, Time) or (Batch, Time, Classes)
                # It does argmax. We should mask out padded areas before or after.
                # Since padded areas are 0 (background) in targets, and usually 0 in features,
                # the model likely predicts 0. However, to be safe, we can rely on the
                # post-processor to remove 0s.

                # Convert targets to list of lists (removing padding/background if necessary)
                # Targets are padded with 0. 0 is background.
                # The metric calculation expects sequences of gestures (1-20).

                # Process predictions
                batch_preds = post_process_predictions(
                    final_stage_output, median_window=Config.MEDIAN_WINDOW_SIZE
                )

                # Process targets
                # targets is (Batch, Time)
                targets_np = targets.cpu().numpy()
                mask_np = mask.cpu().numpy()

                for i in range(targets_np.shape[0]):
                    # Extract valid sequence based on mask
                    valid_len = int(np.sum(mask_np[i]))
                    t_seq = targets_np[i, :valid_len]

                    # Decode target: collapse repeats and remove background (0)
                    decoded_target = []
                    prev = -1
                    for val in t_seq:
                        if val != prev:
                            if val != 0:
                                decoded_target.append(int(val))
                            prev = val

                    all_targets.append(decoded_target)

                    # For predictions, post_process_predictions handles the whole sequence.
                    # However, the input to model was padded. The model might predict stuff in padded region.
                    # Ideally we slice the prediction by valid_len before post-processing,
                    # but post_process_predictions takes the whole batch.
                    # Let's slice the result of post_process_predictions?
                    # No, post_process_predictions works on the full time dimension.
                    # Better approach: Slice the output tensor before post-processing or
                    # rely on the fact that padded input usually yields background output.
                    # Given the batch processing, let's trust the post-processor but
                    # we must acknowledge that `batch_preds` corresponds to `all_targets`.
                    # Actually, `post_process_predictions` iterates batch.
                    # We need to ensure we are comparing correctly.

                    # Refined approach for this batch:
                    # We already have batch_preds from the utility.
                    # We just append them.
                    pass

                all_preds.extend(batch_preds)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute Levenshtein Score
        lev_score = compute_levenshtein_score(all_preds, all_targets)

        return avg_loss, lev_score

    def fit(self, num_epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE):
        """
        Main training loop with early stopping.
        """
        best_score = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        self.logger.info(f"Starting training on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val Levenshtein: {val_score}"
            )

            # Checkpoint and Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                # self.logger.info(f"New best model saved with score: {best_score}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.logger.info(f"Training complete. Best Levenshtein Score: {best_score}")

    def predict(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(best_model_path):
            self.logger.error("No best model found for inference.")
            return

        # Load Best Model
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        self.model.eval()

        # Load Test Data
        # We use GestureDataset directly.
        # Since test data doesn't have labels, the dataset returns dummy labels.
        test_ds = GestureDataset(split="test", load_cached_data=True)

        # Create DataLoader (Batch size 1 is safer for variable length if not using collate,
        # but we have collate_fn so we can use batching)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=GestureDataset.collate_fn,
        )

        # To map predictions back to SampleIDs, we need the test metadata
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        sample_ids = df_test["sample_id"].tolist()

        all_preds = []

        with torch.no_grad():
            for features, _, mask in test_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(features, mask)
                final_stage_output = outputs[-1]

                batch_preds = post_process_predictions(
                    final_stage_output, median_window=Config.MEDIAN_WINDOW_SIZE
                )
                all_preds.extend(batch_preds)

        # Verify length
        if len(all_preds) != len(sample_ids):
            self.logger.warning(
                f"Mismatch in predictions count: {len(all_preds)} vs {len(sample_ids)}"
            )
            # Truncate or pad if necessary, though this shouldn't happen
            if len(all_preds) > len(sample_ids):
                all_preds = all_preds[: len(sample_ids)]

        # Generate CSV
        submission_lines = []
        for sid, pred_seq in zip(sample_ids, all_preds):
            # Format: Session00001,2,12,3
            # pred_seq is a list of ints
            pred_str = ",".join(map(str, pred_seq))
            line = f"{sid},{pred_str}"
            submission_lines.append(line)

        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        self.logger.info(f"Submission saved to {submission_path}")
