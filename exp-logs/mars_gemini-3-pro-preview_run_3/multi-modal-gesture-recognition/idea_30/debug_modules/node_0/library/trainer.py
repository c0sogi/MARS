import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from library import config, model, data_loader, utils


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the HSGKN model.
    """

    def __init__(self):
        # 1. Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # 2. Data Loaders
        # We assume data is already processed/cached or will be processed here
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.test_sample_ids,
            self.test_indices,
        ) = data_loader.get_dataloaders(load_cached_data=True)

        # 3. Model
        self.model = model.HSGKN().to(self.device)

        # 4. Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # 5. Loss Functions
        # Weighted Cross Entropy: Downweight background class (0)
        class_weights = torch.ones(config.NUM_CLASSES).to(self.device)
        class_weights[0] = config.BG_CLASS_WEIGHT
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Smoothing Loss
        self.smooth_loss = utils.TruncatedMSELoss(threshold=config.SMOOTHING_THRESHOLD)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        Returns: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            # Outputs is a dict: {'stage1': ..., 'stage2': ..., 'stage3': ...}
            outputs = self.model(features)

            loss = 0.0
            # Cascaded Loss Calculation
            for stage_name, logits in outputs.items():
                # logits shape: (Batch, Classes, Time)
                # targets shape: (Batch, Time)

                # 1. Classification Loss
                l_ce = self.ce_loss(logits, targets)
                loss += l_ce

                # 2. Smoothing Loss (Only for refinement stages)
                if stage_name in ["stage2", "stage3"]:
                    log_probs = F.log_softmax(logits, dim=1)
                    l_smooth = self.smooth_loss(log_probs)
                    loss += config.SMOOTHING_LOSS_WEIGHT * l_smooth

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, loader, is_test=False):
        """
        Evaluates the model by reconstructing full sequences from sliding windows.

        Args:
            loader: DataLoader (val or test)
            is_test: Boolean, if True returns predictions, else returns metric score.

        Returns:
            Validation Score (Levenshtein Error Rate) OR List of Predictions
        """
        self.model.eval()

        dataset = loader.dataset
        # Access raw skeletons to determine sequence lengths
        # dataset.skeletons is a list of arrays (T, Joints, 3)
        seq_lengths = [s.shape[0] for s in dataset.skeletons]
        num_seqs = len(dataset.skeletons)

        # Initialize buffers for probability accumulation
        # List of tensors: [(T, Classes), ...]
        seq_probs = [
            torch.zeros(l, config.NUM_CLASSES).to(self.device) for l in seq_lengths
        ]
        seq_counts = [torch.zeros(l).to(self.device) for l in seq_lengths]

        with torch.no_grad():
            # We iterate linearly. Since shuffle=False, we can map batch items
            # to dataset indices sequentially.
            global_idx = 0

            for features, _ in loader:
                batch_size = features.size(0)
                features = features.to(self.device)

                # Forward pass
                outputs = self.model(features)
                # Use Stage 3 (Final Refinement) for prediction
                logits = outputs["stage3"]  # (Batch, Classes, Time)
                probs = F.softmax(logits, dim=1)

                # Permute to (Batch, Time, Classes) for easier slicing
                probs = probs.permute(0, 2, 1)

                for b in range(batch_size):
                    # Identify which sequence and start frame this window belongs to
                    seq_idx, start_frame = dataset.indices[global_idx]

                    # Determine the valid range within the sequence
                    seq_len = seq_lengths[seq_idx]
                    end_frame = start_frame + config.WINDOW_SIZE

                    # Clip to sequence length (ignore padding at the end of sequence if any)
                    valid_end = min(end_frame, seq_len)

                    # Length of valid data in this window
                    # Note: The dataset pads the *input* window if it goes out of bounds.
                    # We only want to accumulate the valid part corresponding to the real sequence.
                    window_len_valid = valid_end - start_frame

                    # Accumulate
                    seq_probs[seq_idx][start_frame:valid_end] += probs[b][
                        :window_len_valid
                    ]
                    seq_counts[seq_idx][start_frame:valid_end] += 1.0

                    global_idx += 1

        # Post-process and Calculate Metrics
        predictions = []
        total_lev_dist = 0
        total_ref_gestures = 0

        for i in range(num_seqs):
            # Average probabilities
            counts = seq_counts[i].unsqueeze(1)
            # Avoid div by zero (should not happen given coverage logic)
            counts[counts == 0] = 1.0
            avg_probs = seq_probs[i] / counts

            # Convert to numpy for utils
            avg_probs_np = avg_probs.cpu().numpy()

            # Decode using utils pipeline (Argmax -> RLE -> Filter -> Remove BG)
            pred_seq = utils.process_predictions(
                avg_probs_np, min_length=config.MIN_GESTURE_LENGTH
            )
            predictions.append(pred_seq)

            if not is_test:
                # Get Ground Truth
                # dataset.labels[i] is a frame-wise array
                gt_frame_labels = dataset.labels[i]

                # Extract gesture sequence from frame labels
                # We use min_length=1 for GT to capture all annotated gestures
                gt_seq = utils.process_predictions(gt_frame_labels, min_length=1)

                # Calculate Levenshtein Distance
                dist = utils.levenshtein_distance(pred_seq, gt_seq)
                total_lev_dist += dist
                total_ref_gestures += len(gt_seq)

        if is_test:
            return predictions
        else:
            # Metric: Total Distance / Total Gestures
            score = (
                total_lev_dist / total_ref_gestures
                if total_ref_gestures > 0
                else float("inf")
            )
            return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training for {config.NUM_EPOCHS} epochs...")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.evaluate(self.val_loader, is_test=False)

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Score: {val_score:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
                print(f"  -> New Best Model Saved! Score: {best_score:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.
        """
        # Load Best Model
        if os.path.exists(config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(config.BEST_MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model for inference.")
        else:
            print("Warning: Best model not found. Using current model weights.")

        print("Generating predictions on Test Set...")
        predictions = self.evaluate(self.test_loader, is_test=True)

        # Save to Submission File
        sample_ids = self.test_loader.dataset.sample_ids

        with open(config.SUBMISSION_PATH, "w") as f:
            for sid, pred_seq in zip(sample_ids, predictions):
                # Format: SessionID,label1,label2,...
                pred_str = ",".join(map(str, pred_seq))
                f.write(f"{sid},{pred_str}\n")

        print(f"Submission saved to {config.SUBMISSION_PATH}")


def run_training():
    """
    Entry point function to run the training pipeline.
    """
    # Ensure reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    trainer = Trainer()
    trainer.fit()
    trainer.predict()
