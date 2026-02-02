import os
import time
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from library import config, utils, data_loader, model, losses


class Trainer:
    """
    Manages training, validation, and inference for the ANG-KN model.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = model.ANG_KN().to(self.device)

        # Optimizer: Adam with Weight Decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Function: Cascaded Loss (Deep Supervision + Smoothing)
        self.criterion = losses.CascadedLoss().to(self.device)

        # Metric Tracking
        self.best_score = float("inf")  # Levenshtein Error Rate (Lower is better)

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            skeleton = batch["skeleton"].to(self.device)
            audio = batch["audio"].to(self.device)
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns dict of logits for deep supervision)
            outputs = self.model(skeleton, audio)

            # Compute cascaded loss
            loss, _ = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate_loss(self, loader):
        """
        Computes validation loss (frame-wise).
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                targets = batch["label"].to(self.device)

                outputs = self.model(skeleton, audio)
                loss, _ = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(loader)

    def validate_sequences(self, dataset):
        """
        Performs sliding-window inference on full validation sequences to compute
        the Levenshtein distance metric.
        """
        self.model.eval()

        # Map sequence index to list of window indices in the dataset
        seq_to_windows = defaultdict(list)
        for i, (seq_idx, start_frame) in enumerate(dataset.windows):
            seq_to_windows[seq_idx].append(i)

        total_distance = 0
        total_gestures = 0

        with torch.no_grad():
            # Iterate over all sequences in the dataset
            for seq_idx in range(len(dataset.sequences)):
                seq_data = dataset.sequences[seq_idx]
                full_len = seq_data["label"].shape[0]

                # Accumulators for stitching predictions
                probs_sum = torch.zeros(
                    (full_len, config.NUM_CLASSES), device=self.device
                )
                counts = torch.zeros((full_len, 1), device=self.device)

                window_indices = seq_to_windows[seq_idx]
                if not window_indices:
                    continue

                # Process windows in batches
                batch_size = config.BATCH_SIZE
                for i in range(0, len(window_indices), batch_size):
                    batch_idxs = window_indices[i : i + batch_size]

                    # Collect batch data
                    skeletons = []
                    audios = []
                    starts = []

                    for w_idx in batch_idxs:
                        # dataset[w_idx] handles preprocessing (derivatives, log-modulus)
                        item = dataset[w_idx]
                        skeletons.append(item["skeleton"])
                        audios.append(item["audio"])
                        starts.append(dataset.windows[w_idx][1])

                    skel_batch = torch.stack(skeletons).to(self.device)
                    audio_batch = torch.stack(audios).to(self.device)

                    # Inference
                    outputs = self.model(skel_batch, audio_batch)
                    # Use Stage 3 (final refinement) for prediction
                    logits = outputs["stage3"]
                    probs = torch.softmax(logits, dim=2)  # (Batch, Window, Classes)

                    # Stitch predictions
                    for b in range(len(batch_idxs)):
                        start = starts[b]
                        p = probs[b]

                        end = start + config.WINDOW_SIZE
                        valid_end = min(end, full_len)
                        valid_len = valid_end - start

                        if valid_len > 0:
                            probs_sum[start:valid_end] += p[:valid_len]
                            counts[start:valid_end] += 1.0

                # Average probabilities
                counts[counts == 0] = 1.0
                avg_probs = probs_sum / counts

                # Decode predictions (RLE -> Filter -> Remove Background)
                predicted_ids = utils.decode_predictions(avg_probs)

                # Get Ground Truth IDs
                gt_dense = seq_data["label"]
                gt_segments = utils.run_length_encoding(gt_dense)
                gt_ids = [
                    c for c, _, _ in gt_segments if c != config.BACKGROUND_CLASS_ID
                ]

                # Compute Levenshtein Distance
                dist = utils.levenshtein_distance(predicted_ids, gt_ids)
                total_distance += dist
                total_gestures += len(gt_ids)

        # Calculate Error Rate
        score = total_distance / total_gestures if total_gestures > 0 else 0.0
        return score

    def fit(self, train_loader, val_loader, val_dataset, epochs=config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device}...")

        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate_loss(val_loader)
            val_score = self.validate_sequences(val_dataset)

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_score:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Model Selection
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
                print(f"  New best model saved! Score: {self.best_score:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_score:.6f}")

    def predict(self, dataset):
        """
        Generates predictions for the test dataset.
        Returns: Dict {sample_id: [gesture_ids]}
        """
        # Load best model weights
        if os.path.exists(config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(config.BEST_MODEL_PATH, map_location=self.device)
            )
            self.model.eval()
        else:
            print("Warning: No best model found. Using current weights.")

        predictions = {}

        # Map sequence index to window indices
        seq_to_windows = defaultdict(list)
        for i, (seq_idx, start_frame) in enumerate(dataset.windows):
            seq_to_windows[seq_idx].append(i)

        with torch.no_grad():
            for seq_idx in range(len(dataset.sequences)):
                # Retrieve Sample ID from metadata
                sample_id = dataset.metadata.iloc[seq_idx]["sample_id"]

                seq_data = dataset.sequences[seq_idx]
                full_len = seq_data["label"].shape[0]

                probs_sum = torch.zeros(
                    (full_len, config.NUM_CLASSES), device=self.device
                )
                counts = torch.zeros((full_len, 1), device=self.device)

                window_indices = seq_to_windows[seq_idx]
                if not window_indices:
                    predictions[sample_id] = []
                    continue

                # Process windows in batches
                batch_size = config.BATCH_SIZE
                for i in range(0, len(window_indices), batch_size):
                    batch_idxs = window_indices[i : i + batch_size]

                    skeletons = []
                    audios = []
                    starts = []

                    for w_idx in batch_idxs:
                        item = dataset[w_idx]
                        skeletons.append(item["skeleton"])
                        audios.append(item["audio"])
                        starts.append(dataset.windows[w_idx][1])

                    skel_batch = torch.stack(skeletons).to(self.device)
                    audio_batch = torch.stack(audios).to(self.device)

                    outputs = self.model(skel_batch, audio_batch)
                    logits = outputs["stage3"]
                    probs = torch.softmax(logits, dim=2)

                    for b in range(len(batch_idxs)):
                        start = starts[b]
                        p = probs[b]

                        end = start + config.WINDOW_SIZE
                        valid_end = min(end, full_len)
                        valid_len = valid_end - start

                        if valid_len > 0:
                            probs_sum[start:valid_end] += p[:valid_len]
                            counts[start:valid_end] += 1.0

                counts[counts == 0] = 1.0
                avg_probs = probs_sum / counts

                # Decode
                predicted_ids = utils.decode_predictions(avg_probs)
                predictions[sample_id] = predicted_ids

        return predictions


def run_experiment():
    """
    Orchestrates the full experiment: Setup -> Train -> Predict -> Submit.
    """
    # Reproducibility
    utils.set_seed(config.SEED)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = data_loader.get_loaders()

    # Access underlying datasets for sequence reconstruction
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    # Initialize Trainer
    trainer = Trainer()

    # Train Model
    trainer.fit(train_loader, val_loader, val_dataset)

    # Generate Test Predictions
    print("Generating test predictions...")
    test_preds = trainer.predict(test_dataset)

    # Format and Save Submission
    lines = []
    for sample_id, pred_ids in test_preds.items():
        # Format: SessionID,Label1,Label2,...
        pred_str = ",".join(map(str, pred_ids))
        line = f"{sample_id},{pred_str}"
        lines.append(line)

    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    with open(config.SUBMISSION_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Submission saved to {config.SUBMISSION_PATH}")
