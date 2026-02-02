import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from library
from library.config import (
    SEED,
    DEVICE,
    NUM_WORKERS,
    IDEA_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    WINDOW_SIZE,
    NUM_CLASSES,
    BACKGROUND_CLASS_ID,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LOSS_STAGE_WEIGHTS,
    BG_CLASS_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    TRUNCATION_THRESHOLD,
    DEBUG_SUBSET_SIZE,
)
from library.utils import setup_logger, decode_predictions, generate_submission_file
from library.dataset import GestureDataset
from library.modules import LGKRN


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, logger):
        self.logger = logger
        self.device = DEVICE

        # Initialize Model
        self.model = LGKRN().to(self.device)

        # Optimizer: Adam is used for stable convergence of recurrent layers
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Loss Function Components
        # Class Weights: Down-weight background to focus on gestures
        class_weights = torch.ones(NUM_CLASSES).to(self.device)
        class_weights[BACKGROUND_CLASS_ID] = BG_CLASS_WEIGHT
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def compute_smoothing_loss(self, logits):
        """
        Computes Truncated MSE on Log-Probabilities of adjacent frames
        to enforce temporal smoothness.
        """
        # Log Softmax: (Batch, Time, Classes)
        log_probs = F.log_softmax(logits, dim=2)

        # Diff: P[t] - P[t-1]
        # Shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # MSE per frame-pair: (Batch, Time-1)
        mse = diff.pow(2).sum(dim=2)

        # Truncate to avoid penalizing sharp, valid transitions too heavily
        truncated_mse = torch.clamp(mse, max=TRUNCATION_THRESHOLD)

        # Mean over batch and time
        return truncated_mse.mean()

    def compute_loss(self, logits_list, targets):
        """
        Computes the total cascaded loss for Deep Supervision.
        logits_list: [logits_1, logits_2, logits_3]
        targets: (Batch, Time)
        """
        total_loss = 0.0

        # Flatten targets for CrossEntropy
        targets_flat = targets.view(-1)
        B, T, C = logits_list[0].shape

        for i, logits in enumerate(logits_list):
            stage_weight = LOSS_STAGE_WEIGHTS[i]

            # 1. Cross Entropy Loss
            ce_loss = self.ce_criterion(logits.reshape(-1, C), targets_flat)

            # 2. Smoothing Loss (Applied only to Refinement Stages 2 & 3)
            smooth_loss = 0.0
            if i > 0:
                smooth_loss = (
                    self.compute_smoothing_loss(logits) * SMOOTHING_LOSS_WEIGHT
                )

            stage_total = ce_loss + smooth_loss
            total_loss += stage_weight * stage_total

        return total_loss

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for inputs, targets, _ in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through all stages
            logits_1, logits_2, logits_3 = self.model(inputs)

            # Compute Cascaded Loss
            loss = self.compute_loss([logits_1, logits_2, logits_3], targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _ in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits_1, logits_2, logits_3 = self.model(inputs)

                loss = self.compute_loss([logits_1, logits_2, logits_3], targets)
                running_loss += loss.item()

                # Compute accuracy on the final stage output
                preds = torch.argmax(logits_3, dim=2)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        avg_loss = running_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def predict(self, loader, dataset):
        """
        Runs inference and aggregates sliding window predictions into full sequences.
        Assumes loader is not shuffled.
        """
        self.model.eval()

        # Data structures for aggregation
        # Map sample_id (str) -> accumulated probabilities (SeqLen, NumClasses)
        sample_probs = {}
        sample_counts = {}

        # Initialize buffers based on dataset metadata
        for i, sid in enumerate(dataset.sample_ids):
            seq_len = dataset.lengths[i]
            sample_probs[sid] = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
            sample_counts[sid] = np.zeros((seq_len,), dtype=np.float32)

        # Iterate through sliding windows
        current_window_idx = 0

        with torch.no_grad():
            for inputs, _, _ in loader:
                inputs = inputs.to(self.device)

                # Forward (Use Stage 3 for final prediction)
                _, _, logits_3 = self.model(inputs)
                probs = F.softmax(logits_3, dim=2).cpu().numpy()

                batch_size = inputs.size(0)

                for b in range(batch_size):
                    if current_window_idx >= len(dataset.windows):
                        break

                    # Retrieve window mapping info
                    s_idx, start, end = dataset.windows[current_window_idx]
                    sid = dataset.sample_ids[s_idx]

                    # Determine valid length (exclude padding if window extends beyond sequence)
                    # Note: Dataset pads input if seq_len < WINDOW_SIZE, but 'end' in windows tuple
                    # corresponds to the actual sequence end.
                    valid_len = end - start

                    # Extract valid predictions
                    window_preds = probs[b, :valid_len, :]

                    # Accumulate
                    sample_probs[sid][start:end] += window_preds
                    sample_counts[sid][start:end] += 1.0

                    current_window_idx += 1

        # Normalize and Decode
        final_predictions = []
        final_ids = []

        for i, sid in enumerate(dataset.sample_ids):
            # Avoid division by zero
            counts = sample_counts[sid][:, None]
            counts[counts == 0] = 1.0

            avg_probs = sample_probs[sid] / counts

            # Decode using RLE and Background filtering
            pred_seq = decode_predictions(avg_probs)
            final_predictions.append(pred_seq)
            final_ids.append(sid)

        return final_predictions, final_ids


def main():
    set_seed(SEED)
    logger = setup_logger(os.path.join(IDEA_DIR, "training.log"))
    logger.info("Initializing LG-KRN Training Pipeline")

    # 1. Load Data
    logger.info("Loading Datasets...")
    # Train with augmentation
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH,
        "train",
        load_cache=True,
        augment=True,
        debug_size=DEBUG_SUBSET_SIZE,
    )
    # Val without augmentation
    val_dataset = GestureDataset(
        VAL_METADATA_PATH,
        "val",
        load_cache=True,
        augment=False,
        debug_size=DEBUG_SUBSET_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(f"Train Windows: {len(train_dataset)}, Val Windows: {len(val_dataset)}")

    # 2. Setup Trainer
    trainer = Trainer(logger)

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(trainer.model.state_dict(), MODEL_SAVE_PATH)
            logger.info("  New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # 4. Inference on Test Set
    logger.info("Starting Inference on Test Set...")

    # Load Best Model
    trainer.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    test_dataset = GestureDataset(
        TEST_METADATA_PATH,
        "test",
        load_cache=True,
        augment=False,
        debug_size=DEBUG_SUBSET_SIZE,
    )
    # Important: shuffle=False to maintain order for window reconstruction
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    predictions, sample_ids = trainer.predict(test_loader, test_dataset)

    # 5. Generate Submission
    logger.info(f"Generating submission file at {SUBMISSION_PATH}...")
    generate_submission_file(predictions, sample_ids, SUBMISSION_PATH)
    logger.info("Done.")


if __name__ == "__main__":
    main()
