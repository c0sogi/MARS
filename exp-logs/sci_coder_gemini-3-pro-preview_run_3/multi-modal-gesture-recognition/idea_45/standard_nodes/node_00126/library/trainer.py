import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from library import config, utils, data_loader, model


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LogSpaceSmoothingLoss(nn.Module):
    """
    Penalizes rapid changes in prediction probabilities between adjacent frames.
    Uses truncated MSE on log-probabilities.
    """

    def __init__(self, lambda_smooth, threshold):
        super(LogSpaceSmoothingLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits):
        # logits: (Batch, Time, Classes)
        # Convert to log-probabilities
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate diff between t and t-1
        # Slice: [:, 1:, :] and [:, :-1, :]
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        squared_diff = diff**2

        # Truncate (Clamp)
        # We clamp the squared difference to threshold^2
        truncated_diff = torch.clamp(squared_diff, max=self.threshold**2)

        # Mean over batch, time, classes
        loss = truncated_diff.mean()

        return self.lambda_smooth * loss


class CascadedLoss(nn.Module):
    """
    Combines Weighted Cross Entropy and Smoothing Loss for Deep Supervision.
    """

    def __init__(self, num_classes, bg_weight, smooth_lambda, smooth_threshold, device):
        super(CascadedLoss, self).__init__()

        # Weighted CE
        weights = torch.ones(num_classes, device=device)
        weights[config.BACKGROUND_CLASS_ID] = bg_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        # Smoothing
        self.smooth_loss = LogSpaceSmoothingLoss(smooth_lambda, smooth_threshold)

    def forward(self, stage_outputs, targets):
        """
        stage_outputs: List of logits [logits1, logits2, logits3]
        targets: (Batch, Time)
        """
        total_loss = 0.0

        # Flatten targets for CE: (Batch * Time)
        B, T = targets.shape
        targets_flat = targets.view(-1)

        for logits in stage_outputs:
            # Check shape consistency
            # logits: (Batch, Time, Classes) -> (Batch*Time, Classes) for CE
            logits_flat = logits.reshape(-1, logits.size(2))

            # CE Loss
            ce = self.ce_loss(logits_flat, targets_flat)

            # Smoothing Loss (computed on original temporal shape)
            smooth = self.smooth_loss(logits)

            total_loss += ce + smooth

        return total_loss


class Trainer:
    def __init__(self, device_name="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device_name)
        set_seed(config.SEED)

        # Initialize Model
        self.model = model.SHPAMCN().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss
        self.criterion = CascadedLoss(
            num_classes=config.NUM_CLASSES,
            bg_weight=config.BG_WEIGHT,
            smooth_lambda=config.SMOOTHING_LAMBDA,
            smooth_threshold=config.SMOOTHING_THRESHOLD,
            device=self.device,
        )

        # Best Metric Tracking
        self.best_score = float("inf")
        self.best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    def get_dataloaders(self, debug_sample_size=None):
        # Train Set (Windowed)
        train_dataset = data_loader.GestureDataset(
            split="train",
            load_cached_data=True,
            augment=True,
            debug_sample_size=debug_sample_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Val Set (Full Sequence)
        val_dataset = data_loader.GestureDataset(
            split="val",
            load_cached_data=True,
            augment=False,
            debug_sample_size=debug_sample_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Full sequence inference
            shuffle=False,
            num_workers=1,
        )

        # Test Set (Full Sequence)
        test_dataset = data_loader.GestureDataset(
            split="test",
            load_cached_data=True,
            augment=False,
            debug_sample_size=debug_sample_size,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        count = 0

        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward: returns list of logits
            outputs = self.model(features)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            count += features.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, loader):
        self.model.eval()

        # Load Ground Truth
        # We need to parse the metadata again or pass it in.
        # For simplicity, we parse metadata here to get the GT dict.
        val_metadata = utils.pd.read_csv(config.VAL_METADATA_PATH)
        gt_dict = utils.parse_ground_truth(val_metadata)

        predictions_dict = {}

        with torch.no_grad():
            for features, _, sample_id_tuple in loader:
                # Batch size is 1, sample_id_tuple is a tuple of size 1
                sample_id = sample_id_tuple[0]
                features = features.to(self.device)

                # Forward
                outputs = self.model(features)
                # Use the final stage output (logits3)
                final_logits = outputs[-1]  # (1, Time, Classes)

                # Decode
                probs = torch.softmax(final_logits, dim=2)
                preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

                # Convert to labels
                predicted_labels = utils.decode_predictions_to_labels(preds)
                predictions_dict[sample_id] = predicted_labels

        # Compute Metric
        score = utils.compute_metric(predictions_dict, gt_dict)
        return score

    def run_training(self, epochs=config.NUM_EPOCHS, debug_sample_size=None):
        print(f"Starting training on {self.device}...")
        train_loader, val_loader, _ = self.get_dataloaders(debug_sample_size)

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Score (Levenshtein): {val_score}"
            )

            # Checkpoint
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with score {self.best_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation Score: {self.best_score}")

    def predict_test(self, debug_sample_size=None):
        print("Starting prediction on test set...")

        # Load best model
        if not os.path.exists(self.best_model_path):
            print("No best model found. Skipping prediction.")
            return

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        _, _, test_loader = self.get_dataloaders(debug_sample_size)

        predictions_dict = {}

        with torch.no_grad():
            for features, _, sample_id_tuple in test_loader:
                sample_id = sample_id_tuple[0]
                features = features.to(self.device)

                outputs = self.model(features)
                final_logits = outputs[-1]

                probs = torch.softmax(final_logits, dim=2)
                preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

                predicted_labels = utils.decode_predictions_to_labels(preds)
                predictions_dict[sample_id] = predicted_labels

        # Save submission
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        utils.create_submission_file(predictions_dict, submission_path)
        print(f"Predictions generated for {len(predictions_dict)} samples.")
