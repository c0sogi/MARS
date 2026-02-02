import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from library.configuration import Config
from library.utilities import seed_everything, calculate_accuracy, HierarchyManager
from library.architecture import ConditionalCascadeMLP


def mixup_data(x, y_l1, y_l2, y_l3, alpha=0.2, device="cpu"):
    """
    Applies MixUp augmentation to the feature vectors and labels.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Return pairs of targets for all levels
    y_l1_a, y_l1_b = y_l1, y_l1[index]
    y_l2_a, y_l2_b = y_l2, y_l2[index]
    y_l3_a, y_l3_b = y_l3, y_l3[index]

    return mixed_x, y_l1_a, y_l1_b, y_l2_a, y_l2_b, y_l3_a, y_l3_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the MixUp loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Evaluator:
    """
    Handles the evaluation of the model on the validation set.
    """

    def __init__(self, model, device, criterion):
        self.model = model
        self.device = device
        self.criterion = criterion

    def evaluate(self, dataloader):
        """
        Computes loss and accuracy on the validation set.
        Only L3 accuracy is used for the final metric, but we compute hierarchical loss.
        """
        self.model.eval()
        total_loss = 0.0

        # Accuracy trackers
        all_preds_l3 = []
        all_targets_l3 = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack
                features, l1_targets, l2_targets, l3_targets = batch

                features = features.to(self.device)
                l1_targets = l1_targets.to(self.device)
                l2_targets = l2_targets.to(self.device)
                l3_targets = l3_targets.to(self.device)

                # Forward
                l1_logits, l2_logits, l3_logits = self.model(features)

                # Loss (Sum of Cross Entropy)
                loss_l1 = self.criterion(l1_logits, l1_targets)
                loss_l2 = self.criterion(l2_logits, l2_targets)
                loss_l3 = self.criterion(l3_logits, l3_targets)

                batch_loss = loss_l1 + loss_l2 + loss_l3
                total_loss += batch_loss.item() * features.size(0)

                # Store predictions for accuracy (L3 only)
                preds_l3 = torch.argmax(l3_logits, dim=1)
                all_preds_l3.append(preds_l3)
                all_targets_l3.append(l3_targets)

        # Aggregation
        total_samples = len(dataloader.dataset)
        avg_loss = total_loss / total_samples

        all_preds_l3 = torch.cat(all_preds_l3)
        all_targets_l3 = torch.cat(all_targets_l3)

        accuracy = calculate_accuracy(all_preds_l3, all_targets_l3)

        return avg_loss, accuracy


class Trainer:
    """
    Manages the training lifecycle of the ConditionalCascadeMLP.
    """

    def __init__(self, model_save_path):
        self.device = torch.device(Config.DEVICE)
        self.model_save_path = model_save_path

        # Initialize Model
        self.model = ConditionalCascadeMLP().to(self.device)

        # Optimizer & Loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Label Smoothing Cross Entropy
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Evaluator
        self.evaluator = Evaluator(self.model, self.device, self.criterion)

    def train_one_epoch(self, dataloader, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            features, l1_targets, l2_targets, l3_targets = batch

            features = features.to(self.device)
            l1_targets = l1_targets.to(self.device)
            l2_targets = l2_targets.to(self.device)
            l3_targets = l3_targets.to(self.device)

            # Apply MixUp
            mixed_x, y_l1_a, y_l1_b, y_l2_a, y_l2_b, y_l3_a, y_l3_b, lam = mixup_data(
                features,
                l1_targets,
                l2_targets,
                l3_targets,
                alpha=Config.MIXUP_ALPHA,
                device=self.device,
            )

            self.optimizer.zero_grad()

            # Forward pass
            l1_logits, l2_logits, l3_logits = self.model(mixed_x)

            # Compute Hierarchical Loss with MixUp
            loss_l1 = mixup_criterion(self.criterion, l1_logits, y_l1_a, y_l1_b, lam)
            loss_l2 = mixup_criterion(self.criterion, l2_logits, y_l2_a, y_l2_b, lam)
            loss_l3 = mixup_criterion(self.criterion, l3_logits, y_l3_a, y_l3_b, lam)

            total_loss = loss_l1 + loss_l2 + loss_l3

            # Backward
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        print(f"Starting training on device: {self.device}")
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.evaluator.evaluate(val_loader)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Acc: {val_acc:.8f} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"--> Best model saved with accuracy: {best_val_acc:.8f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs of no improvement."
                    )
                    break

        return best_val_acc


def generate_submission(model_paths, test_loader):
    """
    Generates predictions for the test set using an ensemble of trained models.
    Saves the result to submission.csv.
    """
    seed_everything()
    device = torch.device(Config.DEVICE)

    print(f"Generating submission using ensemble of {len(model_paths)} models...")

    # Load models
    models = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model path {path} does not exist. Skipping.")
            continue

        m = ConditionalCascadeMLP()
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)

    if not models:
        raise RuntimeError("No valid models found for inference.")

    all_ids = []
    all_probs = []

    # Inference Loop
    with torch.no_grad():
        for batch in test_loader:
            features, prod_ids = batch
            features = features.to(device)

            # Store IDs
            all_ids.extend(prod_ids.numpy())

            # Ensemble Prediction
            batch_probs = None

            for model in models:
                # Get logits
                _, _, l3_logits = model(features)
                # Softmax
                probs = F.softmax(l3_logits, dim=1)

                if batch_probs is None:
                    batch_probs = probs
                else:
                    batch_probs += probs

            # Average probabilities
            batch_probs /= len(models)

            # We only need the argmax for the submission, but we process in chunks to save memory
            # if needed. Here we just take argmax immediately to save RAM.
            preds = torch.argmax(batch_probs, dim=1).cpu().numpy()
            all_probs.extend(preds)

    # Decode predictions
    hierarchy_manager = HierarchyManager()
    decoded_preds = hierarchy_manager.decode_predictions(np.array(all_probs))

    # Create DataFrame
    submission_df = pd.DataFrame({"_id": all_ids, "category_id": decoded_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}. Rows: {len(submission_df)}")
