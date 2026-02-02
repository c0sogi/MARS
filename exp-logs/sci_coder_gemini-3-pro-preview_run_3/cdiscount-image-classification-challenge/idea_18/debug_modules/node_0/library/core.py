import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, HierarchyMapper
from library.dataset import create_dataloader
from library.model import ProjectedMultiTaskMLP


class CombinedLoss(nn.Module):
    """
    Computes the sum of Cross-Entropy losses for L1, L2, and L3 heads.
    Supports Label Smoothing and MixUp regularization.
    """

    def __init__(self, label_smoothing=0.0):
        super(CombinedLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds, targets_a, targets_b=None, lam=None):
        """
        Args:
            preds: Tuple of (logits_l1, logits_l2, logits_l3)
            targets_a: Tuple/List of (l1, l2, l3) ground truth labels
            targets_b: Tuple/List of (l1, l2, l3) shuffled labels for MixUp (optional)
            lam: MixUp interpolation coefficient (optional)
        """
        l1_pred, l2_pred, l3_pred = preds

        if targets_b is None or lam is None:
            # Standard Loss
            loss_l1 = self.criterion(l1_pred, targets_a[0])
            loss_l2 = self.criterion(l2_pred, targets_a[1])
            loss_l3 = self.criterion(l3_pred, targets_a[2])
        else:
            # MixUp Loss: lam * Loss(a) + (1 - lam) * Loss(b)
            l1_a, l2_a, l3_a = targets_a
            l1_b, l2_b, l3_b = targets_b

            loss_l1 = lam * self.criterion(l1_pred, l1_a) + (1 - lam) * self.criterion(
                l1_pred, l1_b
            )
            loss_l2 = lam * self.criterion(l2_pred, l2_a) + (1 - lam) * self.criterion(
                l2_pred, l2_b
            )
            loss_l3 = lam * self.criterion(l3_pred, l3_a) + (1 - lam) * self.criterion(
                l3_pred, l3_b
            )

        # Sum losses from all heads
        return loss_l1 + loss_l2 + loss_l3


def train_one_epoch(model, loader, optimizer, criterion, device, alpha):
    model.train()
    total_loss = 0.0
    num_samples = 0

    for features, targets in loader:
        features = features.to(device)
        # targets is a list of tensors: [l1, l2, l3]
        targets = [t.to(device) for t in targets]

        batch_size = features.size(0)

        # Feature-Space MixUp
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        if lam < 1.0:
            index = torch.randperm(batch_size).to(device)
            mixed_features = lam * features + (1 - lam) * features[index]
            targets_a = targets
            targets_b = [t[index] for t in targets]
        else:
            mixed_features = features
            targets_a = targets
            targets_b = None

        # Forward Pass
        optimizer.zero_grad()
        preds = model(mixed_features)

        # Compute Loss
        loss = criterion(preds, targets_a, targets_b, lam)

        # Backward Pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_l3 = 0
    total = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = [t.to(device) for t in targets]

            preds = model(features)
            loss = criterion(preds, targets)

            total_loss += loss.item() * features.size(0)

            # Calculate Accuracy for Target (L3)
            logits_l3 = preds[2]
            _, predicted_l3 = torch.max(logits_l3, 1)
            correct_l3 += (predicted_l3 == targets[2]).sum().item()
            total += features.size(0)

    avg_loss = total_loss / total
    acc_l3 = correct_l3 / total
    return avg_loss, acc_l3


def train_model(model_idx, train_loader, val_loader):
    """
    Trains a single model instance.
    """
    # Set seed for this model instance to ensure diversity in ensemble (via shuffling)
    seed_everything(Config.SEED + model_idx)
    device = torch.device(Config.DEVICE)

    print(f"\n=== Training Model {model_idx + 1}/{Config.NUM_MODELS} ===")

    # Initialize Model
    model = ProjectedMultiTaskMLP().to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = CombinedLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # Training Loop
    best_acc = 0.0
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, f"model_{model_idx}.pth")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            alpha=Config.MIXUP_ALPHA,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc L3: {val_acc}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Model {model_idx} finished. Best Validation Accuracy: {best_acc}")
    return best_acc


def train_ensemble():
    """
    Orchestrates the training of the model ensemble.
    Loads data once to save I/O and memory overhead.
    """
    print("Initializing Hierarchy Mapper...")
    mapper = HierarchyMapper(Config.CATEGORY_NAMES)
    mapper.process(load_cached=True)

    print("Loading Training and Validation Data into RAM...")
    # Load datasets once and pass to training function
    train_loader = create_dataloader(
        Config.TRAIN_FEATURES,
        Config.TRAIN_LABELS,
        mapper,
        mode="train",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = create_dataloader(
        Config.VAL_FEATURES,
        Config.VAL_LABELS,
        mapper,
        mode="val",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    # Train models sequentially
    for i in range(Config.NUM_MODELS):
        train_model(i, train_loader, val_loader)


def generate_submission():
    """
    Generates predictions for the test set using the trained ensemble.
    """
    print("\n=== Generating Submission ===")
    device = torch.device(Config.DEVICE)

    # Load Mapper
    mapper = HierarchyMapper(Config.CATEGORY_NAMES)
    mapper.process(load_cached=True)

    # Load Test Data
    print("Loading Test Data...")
    test_loader = create_dataloader(
        Config.TEST_FEATURES,
        Config.TEST_IDS,
        mapper,
        mode="test",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    # Load Models
    models = []
    for i in range(Config.NUM_MODELS):
        path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")
        if os.path.exists(path):
            print(f"Loading {path}...")
            model = ProjectedMultiTaskMLP().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model checkpoint {path} not found. Skipping.")

    if not models:
        print("Error: No models found for inference.")
        return

    # Inference Loop
    all_preds = []
    all_ids = []

    print("Running Inference...")
    with torch.no_grad():
        for features, ids in test_loader:
            features = features.to(device)

            # Ensemble Averaging
            avg_probs = None
            for model in models:
                # Get L3 logits
                _, _, logits_l3 = model(features)
                probs = torch.softmax(logits_l3, dim=1)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            # Average probabilities
            avg_probs /= len(models)
            preds_idx = torch.argmax(avg_probs, dim=1).cpu().numpy()

            # Map L3 index back to raw category_id
            raw_preds = [mapper.get_raw_category_id(idx) for idx in preds_idx]

            all_preds.extend(raw_preds)
            all_ids.extend(ids.numpy())

    # Save Submission
    df_sub = pd.DataFrame({"_id": all_ids, "category_id": all_preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
