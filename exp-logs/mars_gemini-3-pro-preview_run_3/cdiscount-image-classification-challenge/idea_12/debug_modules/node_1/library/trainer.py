import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import library.config as config
from library.dataset import CachedFeatureDataset
from library.model import HierarchicalMLP
from library.hierarchy_utils import HierarchyMapper


class MultiTaskLoss(nn.Module):
    """
    Computes the sum of CrossEntropyLoss for all three hierarchical levels.
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: Tuple (logits_l1, logits_l2, logits_l3)
            targets: Tuple (labels_l1, labels_l2, labels_l3)
        """
        l1_loss = self.ce(preds[0], targets[0])
        l2_loss = self.ce(preds[1], targets[1])
        l3_loss = self.ce(preds[2], targets[2])
        return l1_loss + l2_loss + l3_loss


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies MixUp to input features and returns mixed inputs and target pairs.
    y is a tuple of (l1, l2, l3) tensors.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # y is a tuple of tensors, we need to mix the tuple elements
    y_a = y
    y_b = tuple(t[index] for t in y)

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """
    Computes the MixUp loss using the MultiTaskLoss criterion.
    """
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, alpha):
    model.train()
    running_loss = 0.0

    for batch_idx, (features, targets) in enumerate(loader):
        features = features.to(device)
        # targets is a list of tensors [(B,), (B,), (B,)], convert to tuple of tensors on device
        targets = tuple(t.to(device) for t in targets)

        # Apply MixUp
        inputs, targets_a, targets_b, lam = mixup_data(features, targets, alpha, device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = tuple(t.to(device) for t in targets)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Calculate L3 Accuracy (Target Metric)
            # outputs[2] is logits_l3, targets[2] is labels_l3
            _, predicted = torch.max(outputs[2].data, 1)
            total += targets[2].size(0)
            correct_l3 += (predicted == targets[2]).sum().item()

    accuracy = correct_l3 / total
    avg_loss = running_loss / len(loader)

    return avg_loss, accuracy


def train_model(model_idx, train_loader, val_loader):
    """
    Trains a single instance of HierarchicalMLP.
    """
    print(f"\n=== Training Ensemble Model {model_idx + 1}/{config.ENSEMBLE_SIZE} ===")

    # Set seed for this model instance to ensure diversity if using different seeds,
    # or reproducibility if using specific seeds.
    # Here we shift seed by model_idx.
    torch.manual_seed(config.SEED + model_idx)
    np.random.seed(config.SEED + model_idx)

    model = HierarchicalMLP().to(config.DEVICE)
    criterion = MultiTaskLoss().to(config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    best_acc = 0.0
    patience_counter = 0
    save_path = config.MODEL_SAVE_PATH_TEMPLATE.format(model_idx)

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config.MIXUP_ALPHA
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        scheduler.step(val_acc)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Time: {elapsed:.1f}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Acc L3: {val_acc}"
        )

        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return save_path, best_acc


def train_ensemble():
    """
    Orchestrates the training of the full ensemble.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Check for features
    if not os.path.exists(config.TRAIN_FEATURES):
        raise FileNotFoundError(
            f"Training features not found at {config.TRAIN_FEATURES}. Run feature extraction first."
        )

    print("Loading datasets...")
    # Initialize Mapper once
    mapper = HierarchyMapper(load_cached_data=True)

    # Create Datasets
    train_dataset = CachedFeatureDataset(
        features_path=config.TRAIN_FEATURES,
        labels_path=config.TRAIN_LABELS_L3,
        ids_path=config.TRAIN_IDS,
        hierarchy_mapper=mapper,
        sample_size=config.DEBUG_SAMPLE_SIZE,
    )

    val_dataset = CachedFeatureDataset(
        features_path=config.VAL_FEATURES,
        labels_path=config.VAL_LABELS_L3,
        ids_path=config.VAL_IDS,
        hierarchy_mapper=mapper,
        sample_size=config.DEBUG_SAMPLE_SIZE,
    )

    # Create DataLoaders
    # Use persistent_workers=True if num_workers > 0 to avoid reloading mmap overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples."
    )

    model_paths = []
    scores = []

    for i in range(config.ENSEMBLE_SIZE):
        path, score = train_model(i, train_loader, val_loader)
        model_paths.append(path)
        scores.append(score)

    print("\n=== Ensemble Training Complete ===")
    print(f"Best Validation Accuracies: {scores}")
    print(f"Average Validation Accuracy: {np.mean(scores)}")

    return model_paths


def generate_submission(model_paths=None):
    """
    Generates predictions for the test set using the trained ensemble.
    """
    print("\n=== Generating Submission ===")

    if model_paths is None:
        # Infer paths from config template
        model_paths = [
            config.MODEL_SAVE_PATH_TEMPLATE.format(i)
            for i in range(config.ENSEMBLE_SIZE)
        ]

    # verify models exist
    valid_paths = [p for p in model_paths if os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError("No trained models found to generate submission.")

    # Load Mapper
    mapper = HierarchyMapper(load_cached_data=True)

    # Load Test Data
    test_dataset = CachedFeatureDataset(
        features_path=config.TEST_FEATURES,
        ids_path=config.TEST_IDS,
        hierarchy_mapper=mapper,
        sample_size=None,  # Always full test set
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_TRAIN,  # Can use large batch size for inference
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Models
    models = []
    for path in valid_paths:
        print(f"Loading model from {path}...")
        m = HierarchicalMLP().to(config.DEVICE)
        m.load_state_dict(torch.load(path, map_location=config.DEVICE))
        m.eval()
        models.append(m)

    all_preds = []
    all_ids = []

    print(f"Starting inference on {len(test_dataset)} test samples...")

    with torch.no_grad():
        for batch_idx, (features, ids) in enumerate(test_loader):
            features = features.to(config.DEVICE)

            # Ensemble Prediction
            batch_probs = None

            for model in models:
                # Output is (logits_l1, logits_l2, logits_l3)
                logits = model(features)[2]
                probs = torch.softmax(logits, dim=1)

                if batch_probs is None:
                    batch_probs = probs
                else:
                    batch_probs += probs

            # Average
            batch_probs /= len(models)

            # Get predictions
            _, preds = torch.max(batch_probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(ids.numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed test batch {batch_idx + 1}...")

    # Concatenate
    final_preds_idx = np.concatenate(all_preds)
    final_ids = np.concatenate(all_ids)

    # Map L3 indices back to category_id
    # mapper.l3_to_cat is a numpy array where index i corresponds to l3_idx i
    print("Mapping predictions to category IDs...")
    final_category_ids = mapper.l3_to_cat[final_preds_idx]

    # Create DataFrame
    submission_df = pd.DataFrame({"_id": final_ids, "category_id": final_category_ids})

    # Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(submission_df.head())
