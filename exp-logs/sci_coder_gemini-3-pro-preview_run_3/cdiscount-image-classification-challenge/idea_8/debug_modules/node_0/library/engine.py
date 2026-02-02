import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.data_utils import HierarchyMapper


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Executes one epoch of training.

    Args:
        model: The HierarchicalMultiTaskNetwork instance.
        dataloader: DataLoader yielding (features, y1_soft, y2_soft, y3_soft).
        optimizer: The optimizer instance.
        device: Torch device (cpu or cuda).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Unpack batch from MixupCollate
        # features: (B, 2048)
        # y*_soft: (B, Num_Classes) - Soft targets
        features, y1_soft, y2_soft, y3_soft = batch

        features = features.to(device)
        y1_soft = y1_soft.to(device)
        y2_soft = y2_soft.to(device)
        y3_soft = y3_soft.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits_l1, logits_l2, logits_l3 = model(features)

        # Multi-Task Loss with Soft Targets
        loss_l1 = F.cross_entropy(logits_l1, y1_soft)
        loss_l2 = F.cross_entropy(logits_l2, y2_soft)
        loss_l3 = F.cross_entropy(logits_l3, y3_soft)

        # Weighted Sum
        loss = (
            (Config.WEIGHT_L3 * loss_l3)
            + (Config.WEIGHT_L2 * loss_l2)
            + (Config.WEIGHT_L1 * loss_l1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The HierarchicalMultiTaskNetwork instance.
        dataloader: DataLoader yielding (features, labels).
        device: Torch device.

    Returns:
        float: Categorization accuracy (Level 3).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Standard collate returns (features, labels)
            # labels: (B, 3) -> [l1, l2, l3]
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            # Target is Level 3 (Fine-grained)
            target_l3 = labels[:, 2]

            # Forward pass
            _, _, logits_l3 = model(features)

            # Hard predictions
            preds = torch.argmax(logits_l3, dim=1)

            correct += (preds == target_l3).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model: The HierarchicalMultiTaskNetwork instance.
        dataloader: DataLoader yielding features.
        device: Torch device.

    Returns:
        np.ndarray: Array of predicted class indices (Level 3).
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Test loader yields features only
            features = batch
            features = features.to(device)

            # Forward pass
            _, _, logits_l3 = model(features)

            # Get class indices
            preds = torch.argmax(logits_l3, dim=1)
            preds_list.append(preds.cpu().numpy())

    return np.concatenate(preds_list)


def generate_submission(model, dataloader, test_ids, device, output_path):
    """
    Runs inference and saves the submission file.

    Args:
        model: The trained model.
        dataloader: Test DataLoader.
        test_ids: Numpy array of test product IDs.
        device: Torch device.
        output_path: Path to save the CSV.
    """
    print("Generating predictions...")
    l3_indices = predict(model, dataloader, device)

    print("Mapping categories...")
    mapper = HierarchyMapper(load_cached_data=True)

    # Map internal L3 indices back to raw category_ids
    # Using list comprehension for efficiency
    category_ids = [mapper.get_category_id(idx) for idx in l3_indices]

    print(f"Saving submission to {output_path}...")
    submission_df = pd.DataFrame({"_id": test_ids, "category_id": category_ids})

    # Ensure correct types
    submission_df["_id"] = submission_df["_id"].astype(int)
    submission_df["category_id"] = submission_df["category_id"].astype(int)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved with {len(submission_df)} records.")
