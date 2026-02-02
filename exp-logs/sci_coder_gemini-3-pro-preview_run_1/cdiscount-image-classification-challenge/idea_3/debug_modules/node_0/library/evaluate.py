import torch
import pandas as pd
import os
from library.config import Config
from library.utils import load_category_hierarchy


def validate(model, loader, criterion_dict, device):
    """
    Evaluates the model on the validation set.
    Computes loss and accuracy for all hierarchy levels.
    Prints metrics with full precision.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion_dict (dict): Dictionary of loss functions for 'l1', 'l2', 'l3'.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (avg_loss, acc_l1, acc_l2, acc_l3)
    """
    model.eval()

    running_loss = 0.0
    correct_l1 = 0
    correct_l2 = 0
    correct_l3 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Move data to device
            images = batch["images"].to(device, non_blocking=True)
            batch_index = batch["batch_index"].to(device, non_blocking=True)

            target_l1 = batch["l1_target"].to(device, non_blocking=True)
            target_l2 = batch["l2_target"].to(device, non_blocking=True)
            target_l3 = batch["l3_target"].to(device, non_blocking=True)

            # Forward pass
            outputs = model(images, batch_index)

            # Compute Losses
            loss_l1 = criterion_dict["l1"](outputs["logits_l1"], target_l1)
            loss_l2 = criterion_dict["l2"](outputs["logits_l2"], target_l2)
            loss_l3 = criterion_dict["l3"](outputs["logits_l3"], target_l3)

            # Weighted Sum
            total_loss = (
                loss_l3 * Config.LOSS_WEIGHT_L3
                + loss_l2 * Config.LOSS_WEIGHT_L2
                + loss_l1 * Config.LOSS_WEIGHT_L1
            )

            # Update metrics
            batch_size = target_l3.size(0)
            running_loss += total_loss.item() * batch_size
            total_samples += batch_size

            correct_l1 += (outputs["logits_l1"].argmax(1) == target_l1).sum().item()
            correct_l2 += (outputs["logits_l2"].argmax(1) == target_l2).sum().item()
            correct_l3 += (outputs["logits_l3"].argmax(1) == target_l3).sum().item()

    # Calculate averages
    avg_loss = running_loss / total_samples
    acc_l1 = correct_l1 / total_samples
    acc_l2 = correct_l2 / total_samples
    acc_l3 = correct_l3 / total_samples

    # Print full precision metrics
    print(f"Validation Results - Loss: {avg_loss}")
    print(f"L1 Accuracy: {acc_l1}")
    print(f"L2 Accuracy: {acc_l2}")
    print(f"L3 Accuracy: {acc_l3}")

    return avg_loss, acc_l1, acc_l2, acc_l3


def generate_predictions(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()

    # Create mapping from model output index (l3_idx) to original category_id
    df_hierarchy = load_category_hierarchy(load_cached_data=True)
    # Ensure l3_idx is integer for correct lookup
    df_hierarchy["l3_idx"] = df_hierarchy["l3_idx"].astype(int)

    # Create dictionary: l3_idx -> category_id
    # df_hierarchy index is category_id
    idx_to_cat = pd.Series(
        df_hierarchy.index.values, index=df_hierarchy["l3_idx"]
    ).to_dict()

    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            batch_index = batch["batch_index"].to(device)
            sample_ids = batch["sample_ids"].cpu().numpy()

            outputs = model(images, batch_index)

            # Get predictions for L3 (Fine-grained)
            preds_l3 = outputs["logits_l3"].argmax(dim=1).cpu().numpy()

            for sid, pred_idx in zip(sample_ids, preds_l3):
                # Map index back to category_id
                cat_id = idx_to_cat.get(pred_idx, -1)
                results.append({"_id": sid, "category_id": cat_id})

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Ensure correct types
    df_sub["_id"] = df_sub["_id"].astype(int)
    df_sub["category_id"] = df_sub["category_id"].astype(int)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
