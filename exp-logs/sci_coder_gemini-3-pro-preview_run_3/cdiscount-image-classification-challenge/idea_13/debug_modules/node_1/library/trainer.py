import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import FeatureDataset, mixup_data
from library.model import HierarchicalMLP, HierarchicalLoss
from library.utils import HierarchyMap


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, alpha):
    """
    Trains the model for one epoch using Feature-Space MixUp.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for features, l1, l2, l3 in loader:
        features = features.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        # Apply MixUp
        mixed_x, l1_a, l1_b, l2_a, l2_b, l3_a, l3_b, lam = mixup_data(
            features, l1, l2, l3, alpha=alpha, device=device
        )

        optimizer.zero_grad()
        outputs = model(mixed_x)

        # Calculate Multi-Task MixUp Loss
        loss = criterion(outputs, (l1_a, l1_b, l2_a, l2_b, l3_a, l3_b), lam=lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Level 3 (Target) Accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    with torch.no_grad():
        for features, l1, l2, l3 in loader:
            features = features.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            l3 = l3.to(device)

            outputs = model(features)

            # Standard Loss (No MixUp)
            loss = criterion(outputs, (l1, l2, l3), lam=None)
            running_loss += loss.item() * features.size(0)

            # Calculate Accuracy for Level 3 (Target)
            _, preds_l3 = torch.max(outputs[2], 1)
            correct_l3 += (preds_l3 == l3).sum().item()
            total_samples += features.size(0)

    return running_loss / total_samples, correct_l3 / total_samples


def train_ensemble_member(model, train_loader, val_loader, save_path, device, epochs):
    """
    Trains a single model instance with Early Stopping.
    """
    criterion = HierarchicalLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MIXUP_ALPHA
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Acc (L3): {val_acc}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! Acc: {best_acc}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for return
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_ensemble():
    """
    Orchestrates the training of the model ensemble.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Ensemble on {device}...")

    # Ensure directories exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Load Hierarchy Map
    hierarchy_map = HierarchyMap(load_cached_data=True)

    # Prepare Datasets
    print("Initializing Datasets...")
    train_dataset = FeatureDataset(
        feature_path=Config.TRAIN_FEATURES,
        label_path=Config.TRAIN_LABELS,
        hierarchy_map=hierarchy_map,
        mode="train",
    )

    val_dataset = FeatureDataset(
        feature_path=Config.VAL_FEATURES,
        label_path=Config.VAL_LABELS,
        hierarchy_map=hierarchy_map,
        mode="val",
    )

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Train Loop
    for i in range(Config.ENSEMBLE_SIZE):
        print(f"\n=== Training Ensemble Member {i+1}/{Config.ENSEMBLE_SIZE} ===")

        model = HierarchicalMLP().to(device)
        save_path = os.path.join(Config.MODEL_DIR, f"ensemble_model_{i}.pth")

        train_ensemble_member(
            model, train_loader, val_loader, save_path, device, Config.EPOCHS
        )

        # Cleanup
        del model
        torch.cuda.empty_cache()

    print("\nEnsemble training complete.")


def predict_ensemble():
    """
    Performs inference using the trained ensemble and generates submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Ensemble Inference on {device}...")

    # Load Hierarchy Map for decoding
    hierarchy_map = HierarchyMap(load_cached_data=True)

    # Prepare Test Dataset
    test_dataset = FeatureDataset(
        feature_path=Config.TEST_FEATURES, id_path=Config.TEST_IDS, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,  # Use same batch size as train for efficiency
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Models
    models = []
    for i in range(Config.ENSEMBLE_SIZE):
        model_path = os.path.join(Config.MODEL_DIR, f"ensemble_model_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Skipping.")
            continue

        model = HierarchicalMLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No trained models found for inference.")

    all_preds = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for features, product_ids in test_loader:
            features = features.to(device)

            # Aggregate probabilities
            ensemble_probs = torch.zeros(
                features.size(0), Config.NUM_CLASSES_L3, device=device
            )

            for model in models:
                _, _, l3_logits = model(features)
                probs = torch.softmax(l3_logits, dim=1)
                ensemble_probs += probs

            # Average
            ensemble_probs /= len(models)

            # Get predictions
            _, preds = torch.max(ensemble_probs, 1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(product_ids.numpy())

    # Concatenate results
    final_preds = np.concatenate(all_preds)
    final_ids = np.concatenate(all_ids)

    # Decode labels
    print("Decoding predictions...")
    final_category_ids = hierarchy_map.l3_encoder.inverse_transform(final_preds)

    # Create DataFrame
    submission_df = pd.DataFrame({"_id": final_ids, "category_id": final_category_ids})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
