import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything, calculate_auc
from library.dataset import get_datasets
from library.models import WhaleClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store for AUC calculation
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    val_auc = calculate_auc(all_targets, all_preds)

    return val_loss, val_auc


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds).flatten()


def train_model(model_name, train_dataset, val_dataset, device):
    """
    Orchestrates the training process for a specific model architecture.
    """
    print(f"\n=== Training Model: {model_name} ===")

    # 1. Prepare DataLoaders with WeightedRandomSampler for training
    # Calculate weights for balancing
    targets = train_dataset.labels
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[t] for t in targets])
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for BN stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = WhaleClassifier(model_name=model_name, pretrained=True)
    model = model.to(device)

    # 3. Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Best Val AUC for {model_name}: {best_auc}")
    return best_model_path


def run_training_pipeline():
    """
    Main driver function for Idea 7:
    1. Loads Data
    2. Trains Ensemble Members (EfficientNet-B2, ResNet-50)
    3. Performs Soft Voting
    4. Generates Submission
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Datasets
    # load_cached_data=True to use the cache generated by dataset.py
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    ensemble_preds = []

    # 2. Train and Predict for each model in the ensemble
    for arch in Config.MODEL_ARCHS:
        # Train
        best_model_path = train_model(arch, train_dataset, val_dataset, device)

        # Load Best Weights for Inference
        print(f"Loading best weights for {arch} from {best_model_path}...")
        model = WhaleClassifier(model_name=arch, pretrained=False)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model = model.to(device)

        # Inference
        print(f"Generating predictions for {arch}...")
        preds = inference(model, test_loader, device)
        ensemble_preds.append(preds)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # 3. Soft Voting (Averaging)
    print("Computing ensemble predictions (Soft Voting)...")
    avg_preds = np.mean(ensemble_preds, axis=0)

    # 4. Create Submission
    print("Creating submission file...")
    # Retrieve clip names from dataset
    clips = test_dataset.clips

    submission_df = pd.DataFrame({"clip": clips, "probability": avg_preds})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
