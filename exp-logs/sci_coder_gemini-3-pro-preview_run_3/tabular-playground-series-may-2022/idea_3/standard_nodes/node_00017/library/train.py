import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    save_checkpoint,
    load_checkpoint,
    compute_auc,
)
from library.data import preprocess_features, ManufacturingDataset
from library.model import DenoisingAutoencoder, ManufacturingClassifier, Encoder


def train_classifier(train_data, val_data, metadata):
    """
    Direct Supervised Training (Cite solution_lesson_node_00013).
    Initializes the Encoder and Classifier from scratch.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # Datasets
    ds_train = ManufacturingDataset(
        train_data["cont"], train_data["cat"], train_data["target"], mode="supervised"
    )
    ds_val = ManufacturingDataset(
        val_data["cont"], val_data["cat"], val_data["target"], mode="supervised"
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model from scratch (No Pretraining)
    print("Initializing model for direct supervised training...")
    encoder = Encoder(metadata["cont_dim"], metadata["cat_cardinalities"])
    model = ManufacturingClassifier(encoder)
    model.to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.FINETUNE_MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.FINETUNE_MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.FINETUNE_EPOCHS,
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    print(
        f"Starting Stage 2: Fine-tuning Classifier for {Config.FINETUNE_EPOCHS} epochs..."
    )

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.FINETUNE_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0

        for (cont, cat), target in train_loader:
            cont, cat, target = cont.to(device), cat.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(cont, cat)
            loss = criterion(logits.squeeze(), target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for (cont, cat), target in val_loader:
                cont, cat = cont.to(device), cat.to(device)
                logits = model(cont, cat)
                probs = torch.sigmoid(logits).squeeze()

                val_preds.append(probs.cpu())
                val_targets.append(target.cpu())

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        val_auc = compute_auc(val_targets, val_preds)

        print(
            f"Epoch {epoch + 1}/{Config.FINETUNE_EPOCHS} | Train Loss: {avg_train_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, Config.BEST_MODEL_PATH
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    print(f"Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(test_data, metadata):
    """
    Generates predictions for the test set using the best fine-tuned model.
    """
    device = get_device()

    # Load Best Model
    # We need to reconstruct the architecture first
    encoder = Encoder(metadata["cont_dim"], metadata["cat_cardinalities"])
    model = ManufacturingClassifier(encoder)

    load_checkpoint(Config.BEST_MODEL_PATH, model, device=device)
    model.to(device)
    model.eval()

    ds_test = ManufacturingDataset(
        test_data["cont"], test_data["cat"], mode="supervised"
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for cont, cat in test_loader:
            cont, cat = cont.to(device), cat.to(device)
            logits = model(cont, cat)
            probs = torch.sigmoid(logits).squeeze()
            preds.extend(probs.cpu().numpy())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_data["id"], "target": preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Orchestrates the full training pipeline.
    """
    Config.setup()

    # 1. Data Processing
    print("--- Step 1: Data Processing ---")
    train_data, val_data, test_data, metadata = preprocess_features(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # 2. Pretraining (DAE)
    print("\n--- Step 2: Unsupervised Pretraining ---")
    encoder = pretrain_dae(train_data, val_data, test_data, metadata)

    # 3. Fine-tuning (Classifier)
    print("\n--- Step 3: Supervised Fine-tuning ---")
    train_classifier(train_data, val_data, metadata, encoder)

    # 4. Submission
    print("\n--- Step 4: Submission Generation ---")
    generate_submission(test_data, metadata)

    print("\nPipeline execution complete.")
