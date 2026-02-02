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


def pretrain_dae(train_data, val_data, test_data, metadata):
    """
    Stage 1: Unsupervised Pretraining using a Denoising Autoencoder.
    Trains on Train + Val + Test data with Swap Noise Augmentation.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # Combine all datasets for unsupervised learning
    # mode='pretrain' enables swap noise augmentation in the Dataset class
    ds_train = ManufacturingDataset(
        train_data["cont"], train_data["cat"], mode="pretrain"
    )
    ds_val = ManufacturingDataset(val_data["cont"], val_data["cat"], mode="pretrain")
    ds_test = ManufacturingDataset(test_data["cont"], test_data["cat"], mode="pretrain")

    full_dataset = ConcatDataset([ds_train, ds_val, ds_test])

    dataloader = DataLoader(
        full_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = DenoisingAutoencoder(metadata["cont_dim"], metadata["cat_cardinalities"])
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.PRETRAIN_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss functions
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    print(f"Starting Stage 1: Pretraining DAE for {Config.PRETRAIN_EPOCHS} epochs...")

    model.train()
    for epoch in range(Config.PRETRAIN_EPOCHS):
        total_loss = 0.0

        for noisy_inputs, clean_targets in dataloader:
            noisy_cont, noisy_cat = noisy_inputs
            clean_cont, clean_cat = clean_targets

            noisy_cont = noisy_cont.to(device)
            noisy_cat = noisy_cat.to(device)
            clean_cont = clean_cont.to(device)
            clean_cat = clean_cat.to(device)

            optimizer.zero_grad()

            # Forward pass
            rec_cont, rec_cats = model(noisy_cont, noisy_cat)

            # Calculate Loss
            # 1. Continuous Reconstruction (MSE)
            loss_cont = criterion_mse(rec_cont, clean_cont)

            # 2. Categorical Reconstruction (CrossEntropy)
            loss_cat = 0.0
            for i, logits in enumerate(rec_cats):
                # clean_cat[:, i] contains the target indices for the i-th categorical feature
                loss_cat += criterion_ce(logits, clean_cat[:, i])

            loss = loss_cont + loss_cat

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{Config.PRETRAIN_EPOCHS} | DAE Loss: {avg_loss}")

    # Save the full DAE checkpoint
    save_checkpoint(
        model, optimizer, Config.PRETRAIN_EPOCHS, avg_loss, Config.PRETRAINED_MODEL_PATH
    )
    print("Pretraining complete.")

    # Return the trained encoder component
    return model.encoder


def train_classifier(train_data, val_data, metadata, encoder=None):
    """
    Stage 2: Supervised Fine-tuning.
    Uses the pretrained encoder and adds a classification head.
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

    # Initialize Model
    if encoder is None:
        print("Loading pretrained encoder from checkpoint...")
        tmp_dae = DenoisingAutoencoder(
            metadata["cont_dim"], metadata["cat_cardinalities"]
        )
        load_checkpoint(Config.PRETRAINED_MODEL_PATH, tmp_dae, device=device)
        encoder = tmp_dae.encoder

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
