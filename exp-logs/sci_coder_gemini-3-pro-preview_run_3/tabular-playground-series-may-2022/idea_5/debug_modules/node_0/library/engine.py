import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import GlobalPreprocessor, ManufacturingDataset
from library.model import LNGatedFunnelNet


def train_fn(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for x_cont, x_cat, y in dataloader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(x_cont, x_cat)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set and returns the AUC score.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in dataloader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            outputs = model(x_cont, x_cat)
            probs = torch.sigmoid(outputs)

            preds.append(probs.cpu())
            targets.append(y)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    auc_score = compute_auc(targets, preds)
    return auc_score


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for x_cont, x_cat in dataloader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            outputs = model(x_cont, x_cat)
            probs = torch.sigmoid(outputs)

            preds.append(probs.cpu())

    return torch.cat(preds).numpy().flatten()


def run_engine(
    load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
):
    """
    Orchestrates the entire training, validation, and submission pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading & Processing
    preprocessor = GlobalPreprocessor()
    data, meta = preprocessor.process_data(load_cached_data=load_cached_data)

    # 3. Create Datasets
    train_dataset = ManufacturingDataset(
        data["train_cont"], data["train_cat"], data["train_y"]
    )
    val_dataset = ManufacturingDataset(data["val_cont"], data["val_cat"], data["val_y"])
    test_dataset = ManufacturingDataset(data["test_cont"], data["test_cat"])

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Initialization
    model = LNGatedFunnelNet(
        num_cont=meta["num_cont"],
        cat_cardinalities=meta["cat_cardinalities"],
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    # 6. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
    )
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    # 7. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = eval_fn(model, val_loader, device)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation AUC: {best_auc}")

    # 8. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_preds = inference_fn(model, test_loader, device)

    # 9. Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": data["test_ids"], "target": test_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
