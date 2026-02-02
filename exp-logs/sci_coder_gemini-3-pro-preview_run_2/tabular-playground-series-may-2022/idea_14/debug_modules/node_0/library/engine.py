import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def train_one_epoch(
    model, dataloader, optimizer, criterion, device=Config.DEVICE, max_batches=None
):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        # Unpack batch
        if len(batch) == 3:
            x_cat, x_cont, y = batch
        else:
            # Fallback if loader structure varies, though dataset.py implies 3 items for train
            x_cat, x_cont = batch
            y = None

        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        if y is not None:
            y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_cat, x_cont)

        if y is not None:
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        count += 1

    return running_loss / count if count > 0 else 0.0


def evaluate(model, dataloader, device=Config.DEVICE, max_batches=None):
    """
    Evaluates the model on a dataloader. Returns AUC and predictions.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            # Unpack batch
            if len(batch) == 3:
                x_cat, x_cont, y = batch
                targets.append(y.numpy())
            else:
                x_cat, x_cont = batch

            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    if len(preds) == 0:
        return 0.0, np.array([])

    preds = np.concatenate(preds)

    auc = 0.0
    if len(targets) > 0:
        targets = np.concatenate(targets)
        try:
            auc = roc_auc_score(targets, preds)
        except ValueError:
            auc = 0.0

    return auc, preds


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs=Config.EPOCHS,
    patience=5,
    device=Config.DEVICE,
    save_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    max_batches=None,
):
    """
    Full training loop with early stopping.
    """
    best_auc = 0.0
    epochs_no_improve = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches
        )
        val_auc, _ = evaluate(model, val_loader, device, max_batches)

        current_lr = optimizer.param_groups[0]["lr"]
        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val AUC: {val_auc} | LR: {current_lr}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epoch+1} epochs. Best AUC: {best_auc}"
            )
            break

        if scheduler is not None:
            scheduler.step()

    return best_auc


def predict_and_submit(
    model,
    test_loader,
    test_ids,
    device=Config.DEVICE,
    output_path=Config.OUTPUT_SUBMISSION,
):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                x_cat, x_cont = batch
            else:
                x_cat, x_cont, _ = batch

            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    if len(preds) > 0:
        preds = np.concatenate(preds).flatten()
    else:
        preds = np.zeros(len(test_ids))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "target": preds})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
