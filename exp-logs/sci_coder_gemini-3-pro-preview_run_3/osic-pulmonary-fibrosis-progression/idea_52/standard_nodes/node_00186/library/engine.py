import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.loss import LaplaceLogLikelihoodLoss, competition_metric
from library.utils import inverse_transform


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Unpack batch: image, restricted_inputs, context_inputs, target
        images, restricted_inputs, context_inputs, targets = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(images, restricted_inputs, context_inputs)

        # Compute loss
        loss = loss_fn(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss (loss.item() is mean of batch, so multiply by batch size)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples


def evaluate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_metric = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images, restricted_inputs, context_inputs, targets = [
                x.to(device) for x in batch
            ]

            # Forward pass
            mu, sigma = model(images, restricted_inputs, context_inputs)

            # Compute loss and metric
            loss = loss_fn(mu, sigma, targets)
            metric = competition_metric(mu, sigma, targets)

            # Accumulate
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_metric += metric * batch_size
            total_samples += batch_size

    return running_loss / total_samples, running_metric / total_samples


def fit(model, train_loader, val_loader, device, epochs=Config.EPOCHS, patience=10):
    """
    Manages the training lifecycle: optimization, scheduling, early stopping, and saving.
    """
    # Differential Learning Rates setup
    # Backbone parameters get LR_BACKBONE
    backbone_params = list(model.backbone.parameters())

    # All other parameters (Stream A, Stream B MLP, Head, Projector) get LR_HEAD
    # We explicitly collect them to ensure no parameter is missed
    head_params = (
        list(model.stream_a.parameters())
        + list(model.img_projector.parameters())
        + list(model.stream_b_mlp.parameters())
        + list(model.head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LaplaceLogLikelihoodLoss()

    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch}: Train Loss {train_loss}, Val Loss {val_loss}, Val Metric {val_metric}"
        )

        # Early Stopping Logic
        # The metric is negative (higher is better)
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model. New Best Metric: {best_metric}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")


def predict(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    mus = []
    sigmas = []

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            images, restricted_inputs, context_inputs, _ = [x.to(device) for x in batch]

            mu, sigma = model(images, restricted_inputs, context_inputs)

            mus.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())

    # Concatenate all batches
    mus = np.concatenate(mus)
    sigmas = np.concatenate(sigmas)

    # Inverse transform to original scale (ml)
    # This also applies the confidence clipping (max(sigma, 70))
    fvc_pred, conf_pred = inverse_transform(mus, sigmas)

    # Reconstruct the submission DataFrame
    # We rely on the DataLoader preserving the order of the dataset
    df = test_loader.dataset.df.copy()

    df["FVC"] = fvc_pred
    df["Confidence"] = conf_pred

    # Create the required unique ID
    df["Patient_Week"] = df["Patient"] + "_" + df["Weeks"].astype(str)

    # Select required columns
    submission_df = df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
