import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.dataset import get_dataloaders
from library.model import DualAxisNet


class LaplaceLoss(nn.Module):
    """
    Differentiable Modified Laplace Log Likelihood Loss.
    Optimizes the negative of the competition metric.
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()

    def forward(self, fvc_true, fvc_pred, sigma):
        # Clip sigma at 70 ml
        sigma_clipped = torch.clamp(sigma, min=70)

        # Calculate absolute error
        delta = torch.abs(fvc_true - fvc_pred)

        # Clip delta at 1000 ml
        delta_clipped = torch.clamp(delta, max=1000)

        # Metric calculation
        # metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=fvc_true.device))
        metric = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
            sqrt_2 * sigma_clipped
        )

        # Return negative mean metric (to be minimized)
        return -torch.mean(metric)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tab = batch["tab"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, confidence = model(img_ax, img_cor, tab, meta)

        # Compute loss
        loss = criterion(target, fvc_pred, confidence)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_conf = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab = batch["tab"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            fvc_pred, confidence = model(img_ax, img_cor, tab, meta)

            all_true.extend(target.cpu().numpy())
            all_pred.extend(fvc_pred.cpu().numpy())
            all_conf.extend(confidence.cpu().numpy())

    # Calculate official metric
    score = laplace_log_likelihood_metric(
        np.array(all_true), np.array(all_pred), np.array(all_conf)
    )
    return score


def predict(model, loader, device, output_path="./submission/submission.csv"):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab = batch["tab"].to(device)
            meta = batch["meta"].to(device)
            patient_weeks = batch["patient_week"]

            fvc_pred, confidence = model(img_ax, img_cor, tab, meta)

            fvc_pred = fvc_pred.cpu().numpy()
            confidence = confidence.cpu().numpy()

            for i in range(len(patient_weeks)):
                # Ensure confidence is at least 70 for submission
                conf_val = max(confidence[i], 70)

                results.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": fvc_pred[i],
                        "Confidence": conf_val,
                    }
                )

    df_sub = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def fit(epochs=20, batch_size=32, patience=6, lr=1e-4, seed=42):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_dir="./metadata",
        cache_dir="./working/idea_13/",
        batch_size=batch_size,
        num_workers=4,
        img_size=224,
    )

    # Determine tabular input dimension
    sample_batch = next(iter(train_loader))
    tabular_dim = sample_batch["tab"].shape[1]
    print(f"Tabular Feature Dimension: {tabular_dim}")

    # 2. Initialize Model and Training Components
    model = DualAxisNet(tabular_input_dim=tabular_dim, embed_dim=512, pretrained=True)
    model.to(device)

    criterion = LaplaceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 3. Training Loop
    best_score = -float("inf")
    best_model_path = "./working/best_model_idea_13.pth"
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.10f}"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            # print("  -> New best model saved!")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Best Validation Score: {best_score:.10f}")

    # 4. Generate Submission
    print("Generating submission...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    predict(model, test_loader, device, output_path="./submission/submission.csv")
