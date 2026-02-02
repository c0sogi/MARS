import os
import numpy as np
import torch
import torch.optim as optim
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import ModalityAwareDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()

    # Hyperparameters
    EPOCHS = 15
    BATCH_SIZE = 16
    LR = 1e-4
    PATIENCE = 5

    # Paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_PATH = "./submission/submission.csv"
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model_runfile.pth")

    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    # Using the full dataset (approx 1100 samples) fits within the "fast baseline" constraint.
    loaders = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, metadata_dir=METADATA_DIR
    )

    # 3. Model Initialization
    model = ModalityAwareDualAxisNet(tabular_input_dim=7, embedding_dim=1280)
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Train one epoch
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )

        # Evaluate on validation set
        val_metric = evaluate(model, loaders["val"], criterion, device)

        # Update scheduler
        scheduler.step()

        # Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= PATIENCE:
            break

    # 6. Final Evaluation & Failure Analysis
    # Load best model
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Collect predictions on Validation set
    val_loader = loaders["val"]
    all_preds_fvc = []
    all_preds_sigma = []
    all_targets = []
    all_tabular = []

    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device, dtype=torch.float32)
            img_coronal = batch["img_coronal"].to(device, dtype=torch.float32)
            tabular = batch["tabular"].to(device, dtype=torch.float32)
            meta = batch["meta"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.float32)

            pred_fvc, pred_sigma = model(img_axial, img_coronal, tabular, meta)

            all_preds_fvc.append(pred_fvc.cpu().numpy())
            all_preds_sigma.append(pred_sigma.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_tabular.append(tabular.cpu().numpy())

    # Concatenate results
    all_preds_fvc = np.concatenate(all_preds_fvc)
    all_preds_sigma = np.concatenate(all_preds_sigma)
    all_targets = np.concatenate(all_targets).flatten()
    all_tabular = np.concatenate(all_tabular)

    # Calculate Final Metric (Modified Laplace Log Likelihood)
    sigma_clipped = np.maximum(all_preds_sigma, 70)
    delta = np.minimum(np.abs(all_targets - all_preds_fvc), 1000)
    metric_values = -(np.sqrt(2) * delta / sigma_clipped) - np.log(
        np.sqrt(2) * sigma_clipped
    )
    final_metric = np.mean(metric_values)

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between absolute error and features
    errors = np.abs(all_targets - all_preds_fvc)
    feature_names = [
        "Age",
        "Sex_Male",
        "Sex_Female",
        "Smoke_Ex",
        "Smoke_Never",
        "Smoke_Current",
        "Percent",
    ]

    print("Failure Analysis (Correlation with Error Magnitude):")
    for i, name in enumerate(feature_names):
        feat_vals = all_tabular[:, i]
        # Avoid correlation calculation if feature is constant
        if np.std(feat_vals) > 1e-9:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
            print(f"{name}: {corr}")
        else:
            print(f"{name}: 0.0")

    # 7. Submission
    THRESHOLD = -6.510164260864258
    if final_metric > THRESHOLD:
        generate_submission(model, loaders["test"], device, output_path=SUBMISSION_PATH)


if __name__ == "__main__":
    main()
