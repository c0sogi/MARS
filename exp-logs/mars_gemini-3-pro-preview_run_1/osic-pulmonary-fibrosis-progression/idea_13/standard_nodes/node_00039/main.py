import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.nn import functional as F

# Import from provided libraries
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.dataset import get_dataloaders
from library.model import DualAxisNet
from library.trainer import LaplaceLoss, train_one_epoch, validate, predict


def analyze_failures(model, loader, device, feature_names=None):
    """
    Runs inference on the validation set and calculates correlations
    between absolute error and input features.
    """
    model.eval()
    all_true = []
    all_pred = []
    all_meta = []  # To store metadata features for correlation

    # We need to extract features corresponding to the batch
    # The loader returns 'meta' which is [Baseline_FVC, Weeks_From_Baseline]
    # We can also use 'tab' features.

    all_tab = []
    all_meta_tensor = []

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

            # Collect features for analysis
            all_meta_tensor.extend(meta.cpu().numpy())
            all_tab.extend(tab.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    # Calculate Absolute Error
    errors = np.abs(y_true - y_pred)

    # Prepare DataFrame for correlation
    meta_arr = np.array(all_meta_tensor)
    tab_arr = np.array(all_tab)

    # meta columns: Baseline_FVC, Weeks_From_Baseline
    # tab columns: derived from RobustScaler + OneHot.
    # Since we don't have exact column names for tab easily available without the encoder,
    # we will focus on the explicit meta features and the scalar Tabular features if possible.
    # However, 'tab' is transformed.
    # Let's rely on the 'meta' tensor which contains the raw-ish numericals used in the residual head.

    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Baseline_FVC": meta_arr[:, 0],
            "Weeks_From_Baseline": meta_arr[:, 1],
            "True_FVC": y_true,
            "Pred_FVC": y_pred,
        }
    )

    # Calculate correlations
    print("\nFailure Analysis (Correlation with Absolute Error):")
    correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
    print(correlations)

    return correlations


def main():
    # 1. Configuration
    EPOCHS = 15
    BATCH_SIZE = 32
    PATIENCE = 5
    LR = 1e-4
    SEED = 42
    METRIC_THRESHOLD = -6.510164260864258

    # Paths
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_13/"
    BEST_MODEL_PATH = "./working/best_model_idea_13.pth"
    SUBMISSION_PATH = "./submission/submission.csv"

    # Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_dir=METADATA_DIR,
        cache_dir=CACHE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,  # Reduced workers to be safe
        img_size=224,
    )

    # Determine tabular dimension
    sample_batch = next(iter(train_loader))
    tabular_dim = sample_batch["tab"].shape[1]
    print(f"Tabular Feature Dimension: {tabular_dim}")

    # 3. Model Initialization
    model = DualAxisNet(tabular_input_dim=tabular_dim, embed_dim=512, pretrained=True)
    model.to(device)

    criterion = LaplaceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.6f}"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping triggered after {PATIENCE} epochs without improvement."
            )
            break

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Compute Final Metric
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission
    if final_metric > METRIC_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({METRIC_THRESHOLD}). Generating submission..."
        )
        predict(model, test_loader, device, output_path=SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({METRIC_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
