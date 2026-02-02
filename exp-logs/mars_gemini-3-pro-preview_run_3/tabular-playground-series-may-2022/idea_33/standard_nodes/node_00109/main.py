import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import seed_everything, get_device
from library.dataset import get_datasets
from library.model import CRHPEModel
from library.trainer import Trainer

# Configuration
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
SUBMISSION_THRESHOLD = 0.9975746465492954
BEST_MODEL_PATH = "./working/best_model.pth"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def perform_failure_analysis(trainer, val_loader, device):
    """
    Analyzes the correlation between model error and input features.
    """
    print("\n=== Failure Analysis ===")
    trainer.model.eval()

    all_cat = []
    all_cont = []
    all_targets = []
    all_preds = []

    # Collect all validation data and predictions
    with torch.no_grad():
        for cat_x, cont_x, y in val_loader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)

            outputs = trainer.model(cat_x, cont_x)

            # Ensemble averaging (Sigmoid -> Mean)
            probs = torch.zeros_like(outputs[0])
            for out in outputs:
                probs += torch.sigmoid(out)
            probs /= len(outputs)

            all_cat.append(cat_x.cpu().numpy())
            all_cont.append(cont_x.cpu().numpy())
            all_targets.append(y.numpy())
            all_preds.append(probs.cpu().numpy())

    # Concatenate
    cat_data = np.concatenate(all_cat, axis=0)
    cont_data = np.concatenate(all_cont, axis=0)
    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds).flatten()

    # Calculate Residuals (Error Magnitude)
    errors = np.abs(targets - preds)

    # Define Feature Names
    # Categorical: f_29, f_30, p_0...p_9
    cat_names = ["f_29", "f_30"] + [f"p_{i}" for i in range(10)]
    # Continuous: f_00...f_28 (excluding f_27) + unique_char_count
    cont_names = [f"f_{i:02d}" for i in range(29) if i != 27] + ["unique_char_count"]

    correlations = {}

    # Compute correlations for Categorical Features (using indices)
    for i, name in enumerate(cat_names):
        if i < cat_data.shape[1]:
            feat_vals = cat_data[:, i]
            # Simple correlation with label indices
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            correlations[name] = corr

    # Compute correlations for Continuous Features
    for i, name in enumerate(cont_names):
        if i < cont_data.shape[1]:
            feat_vals = cont_data[:, i]
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            correlations[name] = corr

    # Sort and Print Top Correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"Correlation between Error Magnitude and Input Features (Top 10):")
    for name, val in sorted_corr[:10]:
        print(f"{name:<20}: {val:.6f}")


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # We use debug=False to use the full dataset for maximum performance
    train_dataset, val_dataset, test_dataset, vocab_sizes, test_ids = get_datasets(
        load_cached_data=True,
        base_dir="./metadata",
        cache_dir="./working/idea_33",
        debug=False,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    num_cont = train_dataset.cont_features.shape[1]
    model = CRHPEModel(vocab_sizes, num_cont).to(device)

    # 4. Training Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3,
    )
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, device, optimizer, scheduler, criterion)

    # 5. Training Loop
    best_auc = 0.0
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        val_auc = trainer.validate(val_loader)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # Print progress (optional, but good for log tracking)
        # print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f}")

    print(f"Training complete. Best Validation AUC during training: {best_auc:.6f}")

    # 6. Final Evaluation & Metric
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Compute final metric on full validation set
    final_val_auc = trainer.validate(val_loader)
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(trainer, val_loader, device)

    # 8. Submission Generation
    if final_val_auc > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric {final_val_auc} exceeds threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        test_preds = trainer.predict(test_loader)

        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        sub_df = pd.DataFrame({"id": test_ids, "target": test_preds})
        sub_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_val_auc} did not meet threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
