import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data_loader import GlobalPreprocessor, ManufacturingDataset
from library.model import FunnelMLP
from library.engine import train_fn, eval_fn, inference_fn


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading & Processing
    # We use load_cached_data=True to attempt loading pre-processed arrays.
    preprocessor = GlobalPreprocessor()
    data, meta = preprocessor.process_data(load_cached_data=True)

    # 3. Data Setup
    # Cite solution_lesson_node_00022: Data Quantity Trumps Architectural Complexity
    # Using full dataset instead of subsampling.
    train_cont = data["train_cont"]
    train_cat = data["train_cat"]
    train_y = data["train_y"]

    val_cont = data["val_cont"]
    val_cat = data["val_cat"]
    val_y = data["val_y"]

    test_cont = data["test_cont"]
    test_cat = data["test_cat"]
    test_ids = data["test_ids"]

    # 4. Create Datasets and DataLoaders
    train_dataset = ManufacturingDataset(train_cont, train_cat, train_y)
    val_dataset = ManufacturingDataset(val_cont, val_cat, val_y)
    test_dataset = ManufacturingDataset(test_cont, test_cat)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model Initialization
    # Cite solution_lesson_node_00004: Simplicity Over Complexity
    model = FunnelMLP(
        num_cont=meta["num_cont"],
        cat_cardinalities=meta["cat_cardinalities"],
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    # 6. Training Configuration
    # Cite solution_lesson_node_00023: Precision Tuning of Weight Decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    # Cite solution_lesson_node_00002: Super-Convergence via One-Cycle Learning Rate Scheduling
    epochs = Config.EPOCHS  # 30 epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_baseline.pth")

    # 7. Training Loop
    for epoch in range(epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = eval_fn(model, val_loader, device)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # 8. Final Validation Metric
    print(f"Final Validation Metric: {best_auc}")

    # 9. Failure Analysis
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on validation set
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for x_cont, x_cat, y in val_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            outputs = model(x_cont, x_cat)
            probs = torch.sigmoid(outputs)
            val_preds_list.append(probs.cpu().numpy())
            val_targets_list.append(y.numpy())

    val_preds = np.concatenate(val_preds_list).flatten()
    val_targets = np.concatenate(val_targets_list)

    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation between continuous features and error
    print("Failure Analysis (Correlation with Error):")
    correlations = []
    num_cont_features = val_cont.shape[1]

    for i in range(num_cont_features):
        feature_values = val_cont[:, i]
        # Avoid division by zero if feature is constant
        if np.std(feature_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_values, errors)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for idx, corr in correlations[:5]:
        print(f"Feature index {idx}: {corr:.6f}")

    # 10. Submission
    threshold = 0.9971550270448856

    if best_auc > threshold:
        test_preds = inference_fn(model, test_loader, device)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission = pd.DataFrame({"id": test_ids, "target": test_preds})
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {best_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
