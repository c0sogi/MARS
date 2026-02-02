import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.utils import set_seed, get_score
from library.dataset import get_dataloaders
from library.model import LateFusionModel
from library.engine import train_one_epoch, validate


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Fast baseline settings: Override default epochs for speed
    EPOCHS = 5

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # 3. Model & Optimizer
    print("Initializing model...")
    model = LateFusionModel(pretrained=Config.PRETRAINED)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # 4. Training Loop
    best_score = 0.0
    best_model_path = Config.MODEL_PATH
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss, train_score = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Train AUC: {train_score:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_score}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best Model Saved! Score: {best_score}")

    # 5. Final Validation & Failure Analysis
    print("\nRunning Final Validation and Failure Analysis...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found. Using current weights.")

    model.eval()

    val_targets = []
    val_preds = []

    # For failure analysis
    img_means = []
    img_stds = []
    img_maxs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Forward
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Store predictions
            val_targets.extend(targets.cpu().numpy().flatten())
            val_preds.extend(probs.cpu().numpy().flatten())

            # Compute image stats for failure analysis
            # inputs shape: (B, 6, 1, 273, 256)
            # Flatten spatial dims for stats: (B, -1)
            B = inputs.size(0)
            flat_inputs = inputs.view(B, -1)

            img_means.extend(flat_inputs.mean(dim=1).cpu().numpy())
            img_stds.extend(flat_inputs.std(dim=1).cpu().numpy())
            img_maxs.extend(flat_inputs.max(dim=1).values.cpu().numpy())

    final_val_score = get_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_val_score}")

    # Failure Analysis
    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)
    errors = np.abs(val_targets - val_preds)

    df_analysis = pd.DataFrame(
        {"error": errors, "mean": img_means, "std": img_stds, "max": img_maxs}
    )

    print("\nFailure Analysis (Correlation with Error):")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 6. Conditional Submission
    THRESHOLD = 0.5236028383825309
    if final_val_score > THRESHOLD:
        print(
            f"\nValidation score {final_val_score} > {THRESHOLD}. Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                test_preds.extend(probs)

        # Create submission dataframe
        test_ids = test_loader.dataset.metadata["id"].values

        # Truncate if mismatch (safety check)
        min_len = min(len(test_ids), len(test_preds))
        submission = pd.DataFrame(
            {"id": test_ids[:min_len], "target": test_preds[:min_len]}
        )

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation score {final_val_score} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
