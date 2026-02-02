import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import (
    DEVICE,
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CACHE_DIR,
)
from library.utils import seed_everything, custom_weight_init
from library.dataset import get_dataloaders, get_test_ids
from library.network import SustainedHybridModel
from library.engine import Trainer, evaluate, predict


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Data Loading
    print("Loading data...")
    # Using load_cached_data=True as requested
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    test_ids = get_test_ids(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = SustainedHybridModel()
    # Apply specific initialization (PosEmbed std=0.02, GLU Xavier)
    custom_weight_init(model)
    model.to(DEVICE)

    # 4. Training Configuration
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # StepLR: decay by 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    # Path to save the best model during this run
    checkpoint_path = os.path.join(CACHE_DIR, "best_model_runfile.pth")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        scheduler=scheduler,
        patience=5,
        save_path=checkpoint_path,
    )

    # 5. Execution
    # Fast baseline: limit to 10 epochs to ensure completion within 2 hours
    # The A100 is fast, but we strictly follow the guideline to limit steps.
    FAST_EPOCHS = 10
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    trainer.fit(train_loader, val_loader, epochs=FAST_EPOCHS)

    # 6. Final Validation
    print("Performing final validation...")
    # Load the best model found during training
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    avg_loss, final_auc = evaluate(model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    all_targets = []
    all_preds = []
    all_cont_features = []

    # Collect data for analysis
    # We need to manually iterate to get features alongside predictions
    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            x_cat = x_cat.to(DEVICE)
            x_cont_gpu = x_cont.to(DEVICE)

            logits = model(x_cat, x_cont_gpu)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_targets.append(y.numpy())
            all_preds.append(probs)
            all_cont_features.append(x_cont.numpy())  # Keep on CPU for numpy ops

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds).flatten()
    all_cont_features = np.concatenate(all_cont_features, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate correlation between Error and each Continuous Feature
    # Continuous features are f_00 to f_30 (excluding f_27)
    # We assume the order matches the dataset generation (f_00..f_26, f_28..f_30)
    print("Correlation between Error Magnitude and Input Features:")
    correlations = []
    for i in range(all_cont_features.shape[1]):
        feat_vals = all_cont_features[:, i]
        # Handle potential constant features to avoid NaN
        if np.std(feat_vals) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 5 correlations
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")

    # 8. Submission
    THRESHOLD = 0.9970005855169476

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions
        test_preds = predict(model, test_loader, DEVICE)
        test_preds = test_preds.flatten()

        # Create dataframe
        submission_df = pd.DataFrame({"id": test_ids, "target": test_preds})

        # Save to ./submission/submission.csv
        output_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print(
            f"\nValidation metric ({final_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
