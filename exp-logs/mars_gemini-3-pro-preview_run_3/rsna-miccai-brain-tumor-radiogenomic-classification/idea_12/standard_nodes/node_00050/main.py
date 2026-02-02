import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    MODEL_SAVE_PATH,
    VAL_METADATA_PATH,
    SEED,
)
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import SliceGroupedFusionNet
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()

    # 2. Data Loading
    # Load cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = SliceGroupedFusionNet()
    model.to(device)

    # 4. Optimization
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = -1.0

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Save Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 6. Final Evaluation
    # Load the best model weights for final assessment
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    final_loss, final_auc = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    val_probs = []
    val_targets = []

    # Collect predictions and targets
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            val_probs.extend(probs)
            val_targets.extend(targets.numpy().flatten())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_probs)

    # Load Validation Metadata to extract features for correlation
    val_df = pd.read_parquet(VAL_METADATA_PATH)

    # Ensure alignment (DataLoaders with shuffle=False preserve order)
    if len(val_df) != len(errors):
        print(
            "Warning: Validation dataframe length mismatch with predictions. Skipping detailed correlation."
        )
    else:
        features = {}
        # Extract slice counts per modality
        for col in ["flair_paths", "t1w_paths", "t1wce_paths", "t2w_paths"]:
            # Handle potential None values in paths
            counts = val_df[col].apply(lambda x: len(x) if x is not None else 0).values
            features[f"{col}_count"] = counts

        # Add target class
        features["target"] = val_targets

        # Compute correlations
        print("Correlation between Error Magnitude and Features:")
        for name, values in features.items():
            if len(values) == len(errors):
                # Compute correlation [0, 1] is the correlation coefficient
                corr = np.corrcoef(errors, values)[0, 1]
                print(f" - {name}: {corr}")

    # 8. Submission Generation
    THRESHOLD = 0.6978181818181817

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        # Ensure submission directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        output_path = os.path.join(submission_dir, "submission.csv")

        # Generate predictions for test set
        generate_submission(model, test_loader, device, output_path)
    else:
        print(
            f"\nValidation metric {final_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
