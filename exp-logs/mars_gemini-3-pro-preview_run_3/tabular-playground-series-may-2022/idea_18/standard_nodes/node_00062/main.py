import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import components from the provided library
from library.config import set_seed, BATCH_SIZE, TabularDataset
from library.model import train_model, generate_submission


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Train Model (Fast Baseline)
    # Limiting to 200,000 samples and 10 epochs to ensure fast execution (< 2 hours)
    # while maintaining sufficient data density for the PIFE architecture.
    print("\n=== Starting Model Training ===")
    model, device, data = train_model(
        epochs=10, max_samples=200000, load_cached_data=True
    )

    # 3. Full Validation Inference
    print("\n=== Performing Full Validation ===")
    # Create dataset and loader for the full validation set
    val_dataset = TabularDataset(data["X_val_cat"], data["X_val_cont"], data["y_val"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)

            # Ensemble prediction: Average probability across the 5 independent streams
            probs = torch.sigmoid(outputs).mean(dim=1)

            val_preds.extend(probs.cpu().numpy())
            val_targets.extend(y.numpy())

    # 4. Metric Calculation
    val_preds = np.array(val_preds)
    # Flatten targets to match predictions shape (N,)
    val_targets = np.array(val_targets).flatten()

    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for continuous features to calculate correlation
    # We use indices as feature names since raw names aren't preserved in the numpy array
    df_features = pd.DataFrame(data["X_val_cont"])
    df_features["error_magnitude"] = errors

    # Calculate correlation with error
    correlations = df_features.corrwith(df_features["error_magnitude"])

    # Drop the self-correlation of error_magnitude
    correlations = correlations.drop("error_magnitude")

    print("Top 10 continuous features correlated with error magnitude:")
    print(correlations.abs().sort_values(ascending=False).head(10))

    # 6. Conditional Submission
    threshold = 0.9971550270448856
    print(f"\n=== Submission Check ===")
    print(f"Threshold: {threshold}")
    print(f"Actual:    {final_auc}")

    if final_auc > threshold:
        print("Threshold met. Generating submission...")
        generate_submission(model, device, data)
    else:
        print("Threshold not met. Skipping submission generation.")


if __name__ == "__main__":
    main()
