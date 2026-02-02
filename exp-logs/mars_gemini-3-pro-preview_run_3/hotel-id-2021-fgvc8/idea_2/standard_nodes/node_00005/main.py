import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, apk
from library.dataset import get_dataloaders
from library.engine import train_loop, predict_and_submit
from library.model import HotelEfficientNet


def run_validation_and_analysis(val_loader, model, train_df, unique_ids, device):
    """
    Runs inference on the validation set to:
    1. Compute the Final Validation Metric (MAP@5).
    2. Perform failure analysis (Correlation between Error and Class Frequency).
    """
    model.eval()

    all_ap = []
    all_hotel_ids = []

    # Pre-calculate class frequencies from training data for analysis
    # unique_ids maps class_index -> hotel_id
    train_counts = train_df["hotel_id"].value_counts().to_dict()

    print("Running inference on validation set for analysis...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # labels are class indices

            # Forward pass (labels=None for inference mode -> cosine similarity)
            outputs = model(images, labels=None)

            # Get top 5 predictions
            _, topk_indices = torch.topk(outputs, k=5, dim=1)

            topk_indices = topk_indices.cpu().numpy()
            labels = labels.numpy()

            # Process batch
            for i in range(len(labels)):
                true_lbl_idx = labels[i]
                pred_lbl_indices = topk_indices[i]

                # Calculate Average Precision (AP@5) for this sample
                # apk expects lists
                ap = apk([true_lbl_idx], pred_lbl_indices.tolist(), k=5)
                all_ap.append(ap)

                # Retrieve original hotel_id for frequency lookup
                original_hotel_id = unique_ids[true_lbl_idx]
                all_hotel_ids.append(original_hotel_id)

    # 1. Compute Final Metric
    final_map = np.mean(all_ap)

    # 2. Failure Analysis
    # Error Magnitude = 1.0 - AP
    errors = 1.0 - np.array(all_ap)

    # Map hotel IDs to their training frequency
    frequencies = [train_counts.get(hid, 0) for hid in all_hotel_ids]

    # Calculate Correlation using Pandas
    df_analysis = pd.DataFrame({"error": errors, "freq": frequencies})

    print("\n--- Failure Analysis ---")
    if df_analysis["error"].std() > 0 and df_analysis["freq"].std() > 0:
        corr = df_analysis["error"].corr(df_analysis["freq"])
        print(
            f"Correlation between Error Magnitude (1-AP) and Class Frequency: {corr:.4f}"
        )
    else:
        print("Correlation could not be computed (insufficient variance).")

    return final_map


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    # Reducing epochs to 8 ensures completion within 2 hours while allowing convergence.
    Config.EPOCHS = 8

    print("Configuration:")
    Config.print_config()

    # 2. Data Loading
    print("\nLoading Data...")
    # Load raw train metadata for analysis
    train_df_raw = pd.read_csv(Config.TRAIN_CSV)

    # Get DataLoaders
    train_loader, val_loader, test_loader, num_classes, unique_ids = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Training
    print("\nStarting Training...")
    # train_loop handles model init, training, and saving best_model.pth
    train_loop(train_loader, val_loader, num_classes)

    # 4. Validation & Analysis
    print("\nLoading best model for final validation and analysis...")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Exiting.")
        return

    device = Config.DEVICE
    model = HotelEfficientNet(num_classes=num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    final_metric = run_validation_and_analysis(
        val_loader, model, train_df_raw, unique_ids, device
    )

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.14571255006929015

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(test_loader, num_classes, unique_ids)
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
