import os
import torch
import numpy as np
import pandas as pd
from library.utils import Config, set_seed, get_device
from library.dataset import get_dataloaders, process_metadata
from library.model import LightweightMetricModel
from library.trainer import run_training
from library.inference import generate_submission


def analyze_failures(model, val_loader, val_df, train_df, device):
    """
    Computes final validation metric and performs failure analysis.
    """
    print("Running Failure Analysis on Validation Set...")
    model.eval()

    # 1. Prepare Metadata Stats
    # Map hotel_id to training frequency (Class Frequency)
    class_counts = train_df["hotel_id"].value_counts().to_dict()

    # Align validation metadata with loader
    # val_loader is created with shuffle=False, so it matches val_df order
    val_hotel_ids = val_df["hotel_id"].values

    all_ap_scores = []
    all_class_freqs = []

    ptr = 0
    total_samples = len(val_df)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)

            # Get top 5 predictions
            _, preds = outputs.topk(5, dim=1, largest=True, sorted=True)

            # Calculate AP@5 for each sample in the batch
            # labels: (B,) -> (B, 1)
            targets = labels.view(-1, 1)
            matches = preds == targets  # (B, 5) boolean

            # Compute score: sum(is_match * 1/(rank+1))
            # Since there is only 1 correct ground truth, this sum is the AP for that sample
            batch_scores = torch.zeros(batch_size, device=device)
            for k in range(5):
                batch_scores += matches[:, k].float() * (1.0 / (k + 1))

            all_ap_scores.extend(batch_scores.cpu().tolist())

            # Collect class frequencies for this batch
            current_hotel_ids = val_hotel_ids[ptr : ptr + batch_size]
            for hid in current_hotel_ids:
                all_class_freqs.append(class_counts.get(hid, 0))

            ptr += batch_size

    # 2. Compute Final Metric
    final_map5 = np.mean(all_ap_scores)
    print(f"Final Validation Metric: {final_map5}")

    # 3. Correlation Analysis
    # Error Magnitude = 1.0 - AP Score (0 is perfect, 1 is total failure)
    errors = 1.0 - np.array(all_ap_scores)
    freqs = np.array(all_class_freqs)

    if len(errors) > 0 and np.std(errors) > 0 and np.std(freqs) > 0:
        correlation = np.corrcoef(errors, freqs)[0, 1]
        print(
            f"Correlation between Error Magnitude and Class Frequency: {correlation:.4f}"
        )
    else:
        print("Correlation could not be computed (insufficient variance).")

    return final_map5


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # 2. Train
    # Using Config.NUM_EPOCHS with optimized Softmax Warmup (Cite {solution_lesson_node_00003})
    print("Starting Training...")
    run_training(
        load_cached_data=True,
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
    )

    # 3. Load Best Model
    print("\nLoading best model for analysis...")
    # Load metadata to get num_classes
    train_df, val_df, _, encoder_classes = process_metadata(load_cached_data=True)
    num_classes = len(encoder_classes)

    model = LightweightMetricModel(
        num_classes=num_classes,
        embedding_dim=Config.EMBEDDING_DIM,
        backbone_name=Config.BACKBONE,
        pretrained=False,
    )
    model = model.to(device)

    checkpoint_path = Config.BEST_MODEL_PATH
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle state_dict key if present
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: Best model checkpoint not found.")
        return

    # 4. Validation & Failure Analysis
    # Get val_loader (shuffle=False is guaranteed by get_dataloaders for val)
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)
    final_metric = analyze_failures(model, val_loader, val_df, train_df, device)

    # 5. Submission
    if final_metric > 0.5747:
        print(
            f"\nValidation Metric ({final_metric:.4f}) > 0.5747. Generating Submission..."
        )
        generate_submission(
            checkpoint_path=Config.BEST_MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
            load_cached_data=True,
        )
    else:
        print(
            f"\nValidation Metric ({final_metric:.4f}) <= 0.5747. Skipping Submission."
        )


if __name__ == "__main__":
    main()
