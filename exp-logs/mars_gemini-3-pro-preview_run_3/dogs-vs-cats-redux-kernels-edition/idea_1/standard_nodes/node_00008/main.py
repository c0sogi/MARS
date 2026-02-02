import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import build_model
from library.engine import train_one_epoch, evaluate, predict


def get_validation_preds(model, loader, device):
    """
    Runs inference on validation set and returns probabilities and labels.
    Used for failure analysis.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Forward pass with TTA (Cite solution_lesson_node_00007)
            outputs = model(images)
            outputs_flip = model(torch.flip(images, dims=[3]))

            # Sigmoid for probabilities
            probs = torch.sigmoid(outputs)
            probs_flip = torch.sigmoid(outputs_flip)

            # Average probabilities
            avg_probs = (probs + probs_flip) / 2.0
            probs = avg_probs.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def run_failure_analysis(model, val_loader, val_csv_path, device):
    """
    Calculates correlations between error magnitude and image metadata.
    """
    print("\n--- Performing Failure Analysis ---")

    # 1. Get Predictions on Validation Set
    probs, labels = get_validation_preds(model, val_loader, device)

    # 2. Calculate Error Magnitude
    # Error = |Label - Prediction|
    errors = np.abs(labels - probs)

    # 3. Load Metadata and Extract Features
    val_df = pd.read_csv(val_csv_path)

    # Ensure alignment between loader predictions and dataframe
    if len(val_df) != len(errors):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(errors)}"
        )
        return

    # Extract features efficiently
    widths = []
    heights = []
    file_sizes = []

    input_dir = Config.INPUT_DIR

    # Iterate through validation dataframe to extract image properties
    for _, row in val_df.iterrows():
        fpath = os.path.join(input_dir, row["filepath"])
        if os.path.exists(fpath):
            file_sizes.append(os.path.getsize(fpath))
            # Read image to get dimensions
            img = cv2.imread(fpath)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
        else:
            file_sizes.append(np.nan)
            widths.append(np.nan)
            heights.append(np.nan)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "file_size": file_sizes}
    )

    # Add Aspect Ratio as a derived feature
    analysis_df["aspect_ratio"] = analysis_df["width"] / analysis_df["height"]

    # Drop rows with missing data (corrupt images)
    analysis_df = analysis_df.dropna()

    # 4. Calculate Correlations
    # Using Pandas corr() which uses Pearson correlation by default
    correlations = analysis_df.corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # Identify highest correlation
    if not correlations.empty:
        max_corr_feat = correlations.abs().idxmax()
        max_corr_val = correlations[max_corr_feat]
        print(f"Strongest correlation: {max_corr_feat} ({max_corr_val:.4f})")


def main():
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    dataloaders = get_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 3. Model Construction
    model = build_model()

    # 4. Optimization Setup
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    # Cite solution_lesson_node_00007: AdamW and Cosine Annealing
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_state = None

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # 6. Final Evaluation & Output
    # Print the exact metric required
    print(f"Final Validation Metric: {best_val_loss}")

    # Restore best model for analysis and submission
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, Config.VAL_CSV, device)

    # 8. Submission
    baseline_metric = 0.014050961788691994
    if best_val_loss < baseline_metric:
        print(
            f"\nValidation metric ({best_val_loss:.6f}) improved over baseline ({baseline_metric:.6f})."
        )
        print("Generating submission...")
        ids, probs = predict(model, test_loader, device)

        submission_df = pd.DataFrame({"id": ids, "label": probs})

        # Ensure IDs are integers and sorted
        submission_df["id"] = submission_df["id"].astype(int)
        submission_df = submission_df.sort_values("id")

        # Save submission file
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric ({best_val_loss:.6f}) did not improve over baseline ({baseline_metric:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
