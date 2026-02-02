import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.model import setup_phase


def analyze_failures(trainer, val_loader, val_df):
    """
    Performs failure analysis by correlating prediction error with input image features.
    """
    print("\n--- Performing Failure Analysis ---")
    trainer.model.eval()

    # Use reduction='none' to get loss per sample
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []

    # 1. Calculate per-sample loss
    # val_loader is shuffle=False, so order matches val_df
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(trainer.device)
            labels = labels.to(trainer.device)

            outputs = trainer.model(images)
            batch_losses = criterion(outputs, labels)
            losses.extend(batch_losses.cpu().numpy())

    val_df["loss"] = losses

    # 2. Extract image metadata features
    print("Extracting image metadata for correlation analysis...")
    widths, heights, areas, aspect_ratios = [], [], [], []

    for idx, row in val_df.iterrows():
        # Construct full path: ./input/train/<id>.jpg
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image to get dimensions
        img = cv2.imread(full_path)
        if img is None:
            # Fallback if file read fails (unlikely given verification)
            h, w = Config.RESIZE_SIZE, Config.RESIZE_SIZE
        else:
            h, w = img.shape[:2]

        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        aspect_ratios.append(w / h if h > 0 else 0)

    val_df["width"] = widths
    val_df["height"] = heights
    val_df["area"] = areas
    val_df["aspect_ratio"] = aspect_ratios

    # 3. Calculate Correlations
    features = ["width", "height", "area", "aspect_ratio"]
    print("\nCorrelation between Error Magnitude (Loss) and Input Features:")

    for feat in features:
        # Calculate Pearson correlation coefficient
        if val_df[feat].std() > 0:
            corr = np.corrcoef(val_df["loss"], val_df[feat])[0, 1]
        else:
            corr = 0.0
        print(f"{feat}: {corr:.4f}")


def run():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True
    )

    # Load validation metadata dataframe for failure analysis
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # 3. Initialize Trainer
    trainer = Trainer()

    # 4. Phase 1: Head Adaptation
    print(
        f"\n--- Starting Phase 1: Head Adaptation ({Config.PHASE1_EPOCHS} Epochs) ---"
    )
    optimizer_p1 = setup_phase(trainer.model, "phase1")

    best_metric = float("inf")

    for epoch in range(Config.PHASE1_EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader, optimizer_p1)
        val_loss, val_metric = trainer.validate(val_loader)
        print(
            f"Phase 1 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val LogLoss={val_metric:.4f}"
        )

        # Global Metric Tracking (Cite solution_lesson_node_00008)
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Phase 1: New best model saved! (LogLoss: {best_metric:.4f})")

    # 5. Phase 2: Fine-Tuning
    print(f"\n--- Starting Phase 2: Fine-Tuning ({Config.PHASE2_EPOCHS} Epochs) ---")
    optimizer_p2 = setup_phase(trainer.model, "phase2")

    # Do NOT reset best_metric here. Continue from Phase 1 best.
    patience_counter = 0

    for epoch in range(Config.PHASE2_EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader, optimizer_p2)
        val_loss, val_metric = trainer.validate(val_loader)

        print(
            f"Phase 2 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val LogLoss={val_metric:.4f}"
        )

        # Early Stopping Logic
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved! (LogLoss: {best_metric:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Evaluation
    print("\n--- Final Evaluation ---")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
        print("Loaded best model weights.")
    else:
        print("Warning: No best model found. Using current weights.")

    final_loss, final_metric = trainer.validate(val_loader)
    # Required output format
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    analyze_failures(trainer, val_loader, val_df)

    # 8. Conditional Submission
    THRESHOLD = 0.2024226008255145

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_loader, classes)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
