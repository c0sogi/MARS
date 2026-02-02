import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Patch tqdm to disable progress bars as per requirements
import tqdm.auto
import tqdm


def noop_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.auto.tqdm = noop_tqdm
tqdm.tqdm = noop_tqdm

# Import library modules after patching
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import EfficientNetB4Unet
from library.engine import train_one_epoch, evaluate
from library.inference import Predictor
from library.loss import HybridLoss


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between loss and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()
    criterion = HybridLoss()

    losses = []
    study_classes = []
    num_boxes_list = []

    with torch.no_grad():
        for images, masks, labels in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # Forward pass
            study_logits, mask_logits = model(images)

            # Calculate loss per item in batch
            # HybridLoss aggregates, so we manually compute per-sample for analysis
            # To be efficient, we'll just use the aggregated batch logic but looped or
            # approximate it. For accuracy, let's loop over batch items here (batch size is small, 8)

            # Convert labels to indices for class extraction
            if labels.dim() == 2:
                cls_indices = torch.argmax(labels, dim=1)
            else:
                cls_indices = labels.long()

            for i in range(images.size(0)):
                # Extract single sample
                s_logit = study_logits[i : i + 1]
                m_logit = mask_logits[i : i + 1]
                s_label = labels[i : i + 1]
                m_mask = masks[i : i + 1]

                # Compute loss
                loss_dict = criterion(s_logit, m_logit, s_label, m_mask)
                total_loss = loss_dict["loss"].item()

                losses.append(total_loss)
                study_classes.append(cls_indices[i].item())

                # Count boxes in ground truth mask
                m_np = m_mask[0, 0].cpu().numpy()
                # Simple contour count
                import cv2

                contours, _ = cv2.findContours(
                    m_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                num_boxes_list.append(len(contours))

    # Create DataFrame
    df_analysis = pd.DataFrame(
        {"loss": losses, "study_class": study_classes, "num_boxes": num_boxes_list}
    )

    # Calculate Correlations
    corr_class = df_analysis["loss"].corr(df_analysis["study_class"])
    corr_boxes = df_analysis["loss"].corr(df_analysis["num_boxes"])

    print(f"Correlation (Loss vs Study Class): {corr_class:.4f}")
    print(f"Correlation (Loss vs Num Boxes): {corr_boxes:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = EfficientNetB4Unet()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_map = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss, val_map = evaluate(model, val_loader, device)

        # Save Best
        if val_map > best_map:
            best_map = val_map
            print(f"New best mAP: {best_map:.5f}. Saving model...")
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

        # scheduler.step()

    # 5. Final Evaluation & Analysis
    print("\nTraining complete. Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Calculate Final Metric
    _, final_map = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_map}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 6. Submission
    THRESHOLD = 0.4729475001
    if final_map > THRESHOLD:
        print(
            f"\nValidation metric ({final_map}) > threshold ({THRESHOLD}). Generating submission..."
        )
        predictor = Predictor(model_path=Config.BEST_MODEL_PATH)
        predictor.generate_submission()
    else:
        print(
            f"\nValidation metric ({final_map}) <= threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
