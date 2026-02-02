import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided libraries
from library.dataset import get_loaders, SaltDataset, get_transforms
from library.model import ResNet34WideLinkNet
from library.engine import (
    train_one_epoch,
    evaluate,
    predict_test,
    set_seed,
)
from library.utils import do_kaggle_metric, unpad_image
from torch.optim.lr_scheduler import CosineAnnealingLR


def failure_analysis(model, loader, device):
    """
    Analyzes the correlation between error (1 - IoU) and depth.
    """
    model.eval()
    ious = []
    depths_list = []

    with torch.no_grad():
        for images, masks, depths, ids in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths_gpu = depths.to(device)

            logits = model(images, depths_gpu)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            depths_np = depths.numpy()  # Original depths (normalized)

            for i in range(len(probs_np)):
                # Calculate IoU for this single image
                p = (probs_np[i, 0] > 0.5).astype(np.uint8)
                t = (masks_np[i, 0] > 0.5).astype(np.uint8)

                intersection = np.sum((p * t) > 0)
                union = np.sum((p + t) > 0)
                if union == 0:
                    iou = 1.0 if np.sum(t) == 0 else 0.0
                else:
                    iou = intersection / union

                ious.append(iou)
                depths_list.append(depths_np[i].item())

    ious = np.array(ious)
    depths_arr = np.array(depths_list)
    errors = 1.0 - ious

    correlation = np.corrcoef(errors, depths_arr)[0, 1]
    print(
        f"Failure Analysis - Correlation between Error (1-IoU) and Depth: {correlation:.4f}"
    )


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    EPOCHS = 50  # Increased to maximize supervised convergence (Cite solution_lesson_node_00045)
    MAX_BATCHES = None

    # 2. Data Loading
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
    )

    # 3. Robust Supervised Training
    print("\n--- Robust Supervised Training ---")
    model = ResNet34WideLinkNet(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_score = -1.0
    best_threshold = 0.5
    best_model_path = "./working/best_model.pth"

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, max_batches=MAX_BATCHES
        )
        scheduler.step()

        # Adaptive Threshold Checkpointing (Cite solution_lesson_node_00033)
        score, thresh = evaluate(
            model, val_loader, device, optimize=True, max_batches=MAX_BATCHES
        )

        if score > best_score:
            best_score = score
            best_threshold = thresh
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New Best Model Saved! mAP: {best_score:.4f} at Threshold: {best_threshold:.4f}"
            )

    print(f"Training Complete. Best mAP: {best_score:.4f}")

    # 6. Final Evaluation & Optimization
    print("\n--- Final Evaluation ---")
    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    # Verify best score with optimal threshold
    final_score = evaluate(
        model,
        val_loader,
        device,
        threshold=best_threshold,
        max_batches=MAX_BATCHES,
    )
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    failure_analysis(model, val_loader, device)

    # 7. Submission
    if final_score > 0.7985:
        print("\n--- Generating Submission ---")
        predict_test(
            model,
            test_loader,
            device,
            threshold=best_threshold,
            output_path="./submission/submission.csv",
        )
    else:
        print(
            f"\nValidation score {final_score} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()
