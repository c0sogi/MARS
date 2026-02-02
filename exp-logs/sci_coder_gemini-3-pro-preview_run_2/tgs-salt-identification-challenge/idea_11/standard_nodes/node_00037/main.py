import os
import random
import numpy as np
import torch
from library.trainer import SaltTrainer
from library.utils import calculate_iou_batch, _calculate_metric_from_iou


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(42)
    print("Initializing Fast Baseline Run...")

    # 2. Train
    # We limit epochs to 20 for a fast baseline execution as requested.
    # The A100 GPU can handle this quickly.
    trainer = SaltTrainer(epochs=20, patience=5, batch_size=32, learning_rate=1e-4)

    # Train returns the best probability threshold found during validation sweeps
    best_threshold = trainer.train()

    # 3. Validation & Metric Calculation
    print("\nRunning Final Validation Assessment...")

    # Load the best model weights
    model_path = "./working/best_model.pth"
    if os.path.exists(model_path):
        trainer.model.load_state_dict(
            torch.load(model_path, map_location=trainer.device)
        )

    trainer.model.eval()

    all_preds = []
    all_targets = []
    all_depths = []

    # Inference loop on validation set
    with torch.no_grad():
        for images, masks, depths, _ in trainer.val_loader:
            images = images.to(trainer.device)
            masks = masks.to(trainer.device)
            depths_gpu = depths.to(trainer.device)

            # Forward pass
            logits = trainer.model(images, depths_gpu)
            probs = torch.sigmoid(logits)

            # Crop back to 101x101 (removing the 128x128 padding)
            # Padding was symmetric: (128-101)/2 = 13.5 -> Top 13, Bottom 14
            # Indices: 13 to 13+101 = 114
            probs_cropped = probs[:, :, 13:114, 13:114]
            masks_cropped = masks[:, :, 13:114, 13:114]

            all_preds.append(probs_cropped.cpu().numpy())
            all_targets.append(masks_cropped.cpu().numpy())
            all_depths.extend(depths.numpy().flatten())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_depths = np.array(all_depths)

    # Calculate IoU per image using the best binarization threshold
    ious = calculate_iou_batch(all_preds, all_targets, threshold=best_threshold)

    # Calculate the competition metric (mAP over IoU thresholds 0.5-0.95)
    final_metric = _calculate_metric_from_iou(ious)

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Error Magnitude (1.0 - IoU)
    # We want to see if higher error correlates with depth
    errors = 1.0 - ious

    if len(all_depths) > 0:
        # Calculate Pearson correlation
        correlation = np.corrcoef(all_depths, errors)[0, 1]
        print(
            f"Correlation between Depth and Error Magnitude (1-IoU): {correlation:.8f}"
        )

        if abs(correlation) > 0.1:
            print(
                "  -> Significant correlation detected. Depth influences performance."
            )
        else:
            print("  -> Low correlation. Performance is relatively invariant to depth.")

    # 5. Submission
    # Generate submission only if metric condition is met
    if final_metric > 0.7985:
        print(f"\nMetric {final_metric} > 0.7985. Generating submission...")
        trainer.predict_test(best_threshold)
    else:
        print(f"\nMetric {final_metric} <= 0.7985. Skipping submission generation.")


if __name__ == "__main__":
    main()
