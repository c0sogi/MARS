import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

# Import from the provided library files
from library.config import CFG, seed_everything
from library.data import get_loaders
from library.models import CassavaClassifier
from library.trainer import fit
from library.inference import generate_ensemble_submission


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    seed_everything(CFG.seed)

    # Override configuration for a fast baseline run
    # 5 epochs are sufficient for transfer learning on A100 to reach high accuracy
    # while keeping runtime well within the limit.
    CFG.epochs = 5

    # Ensure working directories exist
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    print(
        f"Starting orchestration with Config: Epochs={CFG.epochs}, Device={CFG.device}"
    )

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders()

    # 3. Training Phase
    # Train Model A: ViT-Base (Global Context Expert)
    print(f"\n--- Training Model A: {CFG.model_a_name} ---")
    acc_a = fit(CFG.model_a_name, "model_a", train_loader, val_loader)
    print(f"Model A Training Completed. Best Validation Accuracy: {acc_a:.8f}")

    # Train Model B: EfficientNet-B4 (Local Texture Expert)
    print(f"\n--- Training Model B: {CFG.model_b_name} ---")
    acc_b = fit(CFG.model_b_name, "model_b", train_loader, val_loader)
    print(f"Model B Training Completed. Best Validation Accuracy: {acc_b:.8f}")

    # 4. Validation & Failure Analysis
    print("\n--- Starting Ensemble Validation & Failure Analysis ---")

    device = CFG.device

    # Load Best Checkpoints
    print("Loading best models for ensemble...")
    model_a = CassavaClassifier(CFG.model_a_name, CFG.num_classes, pretrained=False)
    state_a = torch.load(
        os.path.join(CFG.output_dir, "model_a_best.pth"), map_location=device
    )
    model_a.load_state_dict(state_a["state_dict"])
    model_a.to(device)
    model_a.eval()

    model_b = CassavaClassifier(CFG.model_b_name, CFG.num_classes, pretrained=False)
    state_b = torch.load(
        os.path.join(CFG.output_dir, "model_b_best.pth"), map_location=device
    )
    model_b.load_state_dict(state_b["state_dict"])
    model_b.to(device)
    model_b.eval()

    # Validation Loop with TTA and Stats Collection
    all_preds = []
    all_labels = []
    all_brightness = []

    tta_steps = CFG.tta_steps
    print(f"Running Inference on Validation Set with TTA={tta_steps}...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Collect feature for failure analysis: Image Brightness
            # Compute mean intensity of the normalized tensor (C, H, W)
            batch_brightness = images.mean(dim=(1, 2, 3)).cpu().numpy()
            all_brightness.extend(batch_brightness)

            # --- Model A Inference (TTA) ---
            batch_probs_a = []
            # 1. Original View
            out = model_a(images)
            batch_probs_a.append(torch.softmax(out, dim=1))
            # 2. Horizontal Flip
            if tta_steps >= 2:
                out_h = model_a(torch.flip(images, dims=[3]))
                batch_probs_a.append(torch.softmax(out_h, dim=1))
            # 3. Vertical Flip
            if tta_steps >= 3:
                out_v = model_a(torch.flip(images, dims=[2]))
                batch_probs_a.append(torch.softmax(out_v, dim=1))

            probs_a = torch.stack(batch_probs_a).mean(dim=0)

            # --- Model B Inference (TTA) ---
            batch_probs_b = []
            # 1. Original View
            out = model_b(images)
            batch_probs_b.append(torch.softmax(out, dim=1))
            # 2. Horizontal Flip
            if tta_steps >= 2:
                out_h = model_b(torch.flip(images, dims=[3]))
                batch_probs_b.append(torch.softmax(out_h, dim=1))
            # 3. Vertical Flip
            if tta_steps >= 3:
                out_v = model_b(torch.flip(images, dims=[2]))
                batch_probs_b.append(torch.softmax(out_v, dim=1))

            probs_b = torch.stack(batch_probs_b).mean(dim=0)

            # --- Ensemble Averaging ---
            ens_probs = (probs_a + probs_b) / 2.0

            all_preds.append(ens_probs.cpu().numpy())
            all_labels.append(labels.numpy())

    # Process Results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_brightness = np.array(all_brightness)

    # Calculate Final Metric
    predictions = np.argmax(all_preds, axis=1)
    final_acc = accuracy_score(all_labels, predictions)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    # Calculate error indicator (1 if prediction is wrong, 0 if correct)
    errors = (predictions != all_labels).astype(int)

    # Calculate Pearson correlation between error and image brightness
    if len(errors) > 1:
        corr = np.corrcoef(errors, all_brightness)[0, 1]
        print(
            f"Failure Analysis: Correlation between Error and Image Brightness: {corr:.4f}"
        )
    else:
        print("Failure Analysis: Insufficient samples for correlation.")

    # 5. Submission Generation
    threshold = 0.9022696929238986

    if final_acc > threshold:
        print(
            f"\nValidation metric ({final_acc:.8f}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_ensemble_submission(
            os.path.join(CFG.output_dir, "model_a_best.pth"),
            os.path.join(CFG.output_dir, "model_b_best.pth"),
            test_loader,
        )
    else:
        print(
            f"\nValidation metric ({final_acc:.8f}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
