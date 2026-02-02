import os
import torch
import numpy as np
import pandas as pd
import scipy.stats
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, load_metadata, save_checkpoint
from library.dataset import get_dataset
from library.model import HybridSegFormer
from library.engine import train_one_epoch, valid_one_epoch, predict_with_z_scanning
from library.metrics import BCEDiceLoss, fbeta_score_numpy


def main():
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = Config.DEVICE
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    # We use the full provided patch set (412 train, 104 val) as it is small enough
    # to fit within the "Fast Baseline" requirement without explicit subsampling.
    train_dataset = get_dataset("train")
    val_dataset = get_dataset("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Using pretrained MiT-B2 backbone with U-Net decoder
    model = HybridSegFormer(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = BCEDiceLoss()

    # 4. Training Loop
    # Limiting to 10 epochs for a fast baseline execution within time limits
    NUM_EPOCHS = 10
    best_score = -1.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_score = valid_one_epoch(model, val_loader, criterion, device)

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(
                model, optimizer, epoch, val_score, filename="best_model.pth"
            )

    # 5. Final Validation & Failure Analysis
    # Load the best checkpoint to ensure we evaluate the optimal state
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    all_preds = []
    all_labels = []
    all_means = []  # Feature for failure analysis: Mean Intensity

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Calculate mean intensity per sample for failure analysis
            # images shape: (B, 3, H, W) -> Mean over (1, 2, 3) -> (B,)
            batch_means = images.mean(dim=[1, 2, 3]).cpu().numpy()
            all_means.extend(batch_means)

            # Inference
            outputs = model(images)
            preds_prob = torch.sigmoid(outputs)

            all_preds.append(preds_prob.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate results
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    all_means_np = np.array(all_means)

    # Calculate Final Metric
    final_metric = fbeta_score_numpy(all_preds_np, all_labels_np, beta=0.5)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Input Intensity
    # Error defined as mean absolute difference per sample
    # all_preds_np: (N, 1, H, W), all_labels_np: (N, 1, H, W)
    errors = np.abs(all_preds_np - all_labels_np).mean(axis=(1, 2, 3))

    if len(errors) > 1:
        correlation, p_value = scipy.stats.pearsonr(errors, all_means_np)
        print(
            f"Failure Analysis - Correlation between Error and Input Mean Intensity: {correlation:.4f} (p-value: {p_value:.4f})"
        )

    # 6. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD_SCORE = 0.597622633

    if final_metric > THRESHOLD_SCORE:
        test_df = load_metadata("test")

        # Run inference using Decoupled Volumetric Z-Scanning
        submission_data = predict_with_z_scanning(model, test_df, device)

        # Save submission
        sub_df = pd.DataFrame(submission_data)
        save_path = "./submission/submission.csv"
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"Final metric {final_metric} did not exceed threshold {THRESHOLD_SCORE}. Submission skipped."
        )


if __name__ == "__main__":
    main()
