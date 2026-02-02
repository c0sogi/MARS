import os
import torch
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything, fbeta_score
from library.data import get_dataloaders
from library.model import WideContextSegFormer
from library.loss import BCEDiceLoss
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import inference


def main():
    # 1. Setup & Seeding
    seed_everything(CFG.seed)
    os.makedirs(CFG.working_dir, exist_ok=True)

    # 2. Data Loading
    # Using load_cached_data=True to utilize pre-processed .npy files if available
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(CFG)

    # 3. Model Initialization
    print(f"Initializing model: {CFG.model_name}")
    model = WideContextSegFormer(CFG)
    model.to(CFG.device)

    # 4. Optimizer & Loss Setup
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience,
        min_lr=CFG.min_lr,
    )

    # 5. Training Loop
    best_val_score = -1.0
    print(f"Starting training for {CFG.epochs} epochs on device: {CFG.device}")

    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch + 1}/{CFG.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, CFG.device, criterion
        )

        # Validate
        val_loss, val_score = valid_one_epoch(model, val_loader, CFG.device, criterion)

        # Scheduler Step
        scheduler.step(val_score)

        # Save Best Model
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), CFG.model_path)
            print(f"New best model saved with F0.5 Score: {best_val_score:.4f}")

    # 6. Final Validation & Failure Analysis
    print("\n--- Running Final Validation & Failure Analysis ---")

    # Load best model for analysis
    if os.path.exists(CFG.model_path):
        model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))
    else:
        print("Error: Best model checkpoint not found.")
        return

    model.eval()

    all_preds = []
    all_targets = []
    patch_mean_intensities = []
    patch_mean_errors = []

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(CFG.device)
            masks = masks.to(CFG.device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Store for global metric
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

            # --- Failure Analysis Data Collection ---
            # Calculate mean intensity of input (avg over channels, height, width)
            # images shape: (B, C, H, W)
            batch_intensities = images.mean(dim=(1, 2, 3)).cpu()
            patch_mean_intensities.append(batch_intensities)

            # Calculate mean error (L1 distance) per patch
            # error shape: (B, 1, H, W) -> mean -> (B,)
            batch_errors = torch.abs(probs - masks).mean(dim=(1, 2, 3)).cpu()
            patch_mean_errors.append(batch_errors)

    # Concatenate all batches
    global_preds = torch.cat(all_preds)
    global_targets = torch.cat(all_targets)
    global_intensities = torch.cat(patch_mean_intensities).numpy()
    global_errors = torch.cat(patch_mean_errors).numpy()

    # Calculate Final Metric (Global F0.5)
    final_metric = fbeta_score(
        global_preds, global_targets, beta=CFG.beta, threshold=CFG.threshold
    )

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if len(global_errors) > 1:
        corr, _ = pearsonr(global_errors, global_intensities)
        print(f"Failure Analysis - Correlation (Error vs Input Intensity): {corr:.6f}")
        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation found. Model performance varies with ink/papyrus density."
            )
        else:
            print(
                "Observation: Low correlation. Errors are likely not driven by simple intensity shifts."
            )

    # 7. Submission Logic
    # Threshold defined in task description logic
    BASELINE_THRESHOLD = 0.597622633

    if final_metric > BASELINE_THRESHOLD:
        print(
            f"\nPerformance ({final_metric}) exceeds baseline ({BASELINE_THRESHOLD}). Generating submission..."
        )
        # Free up memory before inference
        del global_preds, global_targets, all_preds, all_targets
        torch.cuda.empty_cache()

        # Run Inference
        inference(model_path=CFG.model_path, submission_path=CFG.submission_path)
    else:
        print(
            f"\nPerformance ({final_metric}) did not exceed baseline ({BASELINE_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
