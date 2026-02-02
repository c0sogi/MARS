import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library modules
import library.config as cfg
import library.utils as utils
import library.data as data
import library.model as model
import library.engine as engine


def main():
    # 1. Setup and Reproducibility
    # Set random seeds for fully reproducible results
    utils.seed_everything(cfg.SEED)

    # Configuration Overrides for Fast Baseline
    # We limit epochs to 10 to ensure the code completes quickly while allowing the model to converge.
    # The dataset size (approx 2600 training images) is small enough that we don't need to subsample it further.
    cfg.EPOCHS = 10

    # Device configuration
    device = cfg.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # load_cached_data=True utilizes parquet caches in ./working if they exist to speed up loading
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    # Load EfficientNet-B5 with GeM pooling
    net = model.RetinopathyModel(pretrained=True)
    net = net.to(device)

    # 4. Optimizer and Scheduler
    # Adam optimizer with weight decay
    optimizer = optim.Adam(
        net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler to adjust learning rate over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR
    )

    # 5. Training Loop
    print("Starting training...")
    # engine.run_training handles the training loop, per-epoch validation, and saving the best model checkpoint
    best_val_score = engine.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        save_path=cfg.MODEL_SAVE_PATH,
    )

    # 6. Evaluation and Failure Analysis
    print("Running final validation and failure analysis...")

    # Load the best saved model state for accurate analysis
    net.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    # Containers for analysis
    val_targets = []
    val_preds = []
    val_img_means = []
    val_img_stds = []

    # Inference on Validation Set
    # We disable gradients to optimize inference speed and memory usage
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = net(images)

            # Collect targets and raw predictions
            val_targets.extend(labels.cpu().numpy().tolist())
            val_preds.extend(outputs.detach().cpu().numpy().flatten().tolist())

            # Calculate image statistics for failure analysis
            # images shape: (B, C, H, W). We compute stats on the normalized tensor.
            # Average over spatial dims (2, 3) then channel dim (1)
            batch_means = images.mean(dim=(2, 3)).mean(dim=1).cpu().numpy()
            batch_stds = images.std(dim=(2, 3)).mean(dim=1).cpu().numpy()

            val_img_means.extend(batch_means)
            val_img_stds.extend(batch_stds)

    # Compute Final Metric (Quadratic Weighted Kappa)
    final_metric = utils.compute_score(val_targets, val_preds)
    # Print the metric in the required format
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate error magnitude
    y_true = np.array(val_targets)
    y_pred_raw = np.array(val_preds)
    # Clip and round predictions to integer labels for error calculation
    y_pred_rounded = np.round(np.clip(y_pred_raw, 0, 4)).astype(int)
    errors = np.abs(y_true - y_pred_rounded)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": val_img_means,
            "std_intensity": val_img_stds,
        }
    )

    # Calculate Spearman correlations between error magnitude and input features
    corr_mean = df_analysis["error"].corr(
        df_analysis["mean_intensity"], method="spearman"
    )
    corr_std = df_analysis["error"].corr(
        df_analysis["std_intensity"], method="spearman"
    )

    print("\n=== Failure Analysis ===")
    print(f"Correlation (Error vs Input Mean Intensity): {corr_mean}")
    print(f"Correlation (Error vs Input Std Intensity): {corr_std}")

    # 7. Submission
    # Generate submission only if validation metric exceeds the specified threshold
    THRESHOLD = 0.9080926132937968

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        engine.make_submission(net, test_loader, device, cfg.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
