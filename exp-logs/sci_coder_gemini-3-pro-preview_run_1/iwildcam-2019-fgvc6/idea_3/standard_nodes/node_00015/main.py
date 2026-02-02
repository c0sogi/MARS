import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_logger, calculate_metrics
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration & Setup
    # Override Config for a fast baseline execution
    # 12 epochs is sufficient for ConvNeXt on this dataset size with A100
    # to get a good result without running for too long.
    RUN_EPOCHS = 12
    Config.EPOCHS = RUN_EPOCHS

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Initialize Logger
    logger = get_logger(log_file=os.path.join(Config.WORKING_DIR, "run.log"))
    logger.info("Starting execution of runfile.py...")

    # 2. Model Training
    # Initialize Trainer
    trainer = Trainer(debug=False)

    # Run training
    # We explicitly pass epochs here because the default argument in fit()
    # was bound at import time.
    trainer.fit(epochs=RUN_EPOCHS)

    # 3. Validation Assessment & Failure Analysis Data Collection
    logger.info("Performing Validation Assessment and Failure Analysis...")

    # Determine which model to use (EMA is preferred for inference)
    model = trainer.model
    if trainer.ema_model:
        model = trainer.ema_model.module
        logger.info("Using EMA model for evaluation.")

    # Load the best weights saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        checkpoint = torch.load(
            Config.BEST_MODEL_PATH, map_location=Config.get_device()
        )
        # Handle state dict keys if necessary (trainer saves clean keys)
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded best model weights from {Config.BEST_MODEL_PATH}")

    model.eval()
    val_loader = trainer.val_loader
    device = Config.get_device()

    # Lists to store data
    all_preds = []
    all_targets = []
    all_probs = []  # Probability of the true class

    # Features for analysis
    feat_brightness = []
    feat_contrast = []

    # Inference Loop
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Calculate Error Magnitude data: Probability assigned to the True Class
            # gather: along dim 1, pick values at indices specified by targets
            true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            all_probs.extend(true_class_probs.cpu().numpy())

            # Calculate Image Features (on GPU for speed, then move to CPU)
            # Images are [B, 3, H, W].
            # Brightness: Mean value. Contrast: Std Dev.
            # Note: Images are normalized, so these are relative stats.
            batch_brightness = images.mean(dim=[1, 2, 3])
            batch_contrast = images.std(dim=[1, 2, 3])

            feat_brightness.extend(batch_brightness.cpu().numpy())
            feat_contrast.extend(batch_contrast.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_true_prob = np.array(all_probs)

    # 4. Metric Calculation
    final_f1 = calculate_metrics(y_true, y_pred)
    # Print exactly as required
    print(f"Final Validation Metric: {final_f1}")

    # 5. Failure Analysis
    logger.info("Calculating correlations for failure analysis...")

    # Error Magnitude: 0.0 if model is 100% confident in right class,
    # 1.0 if model is 0% confident in right class.
    # This captures "how wrong" the model is better than binary 0/1.
    error_magnitude = 1.0 - y_true_prob

    features = {
        "Brightness": np.array(feat_brightness),
        "Contrast": np.array(feat_contrast),
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, vals in features.items():
        # Compute Pearson correlation
        if len(vals) > 1:
            # np.corrcoef returns matrix, [0,1] is the correlation
            corr = np.corrcoef(error_magnitude, vals)[0, 1]
            print(f"Feature: {name}, Correlation: {corr:.4f}")
        else:
            print(f"Feature: {name}, Correlation: NaN")

    # 6. Submission
    THRESHOLD = 0.44583477715072195

    if final_f1 > THRESHOLD:
        logger.info(
            f"Validation F1 ({final_f1:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # The trainer class has a method for this which loads the best model and saves csv
        trainer.predict_test_set()
    else:
        logger.warning(
            f"Validation F1 ({final_f1:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
