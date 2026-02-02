import os
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.trainer import Trainer


def run_validation_and_analysis(trainer):
    """
    Performs validation inference, calculates metrics, and analyzes failure modes.
    Returns the final AUC score.
    """
    print("\n" + "=" * 40)
    print("Starting Validation and Failure Analysis")
    print("=" * 40)

    # 1. Load the best model weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        return 0.0

    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint)
    trainer.model.eval()

    all_preds = []
    all_targets = []

    # 2. Inference Loop
    # We use the validation loader from the trainer
    # Disable gradients for efficiency and speed
    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in trainer.val_loader:
            inputs = inputs.to(trainer.device)

            # Forward pass
            logits = trainer.model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    # 3. Calculate Final Metric
    val_auc = roc_auc_score(all_targets, all_preds)
    # Print exactly as required by the task
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Access the underlying dataset to get raw spectrograms
    # The WhaleDataset stores 'specs' as a numpy array (N, F, T)
    # We use these raw specs to calculate signal features (loudness, contrast)
    # because the 'inputs' in the loader are instance-normalized (mean=0, std=1),
    # which removes absolute amplitude information needed for this analysis.
    val_dataset = trainer.val_loader.dataset
    raw_specs = val_dataset.specs

    # Calculate features: Mean (Signal Strength) and Std (Dynamic Range)
    # Flatten frequency and time dimensions for calculation
    raw_specs_flat = raw_specs.reshape(raw_specs.shape[0], -1)
    spec_means = raw_specs_flat.mean(axis=1)
    spec_stds = raw_specs_flat.std(axis=1)

    # Calculate Error Magnitude: |y_true - y_pred|
    errors = np.abs(np.array(all_targets) - np.array(all_preds))

    # Calculate Pearson Correlations
    if len(errors) > 1:
        corr_mean, _ = pearsonr(errors, spec_means)
        corr_std, _ = pearsonr(errors, spec_stds)
    else:
        corr_mean, corr_std = 0.0, 0.0

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"  Spectrogram Mean Intensity: {corr_mean:.10f}")
    print(f"  Spectrogram Std Dev: {corr_std:.10f}")

    return val_auc


def main():
    # 1. Configuration Overrides
    # Optimize for a fast baseline execution on A100
    # 15 Epochs is sufficient for EfficientNet-B0 to converge on this dataset size
    Config.EPOCHS = 15
    Config.T_MAX = 15  # Update scheduler cycle to match epochs
    Config.NUM_WORKERS = 8  # Utilize available vCPUs for faster data loading

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 2. Initialize Trainer
    # This loads data (caching if needed) and builds the model
    trainer = Trainer()

    # 3. Train
    # Execute the training loop with Mixup and Early Stopping
    trainer.fit()

    # 4. Validate and Analyze
    # Compute metrics and analyze errors on the hold-out set
    val_auc = run_validation_and_analysis(trainer)

    # 5. Conditional Submission
    # Only submit if the model meets the high performance threshold
    THRESHOLD = 0.9913801393656689

    if val_auc > THRESHOLD:
        print(f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        trainer.predict()
    else:
        print(f"\nValidation AUC ({val_auc}) does not meet threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
