import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data import get_dataloaders
from library.model import AdaptiveResNetCRNN
from library.trainer import Trainer


def train_ensemble(train_loader, val_loader, device):
    """
    Trains the ensemble of models defined by the seeds in Config.
    """
    model_paths = []

    for seed in Config.SEEDS:
        print(f"\n{'='*40}")
        print(f"Training Ensemble Member with Seed: {seed}")
        print(f"{'='*40}")

        # 1. Set Seed for Reproducibility
        set_seed(seed)

        # 2. Initialize Model
        model = AdaptiveResNetCRNN().to(device)

        # 3. Initialize Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.LR_FACTOR,
            patience=Config.LR_PATIENCE,
            min_lr=Config.MIN_LR,
            verbose=True,
        )

        # 4. Initialize Trainer
        trainer = Trainer(model, optimizer, scheduler, device)

        # 5. Train
        save_name = f"model_seed_{seed}.pth"
        trainer.train_model(
            train_loader, val_loader, num_epochs=Config.EPOCHS, save_name=save_name
        )

        model_paths.append(os.path.join(Config.WORKING_DIR, save_name))

    return model_paths


def predict_ensemble(model_paths, loader, device):
    """
    Performs inference using the ensemble of trained models.
    Returns averaged probabilities and true labels (if available).
    """
    # Load all models
    models = []
    for path in model_paths:
        model = AdaptiveResNetCRNN().to(device)
        load_checkpoint(model, filename=os.path.basename(path), device=device)
        model.eval()
        models.append(model)

    all_preds = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            # Handle case where loader returns (data, target) or just (data, dummy)
            if len(batch_data) == 2:
                data, target = batch_data
            else:
                data = batch_data[0]
                target = None

            data = data.to(device)

            # Get predictions from each model
            batch_probs = []
            for model in models:
                logits = model(data)
                probs = torch.sigmoid(logits).squeeze(1)
                batch_probs.append(probs.cpu().numpy())

            # Average probabilities across ensemble
            avg_probs = np.mean(batch_probs, axis=0)

            all_preds.extend(avg_probs)
            if target is not None:
                all_targets.extend(target.numpy())

    return np.array(all_preds), np.array(all_targets)


def perform_failure_analysis(loader, preds, targets, device):
    """
    Analyzes the correlation between error magnitude and input signal statistics.
    """
    print("\n--- Failure Analysis ---")

    errors = np.abs(targets - preds)

    # Calculate simple signal features from the validation set
    # We need to iterate again or we could have stored them.
    # Iterating is safer for memory.

    spec_means = []
    spec_stds = []
    spec_maxs = []

    # We assume the loader order is deterministic (shuffle=False for val)
    idx = 0
    for data, _ in loader:
        # data shape: (B, 1, F, T)
        B = data.size(0)

        # Flatten spatial dims for stats: (B, F*T)
        flat_data = data.view(B, -1).numpy()

        spec_means.extend(np.mean(flat_data, axis=1))
        spec_stds.extend(np.std(flat_data, axis=1))
        spec_maxs.extend(np.max(flat_data, axis=1))

        idx += B

    spec_means = np.array(spec_means)
    spec_stds = np.array(spec_stds)
    spec_maxs = np.array(spec_maxs)

    # Calculate Correlations
    corr_mean, _ = pearsonr(errors, spec_means)
    corr_std, _ = pearsonr(errors, spec_stds)
    corr_max, _ = pearsonr(errors, spec_maxs)

    print(f"Correlation (Error vs Spec Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Spec Std):  {corr_std:.4f}")
    print(f"Correlation (Error vs Spec Max):  {corr_max:.4f}")

    # Interpretation
    strongest = max(
        [(abs(corr_mean), "Mean"), (abs(corr_std), "Std"), (abs(corr_max), "Max")]
    )
    print(f"Strongest feature association: {strongest[1]} (r={strongest[0]:.4f})")


def main():
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # Use cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Train Ensemble
    model_paths = train_ensemble(train_loader, val_loader, device)

    # 4. Validation Evaluation
    print("\nEvaluating Ensemble on Validation Set...")
    val_preds, val_targets = predict_ensemble(model_paths, val_loader, device)

    val_auc = roc_auc_score(val_targets, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    perform_failure_analysis(val_loader, val_preds, val_targets, device)

    # 6. Submission
    threshold = 0.9947519068503985

    if val_auc > threshold:
        print(
            f"\nValidation score ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        test_preds, _ = predict_ensemble(model_paths, test_loader, device)

        # Create Submission DataFrame
        df_test = pd.read_csv(Config.TEST_CSV)

        # Ensure lengths match
        if len(df_test) != len(test_preds):
            print(
                f"Warning: Mismatch in test set size. CSV: {len(df_test)}, Preds: {len(test_preds)}"
            )
            # Truncate or pad if necessary, but this shouldn't happen with correct loaders
            min_len = min(len(df_test), len(test_preds))
            df_test = df_test.iloc[:min_len]
            test_preds = test_preds[:min_len]

        submission = pd.DataFrame({"clip": df_test["clip"], "probability": test_preds})

        # Save
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

        # Verify
        print("Head of submission:")
        print(submission.head())
    else:
        print(
            f"\nValidation score ({val_auc}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
