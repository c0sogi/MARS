import os
import torch
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score

from library.config import PathConfig, TrainConfig, AudioConfig, ModelConfig
from library.trainer import Trainer
from library.dataset import SpeechCommandsDataset, get_dataloaders
from library.model import AudioEfficientNetV2
from library.utils import set_seed


def main():
    # 1. Setup and Configuration
    set_seed(42)
    path_config = PathConfig()

    # Configure for a fast but effective baseline
    # We override some defaults to ensure it fits within the time limit while maintaining performance
    train_config = TrainConfig()
    train_config.epochs = (
        25  # Increased to 25 to improve convergence and reach accuracy threshold
    )
    train_config.batch_size = 32
    train_config.num_workers = 4

    print("=== Starting Training Pipeline ===")
    print(f"Epochs: {train_config.epochs}")
    print(f"Batch Size: {train_config.batch_size}")

    # 2. Training Loop
    trainer = Trainer(train_config)
    trainer.train()

    # 3. Load Best Model for Evaluation
    print("\n=== Loading Best Model for Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_config = AudioConfig()
    model_config = ModelConfig()

    # Re-initialize model structure
    model = AudioEfficientNetV2(
        config=model_config, num_classes=audio_config.num_classes
    )
    model.to(device)

    # Load checkpoint
    checkpoint_path = path_config.model_checkpoint_path
    if os.path.exists(checkpoint_path):
        # Fix: Set weights_only=False to allow loading custom config objects stored in the checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        # Checkpoint contains 'state_dict'
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded best model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    model.eval()

    # 4. Validation & Metrics
    print("\n=== Running Validation ===")
    # We use the standard validation loader
    _, val_loader = get_dataloaders(batch_size=32, num_workers=4, debug=False)

    all_preds = []
    all_targets = []
    all_features_mean = []
    all_features_std = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Collect input stats for failure analysis (Spectrogram stats)
            # inputs shape: (B, 1, F, T)
            # Calculate mean and std per sample in the batch
            batch_means = inputs.view(inputs.size(0), -1).mean(dim=1).cpu().numpy()
            batch_stds = inputs.view(inputs.size(0), -1).std(dim=1).cpu().numpy()
            all_features_mean.extend(batch_means)
            all_features_std.extend(batch_stds)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_acc = accuracy_score(all_targets, all_preds)
    # Print the exact metric format required
    print(f"Final Validation Metric: {val_acc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = (np.array(all_preds) != np.array(all_targets)).astype(int)

    # Correlation with features
    if np.std(errors) > 0:
        corr_mean = np.corrcoef(errors, all_features_mean)[0, 1]
        corr_std = np.corrcoef(errors, all_features_std)[0, 1]
        print("Correlation between Error Magnitude and Input Features:")
        print(f"  Error vs Spectrogram Mean: {corr_mean:.10f}")
        print(f"  Error vs Spectrogram Std:  {corr_std:.10f}")
    else:
        print(
            "Failure Analysis: Perfect accuracy or constant error, cannot compute correlation."
        )

    # 6. Submission Generation
    threshold = 0.9866209549293419
    if val_acc > threshold:
        print(
            f"\nValidation accuracy {val_acc} > {threshold}. Generating submission..."
        )
        generate_submission(model, device, path_config, audio_config)
    else:
        print(f"\nValidation accuracy {val_acc} <= {threshold}. Skipping submission.")


def generate_submission(model, device, path_config, audio_config):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    print("Initializing Test Dataset...")
    # Disable caching for test to avoid long startup time for 150k files
    test_dataset = SpeechCommandsDataset(
        split="test", transform=False, cache_data=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,  # Larger batch size for faster inference
        shuffle=False,  # Critical: must be sequential to match fnames
        num_workers=4,
        pin_memory=True,
    )

    idx_to_label = test_dataset.idx_to_label

    fnames = []
    predictions = []

    model.eval()
    print("Running Inference on Test Set...")

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Get fnames for this batch
            # Since shuffle=False, we can slice the dataframe
            start_idx = i * test_loader.batch_size
            end_idx = start_idx + inputs.size(0)

            # Ensure we don't go out of bounds (though DataLoader handles last batch size correctly)
            # The slice corresponds exactly to the batch
            batch_fnames = test_dataset.df.iloc[start_idx : start_idx + inputs.size(0)][
                "fname"
            ].tolist()

            fnames.extend(batch_fnames)
            predictions.extend(preds.cpu().numpy())

            if (i + 1) % 100 == 0:
                print(f"Processed {len(predictions)} / {len(test_dataset)} samples...")

    # Map predictions to labels
    pred_labels = [idx_to_label[p] for p in predictions]

    # Create DataFrame
    df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})

    # Save
    sub_path = path_config.submission_path
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    main()
