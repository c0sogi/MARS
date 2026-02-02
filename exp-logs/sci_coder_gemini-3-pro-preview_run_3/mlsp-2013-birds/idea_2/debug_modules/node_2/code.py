import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import BirdDataset, load_dataset_df
from library.model import BirdClassifier
from library.train import train_and_evaluate, mixup_data, mixup_criterion
from library.utils import seed_everything, calculate_roc_auc


def main():
    print("==== Starting Library Demonstration ====")

    # 1. Setup & Configuration Override
    # Set seed for reproducibility
    seed_everything(42)

    # Modify Config for speed and demonstration purposes
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.PRETRAINED = False  # Skip downloading weights for this demo
    Config.BATCH_SIZE = 4  # Small batch size for verification

    print(f"Device: {Config.DEVICE}")
    print(f"Input Root: {Config.INPUT_ROOT}")

    # 2. Dataset Demonstration
    print("\n[1] Testing Dataset Loading and Processing...")
    try:
        # Load metadata
        df_train = load_dataset_df("train")
        print(f"Loaded training metadata with {len(df_train)} records.")

        # Create dataset with a small subset (first 10 records)
        subset_df = df_train.head(10)
        dataset = BirdDataset(subset_df, phase="train")

        # Fetch a single item
        spec, label = dataset[0]

        print(f"  Spectrogram Shape: {spec.shape}")
        print(f"  Label Vector Shape: {label.shape}")

        # Verifications
        assert torch.is_tensor(spec), "Spectrogram must be a torch Tensor"
        assert torch.is_tensor(label), "Label must be a torch Tensor"
        assert (
            spec.ndim == 3
        ), f"Spectrogram must be 3D (Channels, Freq, Time), got {spec.ndim}"
        assert spec.shape[0] == 1, f"Expected 1 channel, got {spec.shape[0]}"
        assert (
            spec.shape[1] == Config.N_MELS
        ), f"Expected {Config.N_MELS} Mel bands, got {spec.shape[1]}"
        assert (
            label.shape[0] == Config.NUM_CLASSES
        ), f"Expected {Config.NUM_CLASSES} classes, got {label.shape[0]}"

        print("  Dataset verification passed.")

    except Exception as e:
        print(f"  Dataset verification failed: {e}")
        raise e

    # 3. Model Demonstration
    print("\n[2] Testing Model Architecture...")
    try:
        # Initialize model
        model = BirdClassifier(pretrained=False)
        model.to(Config.DEVICE)
        model.eval()

        # Create dummy input batch: (Batch, Channel, Freq, Time)
        # Using the time dimension from the dataset sample
        time_dim = spec.shape[2]
        dummy_input = torch.randn(2, 1, Config.N_MELS, time_dim).to(Config.DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_input)

        print(f"  Input Shape: {dummy_input.shape}")
        print(f"  Output Logits Shape: {outputs.shape}")

        # Verifications
        assert outputs.shape == (
            2,
            Config.NUM_CLASSES,
        ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {outputs.shape}"

        print("  Model verification passed.")

    except Exception as e:
        print(f"  Model verification failed: {e}")
        raise e

    # 4. Augmentation Logic (Mixup)
    print("\n[3] Testing Mixup Augmentation...")
    try:
        # Create dummy targets
        dummy_targets = (
            torch.randint(0, 2, (2, Config.NUM_CLASSES)).float().to(Config.DEVICE)
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        # Apply Mixup
        mixed_x, y_a, y_b, lam = mixup_data(
            dummy_input, dummy_targets, alpha=1.0, device=Config.DEVICE
        )

        # Calculate Loss
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        # Verifications
        assert mixed_x.shape == dummy_input.shape, "Mixed input shape mismatch"
        assert isinstance(loss.item(), float), "Loss should be a scalar float"

        print(f"  Mixup Lambda: {lam:.4f}")
        print(f"  Mixup Loss: {loss.item():.4f}")
        print("  Mixup verification passed.")

    except Exception as e:
        print(f"  Mixup verification failed: {e}")
        raise e

    # 5. Metric Calculation
    print("\n[4] Testing Metric (ROC AUC)...")
    try:
        # Synthetic ground truth and predictions
        # Case: 3 samples, 3 classes
        y_true_synth = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        y_pred_synth = np.array([[0.9, 0.1, 0.2], [0.2, 0.8, 0.3], [0.8, 0.7, 0.9]])

        score = calculate_roc_auc(y_true_synth, y_pred_synth)
        print(f"  Calculated ROC AUC: {score:.4f}")

        assert 0.0 <= score <= 1.0, "ROC AUC score must be between 0 and 1"
        print("  Metric verification passed.")

    except Exception as e:
        print(f"  Metric verification failed: {e}")
        raise e

    # 6. Full Training Pipeline (Debug Mode)
    print("\n[5] Running Full Pipeline (Debug Mode)...")
    print("  This will simulate training with reduced data, epochs, and folds.")

    try:
        # Run the main training function in debug mode
        # debug=True sets EPOCHS=2, N_FOLDS=2, and subsets the data
        train_and_evaluate(debug=True)

        # Verify Submission File
        if os.path.exists(Config.OUTPUT_FILE):
            print(f"  Submission file found at: {Config.OUTPUT_FILE}")

            sub_df = pd.read_csv(Config.OUTPUT_FILE)
            print(f"  Submission DataFrame Shape: {sub_df.shape}")

            # In debug mode, train.py selects 20 test samples.
            # Total rows should be 20 samples * 19 classes = 380 rows.
            expected_rows = 20 * Config.NUM_CLASSES
            assert (
                len(sub_df) == expected_rows
            ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

            # Check columns
            assert (
                "Id" in sub_df.columns and "Probability" in sub_df.columns
            ), "Submission missing required columns 'Id' or 'Probability'"

            print("  Pipeline execution and submission generation verified.")
        else:
            raise FileNotFoundError(
                f"Submission file not created at {Config.OUTPUT_FILE}"
            )

    except Exception as e:
        print(f"  Pipeline execution failed: {e}")
        raise e

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
