import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Import from provided library
from library.config import train_config, path_config, label_config, audio_config
from library.utils import set_seed, LabelManager
from library.dataset import get_train_val_datasets, get_test_dataset
from library.model import DilatedEfficientNet
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    # Set seed for reproducibility
    set_seed(train_config.seed)

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config for fast baseline execution on A100
    # We increase batch size to leverage GPU memory and reduce epochs to fit time limit
    train_config.batch_size = 128
    train_config.epochs = (
        30  # Increased to accommodate harder augmentation (Noise Mixing)
    )
    train_config.num_workers = 4

    # Ensure directories exist
    os.makedirs(path_config.working_dir, exist_ok=True)
    os.makedirs(path_config.submission_path.rsplit("/", 1)[0], exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Initializing Data...")
    label_manager = LabelManager(load_cached_data=True)
    num_fine_classes = label_manager.get_num_classes()
    print(f"Fine-grained classes: {num_fine_classes}")

    # Load Datasets
    train_dataset, val_dataset = get_train_val_datasets(
        label_manager, load_cached_data=True
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation loader: shuffle=False to match metadata order for analysis
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing Model...")
    model = DilatedEfficientNet(num_classes=num_fine_classes)
    model = model.to(device)

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    print("Starting Training...")
    trainer.fit()

    # ---------------------------------------------------------
    # 5. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("Evaluating Best Model...")
    # Load best model state
    model.load_state_dict(torch.load(path_config.model_save_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_rms = []  # Feature for failure analysis

    # We need ground truth labels from the dataset dataframe for the 12-class metric
    # val_dataset.df contains 'label' (coarse) and 'fine_label'
    ground_truth_coarse = val_dataset.df["label"].values

    with torch.no_grad():
        batch_start_idx = 0
        for inputs, _ in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted_fine_indices = outputs.max(1)

            # Collect predictions
            all_preds.extend(predicted_fine_indices.cpu().numpy())

            # Calculate RMS for failure analysis (Batch, 1, F, T) -> mean over F, T -> sqrt
            # Simple proxy for signal energy
            batch_rms = torch.sqrt(torch.mean(inputs**2, dim=[1, 2, 3]))
            all_rms.extend(batch_rms.cpu().numpy())

    # Map fine-grained predictions to coarse labels
    mapped_preds = []
    for idx in all_preds:
        fine_label = label_manager.convert_idx_to_label(idx)
        coarse_label = label_manager.map_to_submission_label(fine_label)
        mapped_preds.append(coarse_label)

    # Calculate Accuracy
    final_acc = accuracy_score(ground_truth_coarse, mapped_preds)
    print(f"Final Validation Metric: {final_acc}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\nFailure Analysis:")
    # Calculate binary error (0 = correct, 1 = error)
    errors = [1 if p != t else 0 for p, t in zip(mapped_preds, ground_truth_coarse)]

    # Correlation between Error and Signal RMS
    if len(errors) > 0:
        correlation = np.corrcoef(errors, all_rms)[0, 1]
        print(f"Correlation between Error and Input RMS: {correlation:.4f}")

        if abs(correlation) < 0.1:
            print("Observation: Errors are weakly correlated with signal volume.")
        else:
            print("Observation: Signal volume may be influencing error rate.")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    # Threshold check
    if final_acc > 0.9857:
        print("\nValidation metric meets threshold. Generating submission...")

        test_dataset = get_test_dataset(label_manager)
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )

        test_preds = []
        test_fnames = (
            test_dataset.df["filepath"].apply(lambda x: os.path.basename(x)).tolist()
        )

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted_fine_indices = outputs.max(1)

                # Map batch
                for idx in predicted_fine_indices.cpu().numpy():
                    fine_label = label_manager.convert_idx_to_label(idx)
                    coarse_label = label_manager.map_to_submission_label(fine_label)
                    test_preds.append(coarse_label)

        # Create DataFrame
        submission_df = pd.DataFrame({"fname": test_fnames, "label": test_preds})

        # Save
        submission_df.to_csv(path_config.submission_path, index=False)
        print(f"Submission saved to {path_config.submission_path}")
        print(submission_df.head())

    else:
        print(
            f"\nValidation metric {final_acc:.5f} did not meet threshold 0.9857. Submission skipped."
        )


if __name__ == "__main__":
    main()
