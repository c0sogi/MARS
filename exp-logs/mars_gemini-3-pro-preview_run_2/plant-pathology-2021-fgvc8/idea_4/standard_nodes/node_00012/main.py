import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric, save_model
from library.dataset import AppleDataset, get_transforms, load_metadata
from library.models import AppleClassifier
from library.losses import AsymmetricLoss
from library.engine import train_one_epoch, validate, predict


def run():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    # Override Config for fast baseline execution
    # Limiting to 3 epochs to ensure completion within 2 hours
    Config.epochs = 3

    print(f"Device: {device}")
    print(f"Models to train: {Config.model_names}")
    print(f"Epochs per model: {Config.epochs}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = load_metadata("train")
    val_df = load_metadata("val")

    # Create Datasets
    train_dataset = AppleDataset(train_df, transforms=get_transforms("train"))
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Store best model paths for ensemble
    best_model_paths = []

    # 3. Training Loop for Each Model
    for model_name in Config.model_names:
        print(f"\n{'='*20}\nTraining Model: {model_name}\n{'='*20}")

        # Initialize Model
        model = AppleClassifier(model_name, pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.epochs, eta_min=Config.min_lr
        )

        # Loss Function
        criterion = AsymmetricLoss()

        best_f1 = -1.0
        best_model_path = os.path.join(Config.work_dir, f"{model_name}_best.pth")
        best_model_paths.append(best_model_path)

        for epoch in range(1, Config.epochs + 1):
            print(f"\nEpoch {epoch}/{Config.epochs}")

            # Train
            train_loss = train_one_epoch(
                model, optimizer, train_loader, device, criterion, epoch
            )

            # Validate
            val_loss, val_f1 = validate(model, val_loader, device, criterion)

            # Step Scheduler
            scheduler.step()

            # Save Best Model
            if val_f1 > best_f1:
                print(f"New best F1 ({val_f1:.4f}). Saving model...")
                best_f1 = val_f1
                save_model(model, best_model_path)

        # Cleanup to free memory
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # 4. Ensemble Inference on Validation Set
    print(f"\n{'='*20}\nEnsemble Validation\n{'='*20}")

    ensemble_probs = np.zeros((len(val_dataset), Config.num_classes))

    for model_path, model_name in zip(best_model_paths, Config.model_names):
        print(f"Loading {model_name} from {model_path}...")
        model = AppleClassifier(model_name, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Get predictions (using predict function which handles TTA if enabled in Config)
        # Note: predict returns concatenated numpy array
        probs = predict(model, val_loader, device, use_tta=Config.use_tta)
        ensemble_probs += probs

        del model
        torch.cuda.empty_cache()

    # Average probabilities
    ensemble_probs /= len(Config.model_names)

    # Get Ground Truth
    # We iterate loader to ensure order matches
    all_targets = []
    for _, targets in val_loader:
        all_targets.append(targets.numpy())
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric
    final_f1 = calculate_metric(all_targets, ensemble_probs, threshold=0.5)
    print(f"Final Validation Metric: {final_f1}")

    # 5. Failure Analysis
    print(f"\n{'='*20}\nFailure Analysis\n{'='*20}")

    # Calculate error magnitude per sample (Mean Absolute Error)
    # Shape: (N_samples,)
    error_magnitudes = np.mean(np.abs(ensemble_probs - all_targets), axis=1)

    # Collect metadata features
    print("Collecting image metadata for failure analysis...")
    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []

    # We need to access the file paths from the validation dataframe
    # The loader order matches the dataframe order because shuffle=False
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])

        # File Size
        try:
            size = os.path.getsize(full_path)
        except:
            size = 0
        file_sizes.append(size)

        # Dimensions (OpenCV read is relatively fast for 3k images, but let's be efficient)
        # To be safe on time, we'll read just the header if possible, but cv2.imread is easiest
        img = cv2.imread(full_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": error_magnitudes,
            "file_size": file_sizes,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
        }
    )

    # Calculate Correlations
    correlations = analysis_df.corrwith(analysis_df["error"])
    print("\nCorrelation between Error Magnitude and Input Features:")
    print(correlations.drop("error"))  # Drop self-correlation

    # 6. Submission
    THRESHOLD_SCORE = 0.9228752356223593

    if final_f1 > THRESHOLD_SCORE:
        print(
            f"\nValidation score ({final_f1}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        test_df = load_metadata("test")
        test_dataset = AppleDataset(
            test_df, transforms=get_transforms("test")
        )  # Use test transforms
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Ensemble Prediction on Test Set
        test_ensemble_probs = np.zeros((len(test_dataset), Config.num_classes))

        for model_path, model_name in zip(best_model_paths, Config.model_names):
            print(f"Predicting with {model_name}...")
            model = AppleClassifier(model_name, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            probs = predict(model, test_loader, device, use_tta=Config.use_tta)
            test_ensemble_probs += probs

            del model
            torch.cuda.empty_cache()

        # Average
        test_ensemble_probs /= len(Config.model_names)

        # Generate Labels and Save
        # We manually implement the label generation here to use the ensemble probs
        predictions = []
        class_labels = Config.class_labels

        for i in range(len(test_ensemble_probs)):
            row_probs = test_ensemble_probs[i]
            indices = np.where(row_probs > 0.5)[0]

            if len(indices) > 0:
                labels = [class_labels[idx] for idx in indices]
                label_str = " ".join(labels)
            else:
                max_idx = np.argmax(row_probs)
                label_str = class_labels[max_idx]
            predictions.append(label_str)

        submission_df = pd.DataFrame({"image": test_df["image"], "labels": predictions})
        os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nValidation score ({final_f1}) did not meet threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    run()
