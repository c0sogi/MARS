import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import CassavaDataset, get_transforms, Mixup
from library.models import CassavaModel
from library.engine import train_one_epoch, inference_tta


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Use Config.EPOCHS (10) to ensure models are well-tuned (Cite solution_lesson_node_00030)
    EPOCHS = Config.EPOCHS

    print(f"Starting Runfile Execution.")
    print(f"Device: {device}")
    print(f"Models: {Config.MODELS}")
    print(f"Epochs: {EPOCHS}")

    # 2. Data Loading
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Initialize Datasets
    # Train dataset with heavy augmentation
    train_ds = CassavaDataset(df_train, transforms=get_transforms("train"))

    # Validation dataset for inference (output_label=False to be compatible with inference_tta loop structure if needed,
    # though inference_tta handles tuples. We will use df_val['label'] for metrics.)
    val_ds = CassavaDataset(
        df_val, transforms=get_transforms("val"), output_label=False
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    trained_model_paths = []

    # 3. Training Loop (Heterogeneous Ensemble)
    for model_name in Config.MODELS:
        print(f"\n=======================================")
        print(f"Training Model: {model_name}")
        print(f"=======================================")

        # Initialize Model
        model = CassavaModel(
            model_name, num_classes=Config.NUM_CLASSES, pretrained=True
        )
        model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=Config.MIN_LR
        )

        # Mixup Regularization
        mixup_fn = Mixup()

        # Training Epochs
        for epoch in range(EPOCHS):
            avg_loss = train_one_epoch(
                epoch, model, train_loader, optimizer, device, mixup_fn
            )
            scheduler.step()

        # Save Final Model
        save_path = os.path.join(Config.OUTPUT_DIR, f"{model_name}_final.pth")
        torch.save({"state_dict": model.state_dict()}, save_path)
        trained_model_paths.append(save_path)
        print(f"Saved model to {save_path}")

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, mixup_fn
        torch.cuda.empty_cache()

    # 4. Ensemble Validation
    print("\n=======================================")
    print("Running Ensemble Validation (TTA)")
    print("=======================================")

    val_probs_list = []

    for model_path, model_name in zip(trained_model_paths, Config.MODELS):
        print(f"Inference with {model_name}...")
        model = CassavaModel(
            model_name, num_classes=Config.NUM_CLASSES, pretrained=False
        )
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        # Use Test-Time Augmentation for Validation to maximize score
        probs = inference_tta(model, val_loader, device)
        val_probs_list.append(probs)

        del model
        torch.cuda.empty_cache()

    # Average Probabilities (Soft Voting)
    avg_val_probs = torch.stack(val_probs_list).mean(dim=0)
    val_preds = torch.argmax(avg_val_probs, dim=1).numpy()
    val_targets = df_val["label"].values

    # Compute Metric
    final_acc = (val_preds == val_targets).mean()
    print(f"Final Validation Metric: {final_acc}")

    # 5. Failure Analysis
    print("\n=======================================")
    print("Performing Failure Analysis")
    print("=======================================")

    # Calculate Error Magnitude (1.0 - probability assigned to the correct class)
    # avg_val_probs is [N, 5], val_targets is [N]
    probs_correct = avg_val_probs[np.arange(len(val_targets)), val_targets].numpy()
    error_magnitude = 1.0 - probs_correct

    # Extract Input Features (Mean and Std of pixels)
    print("Extracting image statistics for correlation analysis...")
    pixel_means = []
    pixel_stds = []

    # Iterate through validation files to compute raw stats
    for rel_path in df_val["file_path"]:
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)
        img = cv2.imread(full_path)
        if img is not None:
            # Compute global mean and std
            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))
        else:
            pixel_means.append(0.0)
            pixel_stds.append(0.0)

    pixel_means = np.array(pixel_means)
    pixel_stds = np.array(pixel_stds)

    # Compute Correlations
    corr_mean = np.corrcoef(error_magnitude, pixel_means)[0, 1]
    corr_std = np.corrcoef(error_magnitude, pixel_stds)[0, 1]

    print(f"Correlation (Error Magnitude vs Mean Pixel): {corr_mean}")
    print(f"Correlation (Error Magnitude vs Std Pixel): {corr_std}")

    # 6. Submission
    THRESHOLD = 0.9078771695594126

    if final_acc > THRESHOLD:
        print("\n=======================================")
        print("Metric Threshold Passed. Generating Submission.")
        print("=======================================")

        # Prepare Test Loader
        test_ds = CassavaDataset(
            df_test, transforms=get_transforms("val"), output_label=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_probs_list = []

        # Inference on Test Set
        for model_path, model_name in zip(trained_model_paths, Config.MODELS):
            print(f"Predicting with {model_name}...")
            model = CassavaModel(
                model_name, num_classes=Config.NUM_CLASSES, pretrained=False
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)

            probs = inference_tta(model, test_loader, device)
            test_probs_list.append(probs)

            del model
            torch.cuda.empty_cache()

        # Ensemble
        avg_test_probs = torch.stack(test_probs_list).mean(dim=0)
        test_preds = torch.argmax(avg_test_probs, dim=1).numpy()

        # Save
        submission = pd.DataFrame(
            {"image_id": df_test["image_id"], "label": test_preds}
        )
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_acc} did not pass threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
