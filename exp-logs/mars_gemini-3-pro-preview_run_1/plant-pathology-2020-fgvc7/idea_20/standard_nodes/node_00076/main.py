import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_class_weights
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import WeightedSoftTargetCrossEntropy
from library.engine import fit


def predict_ensemble(models, loader, device):
    """
    Predicts using an ensemble of models.
    Returns averaged probabilities.
    """
    avg_preds = None

    # Iterate through each model in the ensemble
    for model in models:
        model.eval()
        preds_list = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                outputs = model(images)
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                preds_list.append(probs.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)

        if avg_preds is None:
            avg_preds = preds
        else:
            avg_preds += preds

    # Average the predictions
    avg_preds /= len(models)
    return avg_preds


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Load Metadata
    # train_metadata contains the 80% of data available for training/cross-validation
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # val_metadata is the strict hold-out set for final evaluation
    df_val_holdout = pd.read_csv(Config.VAL_METADATA_PATH)

    # Calculate class weights on the full training set
    class_weights = calculate_class_weights(df_train_full, load_cached_data=True)

    # 3. Seed Averaging Ensemble Training
    # Train on full training data, validate on hold-out set, using multiple seeds
    # Cite Lesson 00025: Maximize training data (avoid fragmenting small datasets)
    # Cite Lesson 00055: Use Seed Averaging Ensembles for robustness
    # Cite Lesson 00058: Seed Averaging on fixed validation split > High-K CV on small data
    trained_models = []

    # Create Validation Dataset/Loader (Fixed for all seeds)
    val_dataset = AppleDataset(
        df_val_holdout, transforms=get_transforms("val"), mode="train"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    for seed in Config.SEEDS:
        print(f"\nTraining with Seed: {seed}")
        # Set seed for this run (affects initialization and augmentation)
        set_seed(seed)

        # Create Training Dataset/Loader (Full Data)
        train_dataset = AppleDataset(
            df_train_full, transforms=get_transforms("train"), mode="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Scheduler, Loss
        model = get_model(pretrained=Config.PRETRAINED, n_classes=Config.N_CLASSES)
        model.to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.EPOCHS
        )
        criterion = WeightedSoftTargetCrossEntropy(weight=class_weights)

        # Train the model
        save_path = os.path.join(Config.MODELS_DIR, f"resnet34_seed_{seed}.pth")
        fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            Config.EPOCHS,
            save_path,
        )

        # Load the best weights for this seed and keep in memory
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        trained_models.append(model)

    # 4. Final Validation on Hold-out Set
    # Ensemble Inference
    # Note: val_loader is already defined above as the hold-out loader
    val_probs = predict_ensemble(trained_models, val_loader, device)
    val_targets = df_val_holdout[Config.CLASSES].values

    # Calculate Metric
    final_val_auc = roc_auc_score(
        val_targets, val_probs, average="macro", multi_class="ovr"
    )
    print(f"Final Validation Metric: {final_val_auc}")

    # 5. Failure Analysis
    # Calculate Cross Entropy Error per sample
    val_probs_tensor = torch.tensor(val_probs).to(device)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32).to(device)

    epsilon = 1e-7
    val_probs_tensor = torch.clamp(val_probs_tensor, epsilon, 1.0 - epsilon)
    log_probs = torch.log(val_probs_tensor)
    # Error magnitude = Cross Entropy
    errors = -(val_targets_tensor * log_probs).sum(dim=1).cpu().numpy()

    # Extract meta-features from images
    widths = []
    heights = []
    intensities = []

    for idx, row in df_val_holdout.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(full_path)
        if img is not None:
            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            # Normalize and calculate mean intensity
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            intensities.append(img_rgb.mean())
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "intensity": intensities}
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.9901680711448418
    if final_val_auc > THRESHOLD:
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = AppleDataset(
            df_test, transforms=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_probs = predict_ensemble(trained_models, test_loader, device)

        # Create submission file
        submission = pd.DataFrame(test_probs, columns=Config.CLASSES)
        submission.insert(0, "image_id", df_test["image_id"])

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
