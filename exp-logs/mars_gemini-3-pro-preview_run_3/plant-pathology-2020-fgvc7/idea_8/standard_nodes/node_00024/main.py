import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import cv2
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, get_class_weights, calculate_metric
from library.dataset import AppleDataset, get_transforms
from library.models import AppleEfficientNet, AppleMaxViT
from library.engine import train_one_epoch, validate, inference_with_tta


def run_training():
    """
    Orchestrates the 5-Fold CV training for EfficientNet and MaxViT.
    Returns a list of tuples containing model configuration and path to best weights.
    """
    seed_everything(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Load training metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Ensure stratify label exists
    if "stratify_label" not in train_df.columns:
        train_df["stratify_label"] = train_df[Config.CLASSES].idxmax(axis=1)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    train_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["stratify_label"])
    ):
        train_df.loc[val_idx, "fold"] = fold

    # Get class weights (computed globally, applicable due to stratification)
    class_weights = get_class_weights(load_cached_data=True)

    # Define architectures to train
    architectures = [
        ("EffNet", AppleEfficientNet, Config.IMG_SIZE_EFFNET, Config.MODEL_EFFNET),
        ("MaxViT", AppleMaxViT, Config.IMG_SIZE_MAXVIT, Config.MODEL_MAXVIT),
    ]

    trained_models = []

    for fold in range(Config.N_FOLDS):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")

        # Split Data
        fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        for arch_name, model_cls, img_size, model_name in architectures:
            print(f"\nTraining {arch_name} (Fold {fold})...")

            # Prepare Data
            train_dataset = AppleDataset(
                fold_train_df,
                transform=get_transforms("train", img_size),
                return_label=True,
            )
            val_dataset = AppleDataset(
                fold_val_df,
                transform=get_transforms("val", img_size),
                return_label=True,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Initialize Model
            model = model_cls(model_name=model_name, pretrained=True)
            model.to(Config.DEVICE)

            # Optimization
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=1e-6
            )

            # Training Loop
            best_score = 0.0
            patience_counter = 0
            model_save_path = os.path.join(
                Config.OUTPUT_DIR, f"{arch_name}_fold_{fold}_best.pth"
            )

            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    Config.DEVICE,
                    epoch,
                    class_weights,
                )

                val_loss, val_score, _, _ = validate(
                    model, val_loader, Config.DEVICE, class_weights
                )

                scheduler.step()

                # Save best model
                if val_score > best_score:
                    best_score = val_score
                    torch.save(model.state_dict(), model_save_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print(f"Fold {fold} {arch_name} Best AUC: {best_score:.4f}")
            trained_models.append(
                (arch_name, model_cls, img_size, model_name, model_save_path)
            )

            # Cleanup
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
            )
            torch.cuda.empty_cache()

    return trained_models


def run_validation_and_analysis(trained_models):
    """
    Evaluates the ensemble on the hold-out validation set and performs failure analysis.
    """
    print("\n==== Validation & Failure Analysis ====")

    val_df = pd.read_csv(Config.VAL_CSV)
    targets = val_df[Config.CLASSES].values

    # Accumulate predictions from all models
    ensemble_preds = np.zeros((len(val_df), Config.NUM_CLASSES))

    for arch_name, model_cls, img_size, model_name, path in trained_models:
        dataset = AppleDataset(
            val_df, transform=get_transforms("val", img_size), return_label=True
        )
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = model_cls(model_name=model_name, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()

        _, _, preds, _ = validate(model, loader, Config.DEVICE)
        ensemble_preds += preds

        del model, loader, dataset
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds /= len(trained_models)

    # Compute Final Metric
    final_metric = calculate_metric(targets, ensemble_preds)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis: Correlation between Error and Meta-features
    print("Performing failure analysis...")

    # Calculate error magnitude (1 - probability of the correct class)
    true_class_indices = np.argmax(targets, axis=1)
    # Extract probabilities corresponding to the true class
    prob_correct = ensemble_preds[np.arange(len(ensemble_preds)), true_class_indices]
    error_magnitude = 1.0 - prob_correct

    # Extract image stats
    meta_stats = {"brightness": [], "contrast": [], "red": [], "green": [], "blue": []}

    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(full_path)
        if img is None:
            for k in meta_stats:
                meta_stats[k].append(0)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        meta_stats["brightness"].append(np.mean(gray))
        meta_stats["contrast"].append(np.std(gray))

        mean_rgb = np.mean(img_rgb, axis=(0, 1))
        meta_stats["red"].append(mean_rgb[0])
        meta_stats["green"].append(mean_rgb[1])
        meta_stats["blue"].append(mean_rgb[2])

    print("\nCorrelation between Error Magnitude and Image Features:")
    for feat_name, values in meta_stats.items():
        if len(values) == len(error_magnitude):
            corr, _ = pearsonr(error_magnitude, values)
            print(f"  - {feat_name}: {corr:.4f}")

    return final_metric


def run_submission(trained_models):
    """
    Generates submission file using Test-Time Augmentation.
    """
    print("\n==== Generating Submission ====")

    test_df = pd.read_csv(Config.TEST_CSV)
    ensemble_preds = np.zeros((len(test_df), Config.NUM_CLASSES))

    for arch_name, model_cls, img_size, model_name, path in trained_models:
        dataset = AppleDataset(
            test_df, transform=get_transforms("test", img_size), return_label=False
        )
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = model_cls(model_name=model_name, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        # Use TTA for inference
        preds = inference_with_tta(model, loader, Config.DEVICE)
        ensemble_preds += preds

        del model, loader, dataset
        torch.cuda.empty_cache()

    # Average predictions
    ensemble_preds /= len(trained_models)

    # Create submission dataframe
    sub_df = pd.DataFrame({"image_id": test_df["image_id"]})
    for i, cls in enumerate(Config.CLASSES):
        sub_df[cls] = ensemble_preds[:, i]

    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def main():
    # 1. Train Models
    trained_models = run_training()

    # 2. Validate and Analyze
    metric = run_validation_and_analysis(trained_models)

    # 3. Generate Submission
    # We generate submission if metric is valid (>= 0.0) to ensure file creation.
    if metric >= 0.0:
        run_submission(trained_models)


if __name__ == "__main__":
    main()
