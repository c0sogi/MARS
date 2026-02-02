import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_pos_weights
from library.dataset import get_dataloaders
from library.models import BirdModel
from library.engine import train_one_epoch, validate, inference, EarlyStopping


def run_training(
    model_name, train_df, val_df, test_df=None, pseudo_labels=None, stage="teacher"
):
    """
    Generic training function for both Teacher and Student stages.
    """
    # Setup DataLoaders
    dataloaders = get_dataloaders(
        model_name=model_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        pseudo_labels=pseudo_labels,
        load_cached_data=True,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Initialize Model
    device = Config.DEVICE
    model = BirdModel(model_name, pretrained=True).to(device)

    # Loss Function (Weighted BCE)
    # Note: When using pseudo-labels, we still use the weights derived from the original train set
    # or we could re-calculate. For stability, we stick to weights from the clean train set.
    pos_weights = get_pos_weights(train_df, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Early Stopping
    early_stopping = EarlyStopping(patience=8, mode="max")

    best_auc = 0.0
    model_save_path = os.path.join(Config.WORKING_DIR, f"{stage}_{model_name}.pth")

    # Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)

        early_stopping(val_auc)
        if early_stopping.early_stop:
            break

    return best_auc, model_save_path


def main():
    # 1. Setup
    Config.setup()
    device = Config.DEVICE

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Models to use
    models = Config.MODELS_TO_RUN  # ["resnet18", "densenet121", "efficientnet_b0"]

    print(f"Starting execution on {device}...")

    # -------------------------------------------------------------------------
    # STAGE 1: TEACHER TRAINING (Supervised)
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Teacher Training ---")
    teacher_paths = {}

    for model_name in models:
        # print(f"Training Teacher: {model_name}")
        best_auc, save_path = run_training(
            model_name, train_df, val_df, stage="teacher"
        )
        teacher_paths[model_name] = save_path
        # print(f"Teacher {model_name} Best Val AUC: {best_auc:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 2: PSEUDO-LABELING (Inference on Test Set)
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Pseudo-Labeling ---")
    teacher_preds = []

    for model_name in models:
        # Load Teacher
        model = BirdModel(model_name, pretrained=False).to(device)
        model.load_state_dict(
            torch.load(teacher_paths[model_name], map_location=device)
        )

        # Get Test Loader (Resolution specific)
        dataloaders = get_dataloaders(
            model_name, test_df=test_df, load_cached_data=True
        )
        test_loader = dataloaders["test"]

        # Inference
        preds = inference(model, test_loader, device)
        teacher_preds.append(preds)

    # Average predictions to get Soft Pseudo Labels
    soft_pseudo_labels = np.mean(teacher_preds, axis=0)

    # -------------------------------------------------------------------------
    # STAGE 3: STUDENT TRAINING (Semi-Supervised)
    # -------------------------------------------------------------------------
    print("\n--- Stage 3: Student Training ---")
    student_paths = {}

    for model_name in models:
        # print(f"Training Student: {model_name}")
        # Pass test_df and pseudo_labels to enable semi-supervised training
        best_auc, save_path = run_training(
            model_name,
            train_df,
            val_df,
            test_df=test_df,
            pseudo_labels=soft_pseudo_labels,
            stage="student",
        )
        student_paths[model_name] = save_path
        # print(f"Student {model_name} Best Val AUC: {best_auc:.4f}")

    # -------------------------------------------------------------------------
    # FINAL EVALUATION (Ensemble of Students)
    # -------------------------------------------------------------------------
    print("\n--- Final Evaluation ---")

    # 1. Validate Student Ensemble
    student_val_preds = []
    val_targets = None

    for model_name in models:
        # Load Student
        model = BirdModel(model_name, pretrained=False).to(device)
        model.load_state_dict(
            torch.load(student_paths[model_name], map_location=device)
        )

        # Get Val Loader
        dataloaders = get_dataloaders(model_name, val_df=val_df, load_cached_data=True)
        val_loader = dataloaders["val"]

        # Inference on Val
        # We need targets for metric calc, extract them once
        if val_targets is None:
            all_targets = []
            for _, t in val_loader:
                all_targets.append(t.numpy())
            val_targets = np.concatenate(all_targets, axis=0)

        preds = inference(model, val_loader, device)
        student_val_preds.append(preds)

    # Average Student Predictions
    ensemble_val_preds = np.mean(student_val_preds, axis=0)

    # Calculate Final Metric (Macro AUC)
    from sklearn.metrics import roc_auc_score

    aucs = []
    for i in range(Config.NUM_CLASSES):
        if len(np.unique(val_targets[:, i])) > 1:
            aucs.append(roc_auc_score(val_targets[:, i], ensemble_val_preds[:, i]))

    final_metric = np.mean(aucs) if aucs else 0.5
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # FAILURE ANALYSIS
    # -------------------------------------------------------------------------
    # Calculate per-sample error (Mean Absolute Error averaged across classes)
    # Error shape: (N_val,)
    sample_errors = np.mean(np.abs(val_targets - ensemble_val_preds), axis=1)

    # Feature 1: Label Cardinality (Number of birds present)
    cardinality = np.sum(val_targets, axis=1)

    # Feature 2: Signal Intensity (Proxy for noise or loudness)
    # We need to load images to calculate this. Using the first model's loader to get images.
    # Note: This is computationally expensive if we reload everything, but we have cached images.
    # We'll use a simplified approach: just use the cardinality correlation which is required.
    # To strictly follow "input features", let's load the cached val images directly.
    try:
        val_imgs_cache = os.path.join(Config.WORKING_DIR, "val_images.npy")
        if os.path.exists(val_imgs_cache):
            val_imgs = np.load(val_imgs_cache, allow_pickle=True)
            # Calculate mean pixel intensity per image
            # Handle object array if widths differ, though EDA said they are constant
            if val_imgs.dtype == object:
                pixel_means = np.array([np.mean(img) for img in val_imgs])
            else:
                pixel_means = np.mean(val_imgs, axis=(1, 2))

            corr_intensity, _ = pearsonr(sample_errors, pixel_means)
            print(
                f"Correlation between Error and Input Signal Intensity: {corr_intensity:.4f}"
            )
    except Exception:
        pass

    corr_cardinality, _ = pearsonr(sample_errors, cardinality)
    print(f"Correlation between Error and Label Cardinality: {corr_cardinality:.4f}")

    # -------------------------------------------------------------------------
    # SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    threshold = 0.9129501920716607

    if final_metric > threshold:
        # Generate Test Predictions with Student Ensemble
        student_test_preds = []

        for model_name in models:
            model = BirdModel(model_name, pretrained=False).to(device)
            model.load_state_dict(
                torch.load(student_paths[model_name], map_location=device)
            )

            dataloaders = get_dataloaders(
                model_name, test_df=test_df, load_cached_data=True
            )
            test_loader = dataloaders["test"]

            preds = inference(model, test_loader, device)
            student_test_preds.append(preds)

        ensemble_test_preds = np.mean(student_test_preds, axis=0)

        # Format Submission
        # Format: Id,Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []
        rec_ids = test_df["rec_id"].values

        for idx, rec_id in enumerate(rec_ids):
            probs = ensemble_test_preds[idx]
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)

        # Sort by Id to match sample submission structure (optional but good practice)
        submission_df = submission_df.sort_values("Id")

        os.makedirs("submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        # print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
