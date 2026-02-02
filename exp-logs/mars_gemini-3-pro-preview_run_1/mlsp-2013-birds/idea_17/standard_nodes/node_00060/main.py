import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import warnings
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.data import get_dataloaders
from library.model import get_bird_model
from library.engine import fit, evaluate, run_inference, save_submission
from library.utils import set_seed

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure reproducibility
    set_seed(Config.SEED)

    # 2. Load Initial DataLoaders
    # train_loader_base: Labeled data (Fold 0)
    # val_loader: Validation data (Fold 0)
    # test_loader: Test data (Fold 1)
    train_loader_base, val_loader, test_loader = get_dataloaders(pseudo_labels_df=None)

    # ==========================================
    # STAGE 1: Teacher Ensemble Training
    # ==========================================
    teacher_preds_list = []

    # Train 3 independent teachers
    for i in range(Config.NUM_TEACHERS):
        # Initialize Model
        model = get_bird_model(pretrained=Config.PRETRAINED).to(device)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train
        model = fit(
            model=model,
            train_loader=train_loader_base,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=Config.EPOCHS,
            swa_start_epoch=Config.SWA_START_EPOCH,
            patience=100,  # Disable early stopping to ensure SWA runs
        )

        # Inference on Test Set (Standard inference for teachers)
        preds = run_inference(model, test_loader, device, tta=False)
        teacher_preds_list.append(preds)

        # Cleanup to save memory
        del model
        del optimizer
        torch.cuda.empty_cache()

    # Ensemble Predictions (Average)
    ensemble_preds = {}
    rec_ids = list(teacher_preds_list[0].keys())

    for rid in rec_ids:
        # Stack predictions: (Num_Teachers, Num_Classes)
        p_stack = np.stack([tp[rid] for tp in teacher_preds_list])
        # Average
        p_avg = np.mean(p_stack, axis=0)
        ensemble_preds[rid] = p_avg

    # Create Pseudo-Label DataFrame for Student 1
    data = []
    for rid, probs in ensemble_preds.items():
        row = {"rec_id": rid}
        for idx, p in enumerate(probs):
            row[f"species_{idx}"] = p
        data.append(row)

    pseudo_df_1 = pd.DataFrame(data)

    # ==========================================
    # STAGE 2: Student 1 Training
    # ==========================================

    # Get DataLoader with Pseudo-Labels combined
    train_loader_s1, _, _ = get_dataloaders(pseudo_labels_df=pseudo_df_1)

    # Initialize Student 1
    student1 = get_bird_model(pretrained=Config.PRETRAINED).to(device)
    optimizer_s1 = optim.AdamW(
        student1.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train Student 1
    student1 = fit(
        model=student1,
        train_loader=train_loader_s1,
        val_loader=val_loader,
        optimizer=optimizer_s1,
        device=device,
        epochs=Config.EPOCHS,
        swa_start_epoch=Config.SWA_START_EPOCH,
    )

    # Generate Refined Pseudo-Labels (Student 1 with TTA)
    preds_s1 = run_inference(student1, test_loader, device, tta=True)

    # Cleanup
    del student1
    del optimizer_s1
    torch.cuda.empty_cache()

    # Create Pseudo-Label DataFrame for Student 2
    data_s2 = []
    for rid, probs in preds_s1.items():
        row = {"rec_id": rid}
        for idx, p in enumerate(probs):
            row[f"species_{idx}"] = p
        data_s2.append(row)

    pseudo_df_2 = pd.DataFrame(data_s2)

    # ==========================================
    # STAGE 3: Student 2 Training
    # ==========================================

    # Get DataLoader with Refined Pseudo-Labels
    train_loader_s2, _, _ = get_dataloaders(pseudo_labels_df=pseudo_df_2)

    # Initialize Student 2
    student2 = get_bird_model(pretrained=Config.PRETRAINED).to(device)
    optimizer_s2 = optim.AdamW(
        student2.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train Student 2
    student2 = fit(
        model=student2,
        train_loader=train_loader_s2,
        val_loader=val_loader,
        optimizer=optimizer_s2,
        device=device,
        epochs=Config.EPOCHS,
        swa_start_epoch=Config.SWA_START_EPOCH,
    )

    # ==========================================
    # VALIDATION & SUBMISSION
    # ==========================================
    final_val_auc, _ = evaluate(student2, val_loader, device)

    # Print metric
    print(f"Final Validation Metric: {final_val_auc}")

    # Threshold Check
    THRESHOLD = 0.9594082190886809

    if final_val_auc > THRESHOLD:
        # Single forward pass for final submission
        final_preds = run_inference(student2, test_loader, device, tta=False)
        save_submission(final_preds, Config.SUBMISSION_PATH)

    # ==========================================
    # FAILURE ANALYSIS
    # ==========================================
    # 1. Calculate per-sample error on Validation Set
    student2.eval()
    val_errors = []
    val_rec_ids = []

    # Iterate val_loader to get IDs and compute error
    with torch.no_grad():
        for images, labels, rec_ids in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = student2(images)
            probs = torch.sigmoid(outputs)
            # Error metric: Mean Absolute Error across classes
            mae = torch.abs(probs - labels).mean(dim=1)

            val_errors.extend(mae.cpu().numpy())
            val_rec_ids.extend(rec_ids.numpy())

    val_errors = np.array(val_errors)

    # 2. Extract Features (Spectrogram Mean Intensity)
    val_df = pd.read_csv(Config.VAL_CSV)
    id_to_path = dict(zip(val_df.rec_id, val_df.file_path))

    img_means = []
    valid_indices = []

    for i, rid in enumerate(val_rec_ids):
        if rid not in id_to_path:
            continue

        rel_path = id_to_path[rid]
        # Map wav path to spectrogram path
        base_name = os.path.splitext(os.path.basename(rel_path))[0] + ".bmp"
        spec_path = os.path.join(Config.SPECTROGRAM_DIR, base_name)

        if os.path.exists(spec_path):
            img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_means.append(np.mean(img))
                valid_indices.append(i)

    if len(valid_indices) > 1:
        filtered_errors = val_errors[valid_indices]
        filtered_means = np.array(img_means)

        corr, _ = pearsonr(filtered_errors, filtered_means)
        print(f"Correlation between Error Magnitude and Spectrogram Intensity: {corr}")


if __name__ == "__main__":
    main()
