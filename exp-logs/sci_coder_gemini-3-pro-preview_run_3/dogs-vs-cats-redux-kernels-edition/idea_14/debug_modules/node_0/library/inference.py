import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    WORKING_DIR,
    SUBMISSION_PATH,
    MODEL_RESNET,
    MODEL_CONVNEXT,
    MODEL_MAXVIT,
    IMG_SIZE_TEACHER,
    IMG_SIZE_STUDENT,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.dataset import CatDogDataset
from library.models import create_model
from library.utils import load_checkpoint


def predict_with_tta(model, images):
    """
    Performs inference with Test Time Augmentation (Horizontal Flip).
    Returns the average probability.
    """
    # 1. Forward pass original
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Forward pass flipped
    images_flipped = torch.flip(images, dims=[3])  # Flip along width (N, C, H, W)
    logits_flipped = model(images_flipped)
    probs_flipped = torch.sigmoid(logits_flipped)

    # 3. Average
    return (probs_orig + probs_flipped) / 2.0


def inference_fn(
    resnet_checkpoint="resnet_best.pth",
    convnext_checkpoint="convnext_best.pth",
    maxvit_checkpoint="maxvit_best.pth",
):
    """
    Runs the inference pipeline for the heterogeneous ensemble.

    Args:
        resnet_checkpoint (str): Filename of the trained ResNet weights.
        convnext_checkpoint (str): Filename of the trained ConvNeXt weights.
        maxvit_checkpoint (str): Filename of the trained MaxViT weights.
    """
    print("Starting Inference...")

    # -------------------------------------------------------------------------
    # 1. Initialize Models
    # -------------------------------------------------------------------------
    print("Initializing models...")

    # ResNet-50 (Teacher)
    model_resnet = create_model(MODEL_RESNET, pretrained=False, num_classes=1)
    model_resnet.to(DEVICE)
    load_checkpoint(model_resnet, resnet_checkpoint)
    model_resnet.eval()

    # ConvNeXt-Small (Teacher)
    model_convnext = create_model(MODEL_CONVNEXT, pretrained=False, num_classes=1)
    model_convnext.to(DEVICE)
    load_checkpoint(model_convnext, convnext_checkpoint)
    model_convnext.eval()

    # MaxViT-Tiny (Student)
    model_maxvit = create_model(MODEL_MAXVIT, pretrained=False, num_classes=1)
    model_maxvit.to(DEVICE)
    load_checkpoint(model_maxvit, maxvit_checkpoint)
    model_maxvit.eval()

    # -------------------------------------------------------------------------
    # 2. Prepare Data Loaders
    # -------------------------------------------------------------------------
    print("Preparing DataLoaders...")

    # Dataset A: 256x256 for Teachers
    dataset_teacher = CatDogDataset(split="test", img_size=IMG_SIZE_TEACHER)
    loader_teacher = DataLoader(
        dataset_teacher,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Dataset B: 224x224 for Student
    dataset_student = CatDogDataset(split="test", img_size=IMG_SIZE_STUDENT)
    loader_student = DataLoader(
        dataset_student,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Inference Loop
    # -------------------------------------------------------------------------
    print(f"Processing {len(dataset_teacher)} test images...")

    all_ids = []
    all_probs = []

    # Zip loaders to process same batch indices simultaneously
    # Note: Both datasets rely on the same metadata sorted by ID, so alignment is guaranteed.
    with torch.no_grad():
        for (imgs_teacher, ids), (imgs_student, _) in zip(
            loader_teacher, loader_student
        ):
            imgs_teacher = imgs_teacher.to(DEVICE)
            imgs_student = imgs_student.to(DEVICE)

            # --- ResNet Prediction (TTA) ---
            prob_resnet = predict_with_tta(model_resnet, imgs_teacher)

            # --- ConvNeXt Prediction (TTA) ---
            prob_convnext = predict_with_tta(model_convnext, imgs_teacher)

            # --- MaxViT Prediction (TTA) ---
            prob_maxvit = predict_with_tta(model_maxvit, imgs_student)

            # --- Ensemble Averaging ---
            # Arithmetic mean of probabilities
            avg_prob = (prob_resnet + prob_convnext + prob_maxvit) / 3.0

            # Store results
            # ids is a tuple or tensor depending on collate, usually tensor for Ints
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy()

            probs_np = avg_prob.cpu().numpy().flatten()

            all_ids.extend(ids)
            all_probs.extend(probs_np)

    # -------------------------------------------------------------------------
    # 4. Save Submission
    # -------------------------------------------------------------------------
    print("Generating submission file...")

    df_sub = pd.DataFrame({"id": all_ids, "label": all_probs})

    # Ensure ID is integer
    df_sub["id"] = df_sub["id"].astype(int)

    # Sort just in case (though loaders are sorted)
    df_sub = df_sub.sort_values("id")

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(df_sub.head())
