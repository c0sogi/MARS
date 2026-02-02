import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library
from library.config import Config
from library.utils import seed_everything, map_per_image, map5
from library.dataset import get_loaders, get_label_encoder
from library.models import get_model
from library.engine import train_one_epoch, validate, inference
from library.pseudo_labeling import generate_pseudo_labels


# -------------------------------------------------------------------------
# Helper Class for Ensemble
# -------------------------------------------------------------------------
class EnsembleModel(nn.Module):
    """
    Wraps multiple models to act as a single nn.Module.
    Averages the outputs (logits or cosine similarities) of the sub-models.
    """

    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, labels=None):
        # Collect outputs from all models
        outputs = []
        for model in self.models:
            # If labels are provided, models return ArcFace logits (margin applied)
            # If labels are None, models return scaled cosine similarities
            out = model(x, labels)
            outputs.append(out)

        # Stack and average
        # Shape: (Batch, Num_Classes)
        avg_output = torch.stack(outputs).mean(dim=0)
        return avg_output


# -------------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------------
def run():
    # 1. Setup
    print("Initializing Pipeline...")
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    Config.EPOCHS = 8  # Reduced epochs to ensure completion within time limit
    Config.BATCH_SIZE = 32

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load DataLoaders (Original Data)
    print("Loading original datasets...")
    train_loader, val_loader, test_loader, label_encoder = get_loaders(
        load_cached_data=True
    )

    model_names = Config.MODEL_NAMES  # ['densenet121', 'resnet50_ibn_a']
    stage1_checkpoints = []

    # -------------------------------------------------------------------------
    # Stage 1: Train Individual Models on Original Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 1: Training on Original Data")
    print("=" * 40)

    for name in model_names:
        print(f"\nTraining Model: {name}")

        # Initialize Model
        model = get_model(name, pretrained=True).to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

        best_map = 0.0
        best_path = os.path.join(Config.CHECKPOINT_DIR, f"stage1_{name}_best.pth")

        for epoch in range(Config.EPOCHS):
            print(f"Epoch {epoch+1}/{Config.EPOCHS}")
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, scheduler=None
            )  # Scheduler step done manually if needed, or passed
            scheduler.step()

            # Validation
            val_loss, val_map = validate(model, val_loader, device, label_encoder)

            # Save Best
            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_path)
                print(f"New Best MAP: {best_map:.4f} -> Saved to {best_path}")

        stage1_checkpoints.append(best_path)

        # Free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Pseudo-Labeling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Generating Pseudo-Labels")
    print("=" * 40)

    # Load Stage 1 Models for Ensemble Inference
    stage1_models = []
    for name, path in zip(model_names, stage1_checkpoints):
        m = get_model(name, pretrained=False)
        m.load_state_dict(torch.load(path, map_location=device))
        stage1_models.append(m)

    # Generate combined dataframe
    combined_train_df = generate_pseudo_labels(
        stage1_models, device, label_encoder, load_cached_data=False
    )

    # Free memory
    del stage1_models
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Stage 2: Train on Combined Data (Fine-tuning)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("STAGE 2: Training on Combined Data (Original + Pseudo)")
    print("=" * 40)

    # Get new loaders with combined data
    train_loader_st2, val_loader_st2, _, _ = get_loaders(
        load_cached_data=True, extra_train_df=combined_train_df
    )

    stage2_models = []

    for name, stage1_path in zip(model_names, stage1_checkpoints):
        print(f"\nFine-tuning Model: {name}")

        # Initialize Model (Load Stage 1 weights as starting point)
        model = get_model(name, pretrained=False)
        model.load_state_dict(torch.load(stage1_path, map_location=device))
        model.to(device)

        # Optimizer (Lower LR for fine-tuning)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE * 0.5,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

        best_map = 0.0
        best_path = os.path.join(Config.CHECKPOINT_DIR, f"stage2_{name}_best.pth")

        for epoch in range(Config.EPOCHS):
            print(f"Epoch {epoch+1}/{Config.EPOCHS}")
            train_loss = train_one_epoch(
                model, train_loader_st2, optimizer, device, scheduler=None
            )
            scheduler.step()

            val_loss, val_map = validate(model, val_loader_st2, device, label_encoder)

            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_path)
                print(f"New Best MAP: {best_map:.4f} -> Saved to {best_path}")

        # Load best weights for final ensemble
        model.load_state_dict(torch.load(best_path, map_location=device))
        stage2_models.append(model)

    # -------------------------------------------------------------------------
    # Final Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Final Evaluation")
    print("=" * 40)

    final_ensemble = EnsembleModel(stage2_models).to(device)
    final_ensemble.eval()

    # 1. Calculate Final Metric
    # We use engine.validate but with the ensemble model
    # Note: validate() uses TTA internally which is good
    val_loss, final_map5 = validate(final_ensemble, val_loader, device, label_encoder)

    print(f"Final Validation Metric: {final_map5}")

    # 2. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # We need to manually run inference on val set to get per-image predictions for analysis
    # because validate() aggregates them.
    val_df = pd.read_csv(Config.VAL_CSV)

    # Collect metadata for correlation
    # We need to map Image filename to its metadata (e.g. class count)
    train_df_orig = pd.read_csv(Config.TRAIN_CSV)
    class_counts = train_df_orig["Id"].value_counts().to_dict()

    # Store results
    analysis_data = []

    with torch.no_grad():
        for images, labels, image_names in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA Inference
            logits_orig = final_ensemble(images, labels=None)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = final_ensemble(images_flip, labels=None)
            logits = (logits_orig + logits_flip) / 2.0

            _, topk_indices = torch.topk(logits, k=5, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(image_names)):
                img_name = image_names[i]
                true_label_idx = labels_np[i]
                pred_indices = topk_indices[i]

                # Decode
                true_label = label_encoder.inverse_transform([true_label_idx])[0]
                pred_labels = label_encoder.inverse_transform(pred_indices).tolist()

                # Calculate AP
                ap = map_per_image(true_label, pred_labels)

                # Get Metadata
                # Samples in training set for this class
                n_samples = class_counts.get(true_label, 0)

                analysis_data.append(
                    {
                        "Image": img_name,
                        "AP": ap,
                        "SamplesInTrain": n_samples,
                        "IsNewWhale": 1 if true_label == "new_whale" else 0,
                    }
                )

    df_analysis = pd.DataFrame(analysis_data)

    # Calculate Correlations
    corr_samples = df_analysis["AP"].corr(df_analysis["SamplesInTrain"])
    corr_new_whale = df_analysis["AP"].corr(df_analysis["IsNewWhale"])

    print("Failure Analysis Results:")
    print(f"Correlation (AP vs Samples in Train): {corr_samples:.4f}")
    print(f"Correlation (AP vs Is New Whale): {corr_new_whale:.4f}")
    print(
        f"Average AP for 'new_whale': {df_analysis[df_analysis['IsNewWhale']==1]['AP'].mean():.4f}"
    )
    print(
        f"Average AP for known whales: {df_analysis[df_analysis['IsNewWhale']==0]['AP'].mean():.4f}"
    )

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6706947524020695

    if final_map5 > THRESHOLD:
        print(
            f"\nValidation Metric ({final_map5}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Use engine.inference with the ensemble model
        # It handles TTA and saving to CSV
        inference(final_ensemble, test_loader, device, label_encoder)

    else:
        print(
            f"\nValidation Metric ({final_map5}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    run()
