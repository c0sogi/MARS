import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import CFG
from library.dataset import WhaleDataset, get_transforms, process_data
from library.modeling import WhaleModel
from library.engine import train_fn, valid_fn, extract_embeddings
from library.utils import seed_everything, map_at_5, AverageMeter
from library.inference import generate_submission

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# We adjust epochs to ensure execution finishes within the 1-hour limit
# while still attempting to converge reasonably well.
CFG.epochs_p1 = 3  # Phase 1: Warmup at 256px
CFG.epochs_p2 = 2  # Phase 2: Finetune at 384px
CFG.batch_size = 32  # A100 can handle this for B5/V2-M
CFG.working_dir = "./working/idea_8_run"
os.makedirs(CFG.working_dir, exist_ok=True)


def train_model_pipeline(model_name, train_df, val_df, num_classes, save_name):
    """
    Trains a single model using the 2-phase progressive resizing strategy.
    """
    print(f"\n{'='*40}")
    print(f"Starting Pipeline for: {model_name}")
    print(f"{'='*40}")

    device = CFG.device
    best_score = 0.0
    best_path = os.path.join(CFG.working_dir, f"{save_name}_best.pth")

    # Initialize Model
    print(f"Initializing {model_name}...")
    model = WhaleModel(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # ---------------------------------------------------------
    # Phase 1: Resolution 256x256
    # ---------------------------------------------------------
    print(f"\n--- Phase 1: Training @ {CFG.image_size_p1}x{CFG.image_size_p1} ---")

    train_dataset_p1 = WhaleDataset(
        train_df,
        transform=get_transforms(data="train", image_size=CFG.image_size_p1),
        label_col="label_idx",
        id_col="Id",
    )
    train_loader_p1 = DataLoader(
        train_dataset_p1,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # We don't validate strictly in Phase 1 to save time, or just at the end.
    # Let's train for epochs_p1
    for epoch in range(CFG.epochs_p1):
        avg_loss = train_fn(
            train_loader_p1,
            model,
            nn.CrossEntropyLoss(),
            optimizer,
            device,
            epoch=epoch,
        )
        print(f"Phase 1 Epoch {epoch+1}/{CFG.epochs_p1} - Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------
    # Phase 2: Resolution 384x384
    # ---------------------------------------------------------
    print(f"\n--- Phase 2: Training @ {CFG.image_size_p2}x{CFG.image_size_p2} ---")

    # Re-initialize dataset/loader with higher resolution
    train_dataset_p2 = WhaleDataset(
        train_df,
        transform=get_transforms(data="train", image_size=CFG.image_size_p2),
        label_col="label_idx",
        id_col="Id",
    )
    train_loader_p2 = DataLoader(
        train_dataset_p2,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Loaders (Gallery=Train, Query=Val)
    # Note: For validation, we use the training set as the gallery.
    # We need a gallery loader that iterates the training set deterministically (no shuffle).
    gallery_dataset = WhaleDataset(
        train_df,
        transform=get_transforms(data="valid", image_size=CFG.image_size_p2),
        id_col="Id",
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    val_dataset = WhaleDataset(
        val_df,
        transform=get_transforms(data="valid", image_size=CFG.image_size_p2),
        id_col="Id",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Scheduler for Phase 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs_p2, eta_min=CFG.scheduler_params["eta_min"]
    )

    for epoch in range(CFG.epochs_p2):
        print(f"Phase 2 Epoch {epoch+1}/{CFG.epochs_p2}")

        # Train
        avg_loss = train_fn(
            train_loader_p2,
            model,
            nn.CrossEntropyLoss(),
            optimizer,
            device,
            epoch=epoch,
        )
        scheduler.step()

        # Validate
        val_score = valid_fn(gallery_loader, val_loader, model, device)
        print(f"Epoch {epoch+1} Validation MAP@5: {val_score:.5f}")

        # Save Best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)
            print(f"Saved Best Model: {best_score:.5f}")

    # Load best weights before returning
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    return model, best_path


def get_ensemble_validation_metrics(
    model_a_path, model_b_path, train_df, val_df, num_classes
):
    """
    Computes the MAP@5 for the ensemble of Model A and Model B on the validation set.
    """
    print("\nComputing Ensemble Validation Metrics...")
    device = CFG.device

    # Transforms
    transforms = get_transforms(data="valid", image_size=CFG.image_size_p2)

    # Datasets
    gallery_dataset = WhaleDataset(train_df, transform=transforms, id_col="Id")
    query_dataset = WhaleDataset(val_df, transform=transforms, id_col="Id")

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # --- Model A ---
    print(f"Extracting features for Model A: {CFG.model_a_name}")
    model_a = WhaleModel(CFG.model_a_name, num_classes=num_classes, pretrained=False)
    model_a.load_state_dict(torch.load(model_a_path, map_location=device))
    model_a.to(device)
    model_a.eval()

    gal_emb_a, gal_ids = extract_embeddings(
        gallery_loader, model_a, device, tta=CFG.tta_flips
    )
    qry_emb_a, qry_ids = extract_embeddings(
        query_loader, model_a, device, tta=CFG.tta_flips
    )

    del model_a
    torch.cuda.empty_cache()

    # --- Model B ---
    print(f"Extracting features for Model B: {CFG.model_b_name}")
    model_b = WhaleModel(CFG.model_b_name, num_classes=num_classes, pretrained=False)
    model_b.load_state_dict(torch.load(model_b_path, map_location=device))
    model_b.to(device)
    model_b.eval()

    gal_emb_b, _ = extract_embeddings(
        gallery_loader, model_b, device, tta=CFG.tta_flips
    )
    qry_emb_b, _ = extract_embeddings(query_loader, model_b, device, tta=CFG.tta_flips)

    del model_b
    torch.cuda.empty_cache()

    # --- Fusion ---
    print("Fusing Similarities...")
    sim_a = np.dot(qry_emb_a, gal_emb_a.T)
    sim_b = np.dot(qry_emb_b, gal_emb_b.T)
    sim_final = 0.5 * sim_a + 0.5 * sim_b

    # --- Metric Calculation ---
    predictions = []
    # Store detailed results for failure analysis
    analysis_data = []

    train_id_counts = train_df["Id"].value_counts().to_dict()

    for i in range(len(qry_ids)):
        scores = sim_final[i]
        sorted_indices = np.argsort(scores)[::-1]

        pred_ids = []
        seen = set()

        # Get top 1 info for analysis
        top_idx = sorted_indices[0]
        top_id = gal_ids[top_idx]
        top_score = scores[top_idx]

        for idx in sorted_indices:
            pid = gal_ids[idx]
            if pid not in seen:
                pred_ids.append(pid)
                seen.add(pid)
            if len(pred_ids) == 5:
                break

        predictions.append(pred_ids)

        # Calculate per-instance MAP (1/rank)
        true_id = qry_ids[i]
        try:
            rank = pred_ids.index(true_id) + 1
            inst_map = 1.0 / rank
        except ValueError:
            inst_map = 0.0

        analysis_data.append(
            {
                "True_Id": true_id,
                "Pred_Id": top_id,
                "Score": top_score,
                "MAP": inst_map,
                "Train_Freq": train_id_counts.get(true_id, 0),
            }
        )

    final_score = map_at_5(qry_ids, predictions)
    return final_score, pd.DataFrame(analysis_data)


def perform_failure_analysis(analysis_df):
    """
    Correlates error magnitude (1 - MAP) with confidence and class frequency.
    """
    print("\n=== Failure Analysis ===")
    analysis_df["Error"] = 1.0 - analysis_df["MAP"]

    # Correlation with Confidence (Score)
    corr_conf = analysis_df["Error"].corr(analysis_df["Score"])
    print(f"Correlation (Error vs Confidence): {corr_conf:.4f}")

    # Correlation with Train Frequency
    corr_freq = analysis_df["Error"].corr(analysis_df["Train_Freq"])
    print(f"Correlation (Error vs Class Frequency): {corr_freq:.4f}")

    print(
        "Average Confidence for Correct Predictions:",
        analysis_df[analysis_df["MAP"] == 1.0]["Score"].mean(),
    )
    print(
        "Average Confidence for Incorrect Predictions:",
        analysis_df[analysis_df["MAP"] < 1.0]["Score"].mean(),
    )


def main():
    seed_everything(CFG.seed)

    # 1. Load Data
    train_df, val_df, test_df, num_classes = process_data(load_cached_data=True)

    # 2. Train Model A
    _, model_a_path = train_model_pipeline(
        CFG.model_a_name, train_df, val_df, num_classes, "model_a"
    )

    # 3. Train Model B
    _, model_b_path = train_model_pipeline(
        CFG.model_b_name, train_df, val_df, num_classes, "model_b"
    )

    # 4. Ensemble Validation
    final_metric, analysis_df = get_ensemble_validation_metrics(
        model_a_path, model_b_path, train_df, val_df, num_classes
    )

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(analysis_df)

    # 6. Submission
    # Threshold from task description
    THRESHOLD = 0.846985

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        generate_submission(
            model_a_path, model_b_path, output_path="./submission/submission.csv"
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
