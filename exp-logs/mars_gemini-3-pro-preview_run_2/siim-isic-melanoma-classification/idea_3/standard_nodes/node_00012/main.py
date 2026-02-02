import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_loaders
from library.model import ContextGatedEfficientNet
from library.engine import train_one_epoch, evaluate, predict


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for fast baseline execution
    # 10 epochs is sufficient for convergence on this dataset size with A100
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 128

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # load_cached_data=True will use ./working/idea_3/meta_*.npy if available
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Determine metadata dimension from a batch
    dummy_batch = next(iter(train_loader))
    # Batch structure: ((images, meta), targets)
    meta_dim = dummy_batch[0][1].shape[1]
    print(f"Metadata dimension detected: {meta_dim}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing Model...")
    model = ContextGatedEfficientNet(meta_dim=meta_dim, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # ==========================================
    # 4. Optimization Setup
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Schedulers: Linear Warmup -> Cosine Annealing
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[Config.WARMUP_EPOCHS],
    )

    # Loss with Class Weighting
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train AUC: {train_auc:.5f} | Val AUC: {val_auc:.5f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # 6. Final Evaluation & Failure Analysis
    # ==========================================
    print("\nRunning Final Evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    # Calculate final metric on full validation set
    _, final_val_auc = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (images, meta), targets in val_loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Calculate errors
    errors = np.abs(all_targets - all_preds)

    # Load raw metadata for correlation
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Ensure lengths match
    if len(df_val) == len(errors):
        df_val["error"] = errors

        # Preprocess for correlation
        # Fill NaNs with mean (numerical) or 'unknown' (categorical)
        df_val["age_approx"] = df_val["age_approx"].fillna(df_val["age_approx"].mean())
        df_val["sex"] = df_val["sex"].fillna("unknown")
        df_val["anatom_site_general_challenge"] = df_val[
            "anatom_site_general_challenge"
        ].fillna("unknown")

        # Encode categorical variables
        df_val["sex_code"] = df_val["sex"].astype("category").cat.codes
        df_val["site_code"] = (
            df_val["anatom_site_general_challenge"].astype("category").cat.codes
        )

        # Calculate correlations
        corr_age = df_val["age_approx"].corr(df_val["error"])
        corr_sex = df_val["sex_code"].corr(df_val["error"])
        corr_site = df_val["site_code"].corr(df_val["error"])

        print("Correlation of Error with Features:")
        print(f"  Age: {corr_age:.4f}")
        print(f"  Sex: {corr_sex:.4f}")
        print(f"  Anatomical Site: {corr_site:.4f}")
    else:
        print("Warning: Metadata length mismatch. Skipping correlation analysis.")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.8850620049856426

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        image_names, preds = predict(model, test_loader, device)

        submission_df = pd.DataFrame({"image_name": image_names, "target": preds})

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation AUC ({final_val_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
