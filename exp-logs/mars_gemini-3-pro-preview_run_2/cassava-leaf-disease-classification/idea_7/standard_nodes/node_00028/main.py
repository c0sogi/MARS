import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, accuracy
from library.dataset import get_loaders
from library.model import get_model
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import predict_fn


def main():
    # 1. Configuration and Setup
    config = Config()

    # Update working directory to avoid conflicts with previous runs
    config.working_dir = "./working/idea_optimized"
    os.makedirs(config.working_dir, exist_ok=True)

    # Restore full training schedule (12+8 epochs) to prioritize convergence.
    # Cite solution_lesson_node_00022: "Never sacrifice convergence for cross-validation..."
    # Cite solution_lesson_node_00027: Progressive Resolution Fine-Tuning reached 90.76% with full schedule.
    # config.phase1_epochs and config.phase2_epochs use defaults (12 and 8) from Config class.

    seed_everything(config.seed)
    logger = get_logger(os.path.join(config.working_dir, "train.log"))

    logger.info("Starting Orchestration Script")
    logger.info(f"Configuration: {config}")

    # Container for Out-Of-Fold predictions
    # Structure: {'image_id': [], 'target': [], 'pred': [], 'prob': []}
    oof_results = defaultdict(list)

    # 2. Training Loop (Single Fold to maximize compute per model)
    # Cite solution_lesson_node_00022: Prioritizing Training Duration Over Validation Robustness
    # We train only the first fold but for the full duration (20 epochs).
    for fold_idx in range(1):
        logger.info(f"\n{'='*20} FOLD {fold_idx}/0 {'='*20}")

        # --- Phase 1: Coarse Learning (224x224) ---
        logger.info(
            f"--- Phase 1: Coarse Learning ({config.phase1_image_size}x{config.phase1_image_size}) ---"
        )

        # Get Loaders for Phase 1
        train_loader, val_loader, _ = get_loaders(config, phase=1, fold_idx=fold_idx)

        # Initialize Model
        model = get_model(config)
        model.to(config.device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.phase1_epochs, eta_min=config.min_lr
        )

        # Training Loop Phase 1
        for epoch in range(config.phase1_epochs):
            loss = train_one_epoch(
                model, train_loader, optimizer, config.device, epoch, config
            )
            val_loss, val_acc = valid_one_epoch(
                model, val_loader, config.device, config
            )
            scheduler.step()

            logger.info(
                f"[Fold {fold_idx}][P1][Ep {epoch+1}/{config.phase1_epochs}] "
                f"Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

        # --- Phase 2: Fine Tuning (384x384) ---
        logger.info(
            f"--- Phase 2: Fine Tuning ({config.phase2_image_size}x{config.phase2_image_size}) ---"
        )

        # Get Loaders for Phase 2
        train_loader, val_loader, _ = get_loaders(config, phase=2, fold_idx=fold_idx)

        # We keep the model weights from Phase 1 but re-initialize optimizer for stability in Phase 2
        # (Alternatively, one could load a saved checkpoint, but the model object in memory preserves weights)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr * 0.5,  # Slightly lower LR for fine-tuning
            weight_decay=config.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.phase2_epochs, eta_min=config.min_lr
        )

        best_acc = 0.0
        best_model_path = os.path.join(config.working_dir, f"fold_{fold_idx}_best.pth")

        # Training Loop Phase 2
        for epoch in range(config.phase2_epochs):
            loss = train_one_epoch(
                model, train_loader, optimizer, config.device, epoch, config
            )
            val_loss, val_acc = valid_one_epoch(
                model, val_loader, config.device, config
            )
            scheduler.step()

            logger.info(
                f"[Fold {fold_idx}][P2][Ep {epoch+1}/{config.phase2_epochs}] "
                f"Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                logger.info(
                    f"Saved Best Model for Fold {fold_idx} with Acc: {best_acc:.2f}%"
                )

        # --- Generate OOF Predictions for this Fold ---
        logger.info(f"Generating OOF predictions for Fold {fold_idx}...")

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
        model.eval()

        # We need image_ids to map back to metadata for failure analysis
        # The val_loader dataset has the dataframe
        val_df = val_loader.dataset.df
        val_image_ids = val_df["image_id"].values

        fold_preds = []
        fold_targets = []
        fold_probs = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(config.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                # Get max prob and predicted class
                max_probs, preds = torch.max(probs, dim=1)

                fold_preds.extend(preds.cpu().numpy())
                fold_targets.extend(targets.numpy())
                fold_probs.extend(max_probs.cpu().numpy())

        oof_results["image_id"].extend(val_image_ids)
        oof_results["target"].extend(fold_targets)
        oof_results["pred"].extend(fold_preds)
        oof_results["prob"].extend(fold_probs)

        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 3. Validation Assessment & Failure Analysis
    logger.info("\n" + "=" * 40)
    logger.info("VALIDATION ASSESSMENT & FAILURE ANALYSIS")
    logger.info("=" * 40)

    # Create DataFrame from OOF results
    df_oof = pd.DataFrame(oof_results)

    # Calculate Overall Accuracy
    correct = (df_oof["target"] == df_oof["pred"]).sum()
    total = len(df_oof)
    final_metric = correct / total

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Load original train metadata to get file info
    df_meta = pd.read_csv(config.train_metadata_path)
    # We also need to check val.csv because the split puts some there
    df_val_meta = pd.read_csv(config.val_metadata_path)
    df_full_meta = pd.concat([df_meta, df_val_meta]).drop_duplicates(
        subset=["image_id"]
    )

    # Merge OOF results with metadata
    df_analysis = df_oof.merge(
        df_full_meta[["image_id", "file_path"]], on="image_id", how="left"
    )

    # Helper to get file size/dimensions (simplified, assuming we can't read all images now)
    # We will use the 'file_path' to get file size
    def get_file_info(rel_path):
        full_path = os.path.join(config.input_dir, rel_path)
        try:
            size = os.path.getsize(full_path)
            return size
        except:
            return 0

    df_analysis["file_size"] = df_analysis["file_path"].apply(get_file_info)

    # Calculate Error (1 if wrong, 0 if correct)
    df_analysis["error"] = (df_analysis["target"] != df_analysis["pred"]).astype(int)

    # Correlation Analysis
    # We check correlation between Error and File Size
    corr_size = df_analysis["error"].corr(df_analysis["file_size"])

    logger.info("Failure Analysis Report:")
    logger.info(f"Total Samples: {total}")
    logger.info(f"Total Errors: {df_analysis['error'].sum()}")
    logger.info(f"Correlation (Error vs File Size): {corr_size:.4f}")

    # Check for systematic error in specific classes
    class_acc = df_analysis.groupby("target").apply(
        lambda x: (x["target"] == x["pred"]).mean()
    )
    logger.info(f"Class-wise Accuracy:\n{class_acc}")

    # 4. Submission Logic
    # Threshold updated to 0.9076 as per requirements
    THRESHOLD = 0.9076

    if final_metric > THRESHOLD:
        logger.info(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Proceeding to Submission..."
        )

        # Free up memory before inference
        del df_oof, df_analysis, df_meta
        torch.cuda.empty_cache()

        # Ensure inference only loads the single trained model
        config.n_folds = 1

        # Run Inference
        predict_fn(config)
    else:
        logger.info(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
