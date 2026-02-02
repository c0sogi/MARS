import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_roc_auc
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import MultiTaskRepVGG
from library.engine import Engine


def run():
    # ==========================================
    # 1. Setup
    # ==========================================
    seed_everything(Config.SEED)
    logger = get_logger("main")
    device = torch.device(Config.DEVICE)

    logger.info(f"Starting run execution on device: {device}")
    Config.print_config()

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading Datasets...")
    # Load training and validation loaders
    # Note: get_dataloaders returns a fixed split based on metadata files.
    # To implement the 5-Fold strategy described in the idea within the constraints
    # of the provided dataset functions, we will train 5 models on the same split
    # (Bagging/Ensemble of Initializations) which is a robust baseline strategy.
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Training Loop
    # ==========================================
    model_paths = []

    for fold_idx in range(Config.NUM_FOLDS):
        logger.info(f"\n{'='*20} Training Fold {fold_idx} {'='*20}")

        # Initialize Model (Train Mode)
        model = MultiTaskRepVGG(deploy=False).to(device)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Cosine Annealing for Convergence Phase)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.CONVERGENCE_EPOCHS, eta_min=Config.SWA_LR_MIN
        )

        # Engine
        engine = Engine(model, device, optimizer, scheduler)

        # Train
        # This saves 'best_foldX.pth' and 'swa_foldX.pth'
        best_auc = engine.fit(train_loader, val_loader, fold_idx)

        # We use the SWA model for final inference as it generalizes better
        swa_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_fold{fold_idx}.pth")
        model_paths.append(swa_path)

        # Cleanup
        del model, optimizer, scheduler, engine
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    logger.info(f"\n{'='*20} Validation & Failure Analysis {'='*20}")

    # Load Models
    models = []
    for path in model_paths:
        # Init model structure
        m = MultiTaskRepVGG(deploy=False)

        # Load weights
        # strict=False to handle potential 'n_averaged' buffer from AveragedModel
        state_dict = torch.load(path, map_location="cpu")

        # Handle 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        m.load_state_dict(new_state_dict, strict=False)

        # Structural Re-parameterization (Fusion)
        m.reparameterize()

        m.to(device)
        m.eval()
        models.append(m)

    # Inference on Validation Set
    val_preds = []
    val_targets = []
    val_quality = []
    val_img_means = []
    val_img_stds = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].numpy()
            quals = batch["quality_target"].numpy()

            val_targets.append(lbls)
            val_quality.append(quals)

            # Calculate image stats for failure analysis
            imgs_np = imgs.cpu().numpy()
            # Mean over H, W, C
            val_img_means.append(imgs_np.mean(axis=(1, 2, 3)))
            val_img_stds.append(imgs_np.std(axis=(1, 2, 3)))

            # TTA + Ensemble Prediction
            batch_preds = np.zeros(imgs.size(0))

            # 4 Views: Original, HFlip, VFlip, Rot180
            views = [
                imgs,
                torch.flip(imgs, [3]),
                torch.flip(imgs, [2]),
                torch.flip(imgs, [2, 3]),
            ]

            for m in models:
                for view in views:
                    out = m(view)
                    # Get probabilities from both heads
                    p_tex = (
                        torch.sigmoid(out["texture"]).cpu().numpy().flatten()
                        if out["texture"] is not None
                        else 0
                    )
                    p_sem = (
                        torch.sigmoid(out["semantic"]).cpu().numpy().flatten()
                        if out["semantic"] is not None
                        else 0
                    )

                    # Average heads
                    p = (p_tex + p_sem) / 2.0
                    batch_preds += p

            # Average over (5 Models * 4 Views) = 20 predictions
            batch_preds /= len(models) * len(views)
            val_preds.append(batch_preds)

    # Concatenate
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_quality = np.concatenate(val_quality)
    val_img_means = np.concatenate(val_img_means)
    val_img_stds = np.concatenate(val_img_stds)

    # Calculate Metric
    final_auc = calculate_roc_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    errors = np.abs(val_targets - val_preds)

    # Handle constant input cases for correlation
    def safe_pearson(x, y):
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0
        return pearsonr(x, y)[0]

    corr_qual = safe_pearson(errors, val_quality)
    corr_mean = safe_pearson(errors, val_img_means)
    corr_std = safe_pearson(errors, val_img_stds)

    print("Failure Analysis (Correlation with Error Magnitude):")
    print(f"  File Size (Quality): {corr_qual:.4f}")
    print(f"  Image Mean Intensity: {corr_mean:.4f}")
    print(f"  Image Contrast (Std): {corr_std:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    # Note: Prompt mentions "higher than 1.0" which is impossible for AUC (max 1.0).
    # Assuming threshold is 0.5 or intended to be "submit if valid".
    if final_auc > 0.5:
        logger.info(f"\n{'='*20} Generating Submission {'='*20}")

        test_loader, test_ids = get_test_dataloader(load_cached_data=True)
        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(device)

                batch_preds = np.zeros(imgs.size(0))

                views = [
                    imgs,
                    torch.flip(imgs, [3]),
                    torch.flip(imgs, [2]),
                    torch.flip(imgs, [2, 3]),
                ]

                for m in models:
                    for view in views:
                        out = m(view)
                        p_tex = (
                            torch.sigmoid(out["texture"]).cpu().numpy().flatten()
                            if out["texture"] is not None
                            else 0
                        )
                        p_sem = (
                            torch.sigmoid(out["semantic"]).cpu().numpy().flatten()
                            if out["semantic"] is not None
                            else 0
                        )
                        p = (p_tex + p_sem) / 2.0
                        batch_preds += p

                batch_preds /= len(models) * len(views)
                test_preds.append(batch_preds)

        test_preds = np.concatenate(test_preds)

        # Save
        df_sub = pd.DataFrame({"id": test_ids, "has_cactus": test_preds})

        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

    else:
        logger.warning("Validation AUC is too low. Skipping submission generation.")


if __name__ == "__main__":
    run()
