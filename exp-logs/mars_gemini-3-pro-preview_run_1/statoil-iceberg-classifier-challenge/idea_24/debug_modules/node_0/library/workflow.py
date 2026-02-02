import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from library import config, utils, sam, data_factory, model_factory, engine


# ==========================================
# Custom Loss
# ==========================================
class SmoothBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with Label Smoothing.
    Formula: target_smooth = target * (1 - epsilon) + 0.5 * epsilon
    """

    def __init__(self, smoothing=0.05):
        super(SmoothBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # targets shape: (B, 1)
        with torch.no_grad():
            smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smooth_targets)


# ==========================================
# Phase 1: Calibration
# ==========================================
def run_phase_1_calibration(n_splits=5):
    """
    Runs Stratified K-Fold CV to determine the optimal convergence epoch.
    Returns:
        int: Average best epoch across folds.
    """
    logger = utils.get_logger("Phase1_Calibration")
    logger.info("Starting Phase 1: Calibration (Epoch Selection)")

    # Load all data
    t_imgs, t_angs, t_lbls, t_ids, _, _, _ = data_factory.get_all_data(
        load_cached_data=True
    )

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED
    )

    best_epochs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(t_imgs, t_lbls)):
        logger.info(f"--- Fold {fold + 1}/{n_splits} ---")

        # Data Loaders
        train_loader, val_loader, _ = data_factory.get_dataloaders(
            train_idxs=train_idx, val_idxs=val_idx
        )

        # Model Setup
        model = model_factory.get_model()

        # Optimizer: SAM wrapping AdamW
        base_optimizer = torch.optim.AdamW
        optimizer = sam.SAM(
            model.parameters(),
            base_optimizer,
            rho=config.SAM_RHO,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.base_optimizer,
            mode=config.SCHEDULER_MODE,
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        # Loss
        criterion = SmoothBCEWithLogitsLoss(smoothing=config.LABEL_SMOOTHING)

        # Engine
        trainer = engine.Engine(
            model=model,
            device=utils.get_device(),
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
        )

        # Training Loop
        best_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, config.MAX_EPOCHS_PHASE_1 + 1):
            train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch)

            # Validate with TTA
            val_loss, val_acc, _, _ = trainer.validate_tta(val_loader)

            # Step Scheduler
            scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch}: Train Loss={train_loss:.6f}, Train Acc={train_acc:.6f}, "
                f"Val Loss={val_loss:.6f}, Val Acc={val_acc:.6f}"
            )

            # Checkpoint & Early Stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}."
                )
                break

        best_epochs.append(best_epoch)
        logger.info(f"Fold {fold + 1} Best Epoch: {best_epoch} (Loss: {best_loss:.6f})")

    avg_best_epoch = int(np.mean(best_epochs))
    logger.info(
        f"Phase 1 Complete. Optimal Convergence Epoch (E_conv): {avg_best_epoch}"
    )

    return avg_best_epoch


# ==========================================
# Phase 2: Production
# ==========================================
def run_phase_2_production(optimal_epochs, n_models=5):
    """
    Trains an ensemble of models on the full dataset using SAM -> SWA trajectory.
    """
    logger = utils.get_logger("Phase2_Production")
    logger.info(
        f"Starting Phase 2: Production (Full-Fit SAM-SWA) with E_conv={optimal_epochs}"
    )

    # Load all data indices
    t_imgs, _, _, _, _, _, _ = data_factory.get_all_data(load_cached_data=True)
    all_indices = np.arange(len(t_imgs))

    # We use a dummy val_idx because get_dataloaders requires it, but we won't use val_loader
    train_loader, _, _ = data_factory.get_dataloaders(
        train_idxs=all_indices, val_idxs=all_indices[:2]
    )

    device = utils.get_device()

    for i in range(n_models):
        logger.info(f"--- Training Ensemble Model {i+1}/{n_models} ---")

        # Seed for diversity (though SAM provides some, explicit seeding is safer)
        current_seed = config.RANDOM_SEED + i
        utils.seed_everything(current_seed)

        # Initialize
        model = model_factory.get_model()
        base_optimizer = torch.optim.AdamW
        optimizer = sam.SAM(
            model.parameters(),
            base_optimizer,
            rho=config.SAM_RHO,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler (Re-initialized per model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.base_optimizer,
            mode=config.SCHEDULER_MODE,
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )

        criterion = SmoothBCEWithLogitsLoss(smoothing=config.LABEL_SMOOTHING)

        trainer = engine.Engine(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
        )

        # Stage A: SAM Convergence
        logger.info(f"Stage A: Training for {optimal_epochs} epochs...")
        for epoch in range(1, optimal_epochs + 1):
            t_loss, t_acc = trainer.train_one_epoch(train_loader, epoch)
            # Step scheduler based on training loss since we have no validation set
            # Or assume the dynamics from Phase 1 hold.
            # Given ReduceLROnPlateau usually needs val loss, we use training loss here as proxy
            # to allow LR decay if training stalls, though Phase 1 determined the trajectory.
            scheduler.step(t_loss)
            if epoch % 5 == 0 or epoch == optimal_epochs:
                logger.info(f"Epoch {epoch}: Loss={t_loss:.6f}, Acc={t_acc:.6f}")

        # Stage B: SWA Transition
        logger.info(
            f"Stage B: Transitioning to SWA for {config.SWA_DURATION_EPOCHS} epochs..."
        )

        # Initialize SWA Handler
        swa_handler = engine.SWAHandler(model, device)

        # Set SWA Learning Rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.SWA_LR

        for swa_epoch in range(1, config.SWA_DURATION_EPOCHS + 1):
            # Continue training with SAM (SAM-SWA)
            t_loss, t_acc = trainer.train_one_epoch(
                train_loader, optimal_epochs + swa_epoch
            )

            # Update SWA Model
            swa_handler.update(model)
            logger.info(f"SWA Epoch {swa_epoch}: Loss={t_loss:.6f}")

        # Finalize SWA (Update BN)
        logger.info("Updating SWA Batch Normalization statistics...")
        swa_handler.update_bn(train_loader)

        # Save Model
        save_path = os.path.join(config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
        torch.save(swa_handler.get_model().state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")


# ==========================================
# Submission
# ==========================================
def generate_submission(n_models=5):
    """
    Generates submission by averaging TTA predictions from all ensemble models.
    """
    logger = utils.get_logger("Submission")
    logger.info("Generating Submission...")

    # Load Test Data
    _, _, test_loader = data_factory.get_dataloaders(load_cached_data=True)

    device = utils.get_device()
    final_preds = None
    test_ids = None

    for i in range(n_models):
        model_path = os.path.join(config.CHECKPOINT_DIR, f"swa_model_{i}.pth")
        logger.info(f"Processing model: {model_path}")

        # Load Model
        model = model_factory.get_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Engine for prediction
        predictor = engine.Engine(model, device)

        # Predict
        ids, probs = predictor.predict_test_tta(test_loader)

        if final_preds is None:
            final_preds = probs
            test_ids = ids
        else:
            final_preds += probs

    # Average
    final_preds /= n_models

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": final_preds})

    # Save
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")
    logger.info(f"Head:\n{df_sub.head()}")
