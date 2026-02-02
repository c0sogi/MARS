import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from library import config, utils, data, model, engine


def train_production_ensemble(best_epoch, lr_milestones, load_cached_data=True):
    """
    Executes Phase 2: Production (Full-Fit Seed Ensemble).
    Trains multiple independent models on the full dataset (Train + Val) for exactly
    'best_epoch' epochs using the schedule derived from calibration.
    Generates the final submission by averaging TTA predictions from all models.

    Args:
        best_epoch (int): The optimal number of training epochs determined in Phase 1.
        lr_milestones (list): List of epochs to decay the learning rate.
        load_cached_data (bool): Whether to use cached numpy arrays for data.

    Returns:
        None
    """
    print(f"Starting Phase 2: Production (Full-Fit Seed Ensemble)")
    print(
        f"Training {config.NUM_ENSEMBLE_MODELS} models for {best_epoch} epochs on full dataset."
    )
    print(f"LR Milestones: {lr_milestones}")

    device = config.DEVICE
    model_paths = []

    # 1. Train Ensemble Models
    for i in range(config.NUM_ENSEMBLE_MODELS):
        # Set a unique seed for each model to ensure diversity in initialization and batch shuffling
        current_seed = config.RANDOM_SEED + i
        utils.seed_everything(current_seed)

        print(
            f"\n--- Training Ensemble Model {i+1}/{config.NUM_ENSEMBLE_MODELS} (Seed {current_seed}) ---"
        )

        # Load Full Dataset (Train + Val combined)
        # We re-create dataloaders here to ensure the shuffling respects the new seed
        train_loader, _, test_loader = data.get_dataloaders(
            load_cached_data=load_cached_data, full_train=True
        )

        # Initialize Model
        net = model.IcebergResNet18()
        net.to(device)

        # Initialize Optimizer
        optimizer = optim.AdamW(
            net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        # Initialize Scheduler (MultiStepLR based on calibration)
        # gamma=0.1 matches the factor used in ReduceLROnPlateau during calibration
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

        # Training Loop
        for epoch in range(best_epoch):
            # engine.train_one_epoch prints the loss
            avg_loss = engine.train_one_epoch(
                net, train_loader, optimizer, device, epoch + 1
            )

            # Step the scheduler
            scheduler.step()

            # Optional: Print LR if it changed (checking milestones)
            if (epoch + 1) in lr_milestones:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"LR reduced at epoch {epoch + 1} to {current_lr}")

        # Save Model Checkpoint
        save_path = os.path.join(config.WORKING_DIR, f"model_ensemble_{i}.pth")
        utils.save_checkpoint(net.state_dict(), save_path)
        model_paths.append(save_path)
        print(f"Model {i+1} saved to {save_path}")

        # Cleanup
        del net, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()

    # 2. Generate Predictions (Ensemble Inference)
    print("\n--- Generating Ensemble Predictions ---")

    # We need the test loader one last time (order is deterministic)
    _, _, test_loader = data.get_dataloaders(
        load_cached_data=load_cached_data, full_train=True
    )

    ensemble_probs = []
    test_ids = None

    for i, path in enumerate(model_paths):
        print(f"Predicting with Model {i+1}...")

        # Load Model
        net = model.IcebergResNet18()
        net.to(device)
        utils.load_checkpoint(path, net, device=device)

        # Predict with TTA
        ids, probs = engine.predict_tta(net, test_loader, device)

        ensemble_probs.append(probs)

        if test_ids is None:
            test_ids = ids
        else:
            # Sanity check to ensure ID order is preserved
            if test_ids != ids:
                raise ValueError("Test ID mismatch between ensemble models.")

        del net
        torch.cuda.empty_cache()

    # 3. Average Predictions
    print("Averaging predictions...")
    ensemble_probs = np.array(ensemble_probs)  # Shape: (n_models, n_samples)
    avg_probs = np.mean(ensemble_probs, axis=0)  # Shape: (n_samples,)

    # 4. Save Submission
    engine.save_submission(test_ids, avg_probs.tolist())
    print("Phase 2 Complete.")
