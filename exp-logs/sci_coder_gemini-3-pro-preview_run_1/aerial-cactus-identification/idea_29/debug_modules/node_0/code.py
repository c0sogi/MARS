import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    calculate_trust_score,
    reparameterize_repvgg,
)
from library.data import get_dataloaders, get_test_dataloader
from library.models import CactusRepVGG, CactusResNet, TrustRouter
from library.engine import (
    train_expert_one_epoch,
    validate_expert,
    train_router_epoch,
    validate_router,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("Initializing Experiment...")

    # Override Config for Speed/Demonstration
    Config.DEBUG = True  # Use a small subset of data
    Config.EPOCHS = 1  # Train experts for only 1 epoch
    Config.GATE_EPOCHS = 2  # Train router for 2 epochs
    Config.BATCH_SIZE = 32  # Smaller batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.EXPERIMENT_ID = "demo_run"

    # Setup directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Logger
    logger = get_logger("demo", log_file=os.path.join(Config.WORKING_DIR, "demo.log"))
    logger.info("Configuration set for fast demonstration.")

    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Loading Data...")

    # get_dataloaders returns train/val loaders and size statistics for normalization
    train_loader, val_loader, size_stats = get_dataloaders(load_cached_data=False)

    # Verification
    batch = next(iter(train_loader))
    images, labels, log_sizes = batch["image"], batch["label"], batch["log_size"]

    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Incorrect image shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label shape"
    assert log_sizes.shape == (Config.BATCH_SIZE,), "Incorrect log_size shape"
    logger.info("Data loaded and verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Initialize Experts
    # -------------------------------------------------------------------------
    logger.info("Initializing Experts...")

    # Expert 1: RepVGG
    expert_repvgg = CactusRepVGG(num_classes=1).to(device)
    optimizer_rep = optim.AdamW(expert_repvgg.parameters(), lr=Config.LEARNING_RATE)

    # Expert 2: ResNet
    expert_resnet = CactusResNet(num_classes=1).to(device)
    optimizer_res = optim.AdamW(expert_resnet.parameters(), lr=Config.LEARNING_RATE)

    # Loss functions
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_aux = nn.MSELoss()  # Regression loss for file size

    # -------------------------------------------------------------------------
    # 4. Train Experts (Stage 1)
    # -------------------------------------------------------------------------
    logger.info("Training Experts (Stage 1)...")

    experts = [
        ("RepVGG", expert_repvgg, optimizer_rep),
        ("ResNet", expert_resnet, optimizer_res),
    ]

    # Store validation results for Router training
    val_results = {}

    for name, model, optimizer in experts:
        logger.info(f"Training {name}...")

        # Train one epoch
        train_loss, cls_loss, aux_loss = train_expert_one_epoch(
            train_loader,
            model,
            criterion_cls,
            criterion_aux,
            optimizer,
            device,
            epoch=0,
        )
        logger.info(
            f"[{name}] Train Loss: {train_loss:.4f} (Cls: {cls_loss:.4f}, Aux: {aux_loss:.4f})"
        )

        # Validate
        metrics = validate_expert(
            val_loader, model, criterion_cls, criterion_aux, device
        )
        logger.info(f"[{name}] Val AUC: {metrics['auc']:.4f}")

        # Store predictions and targets for Step 5
        val_results[name] = metrics

        # Verify outputs are finite
        assert np.isfinite(train_loss), f"{name} training loss is not finite"

    # Demonstrate Reparameterization for RepVGG
    logger.info("Demonstrating RepVGG Reparameterization...")
    expert_repvgg.eval()
    # Save state before
    with torch.no_grad():
        out_before, _ = expert_repvgg(images.to(device))

    # Switch to deploy mode (fuses Conv+BN)
    expert_repvgg = reparameterize_repvgg(expert_repvgg)

    # Check output consistency (should be very close)
    with torch.no_grad():
        out_after, _ = expert_repvgg(images.to(device))

    diff = (out_before - out_after).abs().mean().item()
    logger.info(f"RepVGG Reparameterization diff: {diff:.6f}")
    assert diff < 1e-4, "Reparameterization changed model output significantly"

    # -------------------------------------------------------------------------
    # 5. Prepare Data for Router (Stage 2)
    # -------------------------------------------------------------------------
    logger.info("Preparing Trust Scores for Router...")

    # Retrieve validation outputs
    # Targets are the same for both experts, pick one
    val_labels = torch.tensor(val_results["RepVGG"]["targets"]).float().view(-1, 1)
    val_aux_targets = (
        torch.tensor(val_results["RepVGG"]["aux_targets"]).float().view(-1, 1)
    )

    # Expert Predictions (Probabilities) and Aux Predictions (Log Sizes)
    # Shape: (N, 1)
    pred_rep = torch.tensor(val_results["RepVGG"]["preds"]).float().view(-1, 1)
    aux_rep = torch.tensor(val_results["RepVGG"]["aux_preds"]).float().view(-1, 1)

    pred_res = torch.tensor(val_results["ResNet"]["preds"]).float().view(-1, 1)
    aux_res = torch.tensor(val_results["ResNet"]["aux_preds"]).float().view(-1, 1)

    # Calculate Trust Scores: |Predicted_Size - True_Size|
    trust_rep = calculate_trust_score(aux_rep, val_aux_targets)
    trust_res = calculate_trust_score(aux_res, val_aux_targets)

    # Stack inputs for Router: (N, Num_Experts)
    # Router Input: Trust Scores
    router_input = torch.cat([trust_rep.view(-1, 1), trust_res.view(-1, 1)], dim=1)

    # Expert Predictions for weighted ensemble
    expert_preds_stacked = torch.cat([pred_rep, pred_res], dim=1)

    assert router_input.shape[1] == 2, "Router input should have 2 columns (2 experts)"

    # -------------------------------------------------------------------------
    # 6. Train Router
    # -------------------------------------------------------------------------
    logger.info("Training Trust Router (Stage 2)...")

    router = TrustRouter(num_experts=2).to(device)
    optimizer_router = optim.Adam(router.parameters(), lr=Config.GATE_LR)
    criterion_router = nn.BCELoss()  # Weighted prediction vs Label

    for epoch in range(Config.GATE_EPOCHS):
        loss = train_router_epoch(
            router_input,
            expert_preds_stacked,
            val_labels,
            router,
            criterion_router,
            optimizer_router,
            device,
        )
        logger.info(f"[Router] Epoch {epoch+1} Loss: {loss:.4f}")

    # Validate Router
    val_loss, val_auc = validate_router(
        router_input, expert_preds_stacked, val_labels, router, criterion_router, device
    )
    logger.info(f"[Router] Final Validation AUC: {val_auc:.4f}")

    # -------------------------------------------------------------------------
    # 7. Inference on Test Set
    # -------------------------------------------------------------------------
    logger.info("Running Inference on Test Set...")

    test_loader, test_ids = get_test_dataloader(size_stats, load_cached_data=False)

    expert_repvgg.eval()
    expert_resnet.eval()
    router.eval()

    final_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            # For test set, 'log_size' is the ground truth file size derived from disk
            true_log_sizes = batch["log_size"].to(device).float().view(-1, 1)

            # 1. Get Expert Predictions
            logits_rep, aux_rep = expert_repvgg(imgs)
            logits_res, aux_res = expert_resnet(imgs)

            prob_rep = torch.sigmoid(logits_rep)
            prob_res = torch.sigmoid(logits_res)

            # 2. Calculate Trust Scores at Inference Time
            # We know the file size of the test image, so we can check if the model
            # correctly predicted the file size (aux task).
            trust_rep = calculate_trust_score(aux_rep, true_log_sizes)
            trust_res = calculate_trust_score(aux_res, true_log_sizes)

            router_in = torch.cat([trust_rep.view(-1, 1), trust_res.view(-1, 1)], dim=1)

            # 3. Get Gating Weights
            weights = router(router_in)  # (B, 2)

            # 4. Weighted Ensemble
            # weights[:, 0] * rep + weights[:, 1] * res
            preds_stacked = torch.cat([prob_rep, prob_res], dim=1)
            weighted_preds = torch.sum(weights * preds_stacked, dim=1)

            final_predictions.extend(weighted_preds.cpu().numpy())

    # -------------------------------------------------------------------------
    # 8. Create Submission
    # -------------------------------------------------------------------------
    logger.info("Creating Submission File...")

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_predictions})

    # Ensure probabilities are valid
    submission_df["has_cactus"] = submission_df["has_cactus"].clip(0, 1)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("Head of submission:")
    print(submission_df.head())

    logger.info("Demonstration Complete.")


if __name__ == "__main__":
    main()
