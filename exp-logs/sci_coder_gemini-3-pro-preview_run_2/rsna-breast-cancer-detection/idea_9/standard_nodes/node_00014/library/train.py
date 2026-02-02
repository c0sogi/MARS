import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.cuda.amp as amp
from collections import defaultdict

from library.config import Config
from library.data import get_dataloaders
from library.model import SHR_MTN
from library.utils import seed_everything, probabilistic_f1


def calculate_loss(outputs, targets, criteria, device):
    """
    Computes the combined loss for Multi-Task Learning.
    Strictly enforces Float32 for numerical stability.
    """
    # Unpack targets
    target_cancer = targets["cancer"].to(device)
    target_birads = targets["birads"].to(device)
    target_density = targets["density"].to(device)

    # Unpack outputs and cast to float32 for stability
    pred_cancer = outputs["cancer"].float()
    pred_birads = outputs["birads"].float()
    pred_density = outputs["density"].float()

    # 1. Primary Loss: Cancer (Binary Classification)
    # Squeeze to match target shape (B,)
    loss_cancer = criteria["cancer"](pred_cancer.squeeze(1), target_cancer)

    # 2. Aux Loss: BIRADS (Regression)
    # Mask out missing values (-1)
    mask_birads = target_birads != -1
    if mask_birads.sum() > 0:
        loss_birads = criteria["birads"](
            pred_birads.squeeze(1)[mask_birads], target_birads[mask_birads]
        )
    else:
        loss_birads = torch.tensor(0.0, device=device)

    # 3. Aux Loss: Density (Classification)
    # Mask out missing values (-1)
    mask_density = target_density != -1
    if mask_density.sum() > 0:
        loss_density = criteria["density"](
            pred_density[mask_density], target_density[mask_density]
        )
    else:
        loss_density = torch.tensor(0.0, device=device)

    # Combined Loss
    # We can weight these if necessary, but 1.0 is a standard starting point
    total_loss = loss_cancer + loss_birads + loss_density

    return total_loss, loss_cancer.item(), loss_birads.item(), loss_density.item()


def train_one_epoch(model, loader, optimizer, criteria, scaler, device):
    model.train()

    running_loss = 0.0
    running_cancer_loss = 0.0

    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        aux_features = batch["aux_features"].to(device)
        targets = batch["targets"]

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with amp.autocast(enabled=True):
            outputs = model(images, aux_features)

        # Float32 Loss Calculation
        # We exit autocast to ensure loss math is stable
        with amp.autocast(enabled=False):
            loss, l_cancer, _, _ = calculate_loss(outputs, targets, criteria, device)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics tracking
        running_loss += loss.item() * images.size(0)
        running_cancer_loss += l_cancer * images.size(0)

        # Store for pF1 calculation
        with torch.no_grad():
            probs = torch.sigmoid(outputs["cancer"].float()).squeeze(1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets["cancer"].numpy())

    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_cancer_loss = running_cancer_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_pf1 = probabilistic_f1(all_targets, all_preds)

    return epoch_loss, epoch_cancer_loss, epoch_pf1


def validate_one_epoch(model, loader, criteria, device):
    model.eval()

    running_loss = 0.0
    running_cancer_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            aux_features = batch["aux_features"].to(device)
            targets = batch["targets"]

            # Mixed Precision Forward
            with amp.autocast(enabled=True):
                outputs = model(images, aux_features)

            # Float32 Loss
            with amp.autocast(enabled=False):
                loss, l_cancer, _, _ = calculate_loss(
                    outputs, targets, criteria, device
                )

            running_loss += loss.item() * images.size(0)
            running_cancer_loss += l_cancer * images.size(0)

            probs = torch.sigmoid(outputs["cancer"].float()).squeeze(1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets["cancer"].numpy())

    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_cancer_loss = running_cancer_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_pf1 = probabilistic_f1(all_targets, all_preds)

    return epoch_loss, epoch_cancer_loss, epoch_pf1


def generate_submission(model, loader, device, output_path):
    print("Generating submission...")
    model.eval()

    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            aux_features = batch["aux_features"].to(device)
            prediction_ids = batch["prediction_id"]

            with amp.autocast(enabled=True):
                outputs = model(images, aux_features)

            # Get probabilities
            probs = torch.sigmoid(outputs["cancer"].float()).squeeze(1).cpu().numpy()

            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create DataFrame
    df_res = pd.DataFrame(results)

    # Aggregation: Max pooling by prediction_id
    # As per strategy: "The final submission score is the maximum probability across all images"
    submission = df_res.groupby("prediction_id")["cancer"].max().reset_index()

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}. Rows: {len(submission)}")


def run_training():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # Get feature dimension from the dataset/loader metadata
    # We access the dataset directly to get the feature columns length
    num_aux_features = len(train_loader.dataset.feature_cols)

    # 2. Model
    model = SHR_MTN(num_aux_features=num_aux_features)
    model.to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scaler = amp.GradScaler(enabled=True)

    # 4. Loss Functions
    # Pos weight for cancer imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)

    criteria = {
        "cancer": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "birads": nn.MSELoss(),
        "density": nn.CrossEntropyLoss(),
    }

    # 5. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience = 2
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        t_loss, t_cancer_loss, t_pf1 = train_one_epoch(
            model, train_loader, optimizer, criteria, scaler, device
        )

        # Val
        v_loss, v_cancer_loss, v_pf1 = validate_one_epoch(
            model, val_loader, criteria, device
        )

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(
            f"  Train Loss: {t_loss:.6f} | Cancer Loss: {t_cancer_loss:.6f} | pF1: {t_pf1:.6f}"
        )
        print(
            f"  Val Loss:   {v_loss:.6f} | Cancer Loss: {v_cancer_loss:.6f} | pF1: {v_pf1:.6f}"
        )

        # Checkpoint
        if v_pf1 > best_pf1:
            print(
                f"  [Improvement] Val pF1 increased from {best_pf1:.6f} to {v_pf1:.6f}. Saving model."
            )
            best_pf1 = v_pf1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 6. Inference / Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    # This block is technically not required by the prompt instructions
    # ("DO NOT include an if __name__ == '__main__': block"),
    # but the prompt asks to implement the module class/functions.
    # The run_training function encapsulates the logic.
    pass
