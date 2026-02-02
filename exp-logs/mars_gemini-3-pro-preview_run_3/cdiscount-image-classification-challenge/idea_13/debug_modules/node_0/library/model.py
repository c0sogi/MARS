import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pandas as pd
from library.config import Config
from library.dataset import mixup_data


class HierarchicalMLP(nn.Module):
    """
    Multi-Task MLP with a shared trunk and three hierarchical output heads.
    """

    def __init__(
        self, input_dim=Config.TOTAL_FEAT_DIM, dropout_rate=Config.DROPOUT_RATE
    ):
        super(HierarchicalMLP, self).__init__()

        # Shared Trunk
        # Input: 3328 -> 2048 -> 1024
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Hierarchical Heads
        self.fc_l1 = nn.Linear(1024, Config.NUM_CLASSES_L1)
        self.fc_l2 = nn.Linear(1024, Config.NUM_CLASSES_L2)
        self.fc_l3 = nn.Linear(1024, Config.NUM_CLASSES_L3)

    def forward(self, x):
        features = self.trunk(x)
        l1_out = self.fc_l1(features)
        l2_out = self.fc_l2(features)
        l3_out = self.fc_l3(features)
        return l1_out, l2_out, l3_out


class HierarchicalLoss(nn.Module):
    """
    Computes the sum of CrossEntropy losses for all three levels.
    Supports MixUp regularization by accepting mixed targets and lambda.
    """

    def __init__(self):
        super(HierarchicalLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, lam=None):
        """
        Args:
            outputs: Tuple (l1_logits, l2_logits, l3_logits)
            targets: Tuple of targets.
                     If lam is None: (l1, l2, l3)
                     If lam is set: (l1_a, l1_b, l2_a, l2_b, l3_a, l3_b)
            lam: MixUp coefficient (float) or None
        """
        l1_logits, l2_logits, l3_logits = outputs

        if lam is None:
            l1, l2, l3 = targets
            loss = (
                self.ce(l1_logits, l1) + self.ce(l2_logits, l2) + self.ce(l3_logits, l3)
            )
            return loss
        else:
            l1_a, l1_b, l2_a, l2_b, l3_a, l3_b = targets

            loss_l1 = lam * self.ce(l1_logits, l1_a) + (1 - lam) * self.ce(
                l1_logits, l1_b
            )
            loss_l2 = lam * self.ce(l2_logits, l2_a) + (1 - lam) * self.ce(
                l2_logits, l2_b
            )
            loss_l3 = lam * self.ce(l3_logits, l3_a) + (1 - lam) * self.ce(
                l3_logits, l3_b
            )

            return loss_l1 + loss_l2 + loss_l3


def train_one_epoch(model, loader, optimizer, criterion, device, alpha):
    """
    Trains the model for one epoch using Feature-Space MixUp.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for features, l1, l2, l3 in loader:
        features = features.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        # Apply MixUp
        mixed_x, l1_a, l1_b, l2_a, l2_b, l3_a, l3_b, lam = mixup_data(
            features, l1, l2, l3, alpha=alpha, device=device
        )

        optimizer.zero_grad()
        outputs = model(mixed_x)

        # Calculate Multi-Task MixUp Loss
        loss = criterion(outputs, (l1_a, l1_b, l2_a, l2_b, l3_a, l3_b), lam=lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Level 3 (Target) Accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    with torch.no_grad():
        for features, l1, l2, l3 in loader:
            features = features.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            l3 = l3.to(device)

            outputs = model(features)

            # Standard Loss (No MixUp)
            loss = criterion(outputs, (l1, l2, l3), lam=None)
            running_loss += loss.item() * features.size(0)

            # Calculate Accuracy for Level 3 (Target)
            _, preds_l3 = torch.max(outputs[2], 1)
            correct_l3 += (preds_l3 == l3).sum().item()
            total_samples += features.size(0)

    return running_loss / total_samples, correct_l3 / total_samples


def train_model(
    model, train_loader, val_loader, save_path, device, epochs=Config.EPOCHS
):
    """
    Runs the full training loop with Early Stopping.
    """
    criterion = HierarchicalLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    best_acc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MIXUP_ALPHA
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc (L3): {val_acc:.8f}"
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! Acc: {best_acc:.8f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for return
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_ensemble(train_loader, val_loader):
    """
    Trains the complete ensemble of models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Ensemble on {device}...")

    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    for i in range(Config.ENSEMBLE_SIZE):
        print(f"\n=== Training Ensemble Member {i+1}/{Config.ENSEMBLE_SIZE} ===")

        # Initialize new model instance
        model = HierarchicalMLP().to(device)
        save_path = os.path.join(Config.MODEL_DIR, f"ensemble_model_{i}.pth")

        # Train
        train_model(model, train_loader, val_loader, save_path, device)

        # Clear memory
        del model
        torch.cuda.empty_cache()

    print("\nEnsemble training complete.")


def predict_ensemble(test_loader):
    """
    Performs inference using the trained ensemble and generates submission.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Ensemble Inference on {device}...")

    # Load all models
    models = []
    for i in range(Config.ENSEMBLE_SIZE):
        model_path = os.path.join(Config.MODEL_DIR, f"ensemble_model_{i}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Skipping.")
            continue

        model = HierarchicalMLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No trained models found for inference.")

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for features, product_ids in test_loader:
            features = features.to(device)

            # Aggregate probabilities from all models
            ensemble_probs = torch.zeros(
                features.size(0), Config.NUM_CLASSES_L3, device=device
            )

            for model in models:
                _, _, l3_logits = model(features)
                probs = torch.softmax(l3_logits, dim=1)
                ensemble_probs += probs

            # Average
            ensemble_probs /= len(models)

            # Get predictions
            _, preds = torch.max(ensemble_probs, 1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(product_ids.numpy())

    # Concatenate results
    final_preds = np.concatenate(all_preds)
    final_ids = np.concatenate(all_ids)

    # Decode labels using HierarchyMap (need to instantiate to get inverse mapping)
    # Note: We need to reconstruct the L3 encoder to map int -> original category_id
    # We can do this by loading the hierarchy map again.
    from library.utils import HierarchyMap

    h_map = HierarchyMap(load_cached_data=True)

    # Inverse transform
    # The h_map.l3_encoder.inverse_transform expects an array
    final_category_ids = h_map.l3_encoder.inverse_transform(final_preds)

    # Create DataFrame
    submission_df = pd.DataFrame({"_id": final_ids, "category_id": final_category_ids})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
