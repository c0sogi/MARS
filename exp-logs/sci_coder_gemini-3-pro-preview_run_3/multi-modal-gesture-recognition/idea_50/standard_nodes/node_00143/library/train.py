import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from library import config, utils, data_loader, model

# Set reproducible seeds
utils.set_seed(config.SEED)


class CombinedLoss(nn.Module):
    """
    Combines Weighted Cross-Entropy with Log-Space Temporal Smoothing.
    """

    def __init__(self, weight=None, smoothing_lambda=0.15, smoothing_threshold=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction="mean")
        self.lambda_smooth = smoothing_lambda
        self.threshold = smoothing_threshold

    def forward(self, logits, targets):
        """
        Args:
            logits: (Batch, Time, NumClasses)
            targets: (Batch, Time)
        """
        # 1. Weighted Cross-Entropy
        # Transpose logits to (Batch, NumClasses, Time) for CE Loss
        ce = self.ce_loss(logits.transpose(1, 2), targets)

        # 2. Log-Space Temporal Smoothing (Truncated MSE)
        # Calculate log-probabilities
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate temporal difference: log_probs[t] - log_probs[t-1]
        # shape: (Batch, Time-1, NumClasses)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Truncate (Clamp) the differences
        clamped_diff = torch.clamp(diff, -self.threshold, self.threshold)

        # MSE of clamped differences
        mse = torch.mean(clamped_diff**2)

        return ce + self.lambda_smooth * mse


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(skeleton, audio)

        # Deep Supervision Loss
        # Calculate loss for each stage output
        loss_p1 = criterion(outputs["p1"], labels)
        loss_p2 = criterion(outputs["p2"], labels)
        loss_p3 = criterion(outputs["p3"], labels)

        # Sum losses
        loss = loss_p1 + loss_p2 + loss_p3

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            # labels might be padded in collate, but for val batch_size=1, it's just the sequence
            labels = batch["labels"].numpy()[0]  # (Time,)

            outputs = model(skeleton, audio)

            # Use final stage predictions
            logits = outputs["p3"]  # (1, Time, Classes)
            probs = F.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2).cpu().numpy()[0]  # (Time,)

            # Decode sequence
            pred_seq = utils.decode_predictions_to_sequence(preds)
            target_seq = utils.decode_predictions_to_sequence(labels)

            predictions.append(pred_seq)
            ground_truths.append(target_seq)

    # Compute metric
    score = utils.compute_normalized_levenshtein(predictions, ground_truths)
    return score


def generate_submission(model, loader, device, output_path):
    model.eval()
    results = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            sample_ids = batch["sample_ids"]

            outputs = model(skeleton, audio)
            logits = outputs["p3"]

            # Iterate over batch (though batch_size is usually 1 for inference)
            for i in range(len(sample_ids)):
                preds = torch.argmax(logits[i], dim=1).cpu().numpy()
                pred_seq = utils.decode_predictions_to_sequence(preds)

                # Format: SessionID,Label1,Label2,...
                seq_str = ",".join(map(str, pred_seq))
                results.append(f"{sample_ids[i]},{seq_str}")

    # Write to CSV
    with open(output_path, "w") as f:
        for line in results:
            f.write(line + "\n")
    print(f"Submission saved to {output_path}")


def train_model():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    # Set max_samples to config.MAX_SAMPLES for debugging if needed
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.BATCH_SIZE, max_samples=config.MAX_SAMPLES
    )

    # Initialize Model
    net = model.SKD_GN().to(device)

    # Loss Function
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = CombinedLoss(
        weight=class_weights,
        smoothing_lambda=config.SMOOTHING_LAMBDA,
        smoothing_threshold=config.SMOOTHING_THRESHOLD,
    )

    # Optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Training Loop
    best_score = float("inf")
    patience = 10
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(net, train_loader, optimizer, criterion, device)
        val_score = validate(net, val_loader, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Score (Levenshtein): {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            # print(f"New best model saved with score {best_score}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # Load best model for submission
    net.load_state_dict(torch.load(best_model_path))

    # Generate Submission
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    generate_submission(net, test_loader, device, submission_path)


if __name__ == "__main__":
    train_model()
