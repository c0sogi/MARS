import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from library.config import Paths, TrainConfig, DataConfig, LABEL_MAP
from library.utils import (
    set_seed,
    compute_levenshtein,
    LogSpaceSmoothingLoss,
    rle_encode,
    predictions_to_string,
)
from library.data_loader import get_dataloaders
from library.model import CKARFNet


def run_training():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure output directories exist
    os.makedirs(Paths.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Paths.SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=2,
        debug_size=DataConfig.DEBUG_SAMPLE_SIZE,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = CKARFNet().to(device)

    # ==========================================
    # 4. Optimizer & Loss
    # ==========================================
    optimizer = optim.Adam(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY,
    )

    # Class weights: Background (0) gets 0.2, others get 1.0
    class_weights = torch.ones(DataConfig.NUM_CLASSES).to(device)
    class_weights[0] = TrainConfig.BACKGROUND_WEIGHT

    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)

    criterion_smooth = LogSpaceSmoothingLoss(
        lambda_weight=TrainConfig.SMOOTHING_LAMBDA,
        threshold=TrainConfig.SMOOTHING_THRESHOLD,
    )

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_levenshtein = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Paths.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {TrainConfig.EPOCHS} epochs...")

    for epoch in range(1, TrainConfig.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)  # (B, T, InputDim)
            labels = batch["labels"].to(device)  # (B, T)

            optimizer.zero_grad()

            # Forward pass returns tuple: (logits1, logits2, logits3)
            # Shapes: (B, T, NumClasses)
            outputs = model(features)

            batch_loss = 0.0

            # Calculate Cascaded Loss
            for stage_idx, logits in enumerate(outputs):
                # Cross Entropy expects (B, C, T)
                logits_permuted = logits.permute(0, 2, 1)
                loss_ce = criterion_ce(logits_permuted, labels)

                # Smoothing Loss expects LogProbs (B, T, C)
                log_probs = F.log_softmax(logits, dim=2)
                loss_sm = criterion_smooth(log_probs)

                # Weighted Stage Loss
                stage_loss = loss_ce + loss_sm
                batch_loss += stage_loss * TrainConfig.LOSS_STAGES_WEIGHTS[stage_idx]

            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)  # (1, T)

                # Forward pass - use final stage output
                outputs = model(features)
                final_logits = outputs[-1]  # (1, T, C)

                # Decode Prediction
                preds = torch.argmax(final_logits, dim=2)  # (1, T)
                pred_seq = preds[0].cpu().numpy()
                pred_gestures = rle_encode(pred_seq, min_duration=5, background_class=0)

                # Decode Target
                target_seq = labels[0].cpu().numpy()
                target_gestures = rle_encode(
                    target_seq, min_duration=1, background_class=0
                )

                val_preds.append(pred_gestures)
                val_targets.append(target_gestures)

        # Compute Metric
        val_score = compute_levenshtein(val_preds, val_targets)

        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Levenshtein: {val_score:.6f}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_score < best_levenshtein:
            best_levenshtein = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {val_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= TrainConfig.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("Generating submission...")

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No best model found. Using current model.")

    model.eval()
    submission_lines = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            sample_id = batch["sample_id"][0]

            outputs = model(features)
            final_logits = outputs[-1]

            preds = torch.argmax(final_logits, dim=2)
            pred_seq = preds[0].cpu().numpy()

            # Apply RLE and filtering
            pred_gestures = rle_encode(pred_seq, min_duration=5, background_class=0)

            # Format string
            line = predictions_to_string(sample_id, pred_gestures)
            submission_lines.append(line)

    # Write to file
    with open(Paths.SUBMISSION_FILE, "w") as f:
        for line in submission_lines:
            f.write(line + "\n")

    print(f"Submission saved to {Paths.SUBMISSION_FILE}")


if __name__ == "__main__":
    run_training()
