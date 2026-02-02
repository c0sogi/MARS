import os
import torch
import torch.optim as optim
import numpy as np
from library import config, utils, data_loader, model, train


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Using default load_cached_data=True from library
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=config.BATCH_SIZE, num_workers=2
    )

    # 3. Model Initialization
    net = model.SKD_GN().to(device)

    # 4. Training Configuration
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = train.CombinedLoss(
        weight=class_weights,
        smoothing_lambda=config.SMOOTHING_LAMBDA,
        smoothing_threshold=config.SMOOTHING_THRESHOLD,
    )

    optimizer = optim.Adam(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 5. Training Loop
    # Limit epochs to ensure execution finishes within time constraints
    NUM_EPOCHS = 20
    best_score = float("inf")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train.train_epoch(net, train_loader, optimizer, criterion, device)
        val_score = train.validate(net, val_loader, device)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(net.state_dict(), best_model_path)

    print(f"Training complete. Best Validation Score: {best_score}")

    # 6. Final Validation & Failure Analysis
    print("Running final validation and failure analysis...")
    # Load best model
    net.load_state_dict(torch.load(best_model_path))
    net.eval()

    val_errors = []  # Store error magnitude (Levenshtein distance)
    val_lengths = []  # Store input feature (Sequence Length)

    total_distance = 0
    total_truth_length = 0

    with torch.no_grad():
        for batch in val_loader:
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            # Labels in val_loader (batch_size=1) are (1, T)
            labels = batch["labels"].numpy()[0]

            # Forward pass
            outputs = net(skeleton, audio)
            logits = outputs["p3"]  # Use final stage predictions

            # Decode predictions
            preds = torch.argmax(logits, dim=2).cpu().numpy()[0]
            pred_seq = utils.decode_predictions_to_sequence(preds)
            target_seq = utils.decode_predictions_to_sequence(labels)

            # Compute Metric
            dist = utils.levenshtein_distance(pred_seq, target_seq)
            truth_len = len(target_seq)

            total_distance += dist
            total_truth_length += truth_len

            # Collect data for failure analysis
            val_errors.append(dist)
            val_lengths.append(skeleton.shape[1])  # Sequence length (Time dimension)

    final_metric = (
        total_distance / total_truth_length if total_truth_length > 0 else float("inf")
    )

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlation
    if len(val_errors) > 1:
        # Compute Pearson correlation coefficient
        corr_matrix = np.corrcoef(val_errors, val_lengths)
        correlation = corr_matrix[0, 1]
        print(f"Correlation (Error Magnitude vs Input Length): {correlation:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 7. Submission Generation
    THRESHOLD = 0.16539050535987748
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Ensure submission directory exists
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        train.generate_submission(net, test_loader, device, submission_path)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
