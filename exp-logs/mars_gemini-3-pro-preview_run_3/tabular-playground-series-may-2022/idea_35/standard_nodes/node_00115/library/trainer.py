import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    WORKING_DIR,
    INPUT_DIR,
)
from library.data_loader import prepare_data
from library.model import HPFEModel


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    Aggregates loss from all 5 independent streams.
    """
    model.train()
    running_loss = 0.0

    for cat_inputs, cont_inputs, targets in loader:
        cat_inputs = cat_inputs.to(device)
        cont_inputs = cont_inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape [Batch, 1]

        optimizer.zero_grad()

        # Forward pass: returns list of 5 tensors
        outputs_list = model(cat_inputs, cont_inputs)

        # Calculate loss: Sum of BCE for each stream
        loss = 0
        for output in outputs_list:
            loss += criterion(output, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Predictions are the mean of the probabilities from the 5 streams.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for cat_inputs, cont_inputs, targets in loader:
            cat_inputs = cat_inputs.to(device)
            cont_inputs = cont_inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Forward pass
            outputs_list = model(cat_inputs, cont_inputs)

            # Calculate validation loss for monitoring (sum of streams)
            loss = 0
            stream_probs = []

            for output in outputs_list:
                loss += criterion(output, targets)
                # Apply sigmoid to get probability for this stream
                stream_probs.append(torch.sigmoid(output))

            running_loss += loss.item()

            # Ensemble prediction: Mean of probabilities
            # Stack streams [5, Batch, 1] -> Mean dim 0 -> [Batch, 1]
            avg_probs = torch.stack(stream_probs).mean(dim=0)

            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc_score = roc_auc_score(all_targets, all_preds)
    avg_loss = running_loss / len(loader)

    return avg_loss, auc_score


def run_training(load_cached_data=True, patience=5):
    """
    Main training loop with Early Stopping.
    """
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print("Preparing data...")
    train_ds, val_ds, test_ds, vocab_sizes = prepare_data(
        load_cached_data=load_cached_data
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Determine continuous input dimension
    # We can get one sample to check shape
    _, sample_cont, _ = train_ds[0]
    cont_dim = sample_cont.shape[0]

    print(
        f"Initializing HPFE Model with {len(vocab_sizes)} categorical and {cont_dim} continuous features."
    )
    model = HPFEModel(vocab_sizes=vocab_sizes, cont_dim=cont_dim)
    model.to(DEVICE)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Warmup for first 10%
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")
    return best_model_path, test_ds, vocab_sizes, cont_dim


def generate_submission(model_path, test_ds, vocab_sizes, cont_dim):
    """
    Generates predictions for the test set using the best model.
    Saves to ./submission/submission.csv
    """
    print("Generating submission...")
    set_seed(SEED)

    # Load Model
    model = HPFEModel(vocab_sizes=vocab_sizes, cont_dim=cont_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    all_preds = []

    with torch.no_grad():
        for cat_inputs, cont_inputs in test_loader:
            cat_inputs = cat_inputs.to(DEVICE)
            cont_inputs = cont_inputs.to(DEVICE)

            outputs_list = model(cat_inputs, cont_inputs)

            # Ensemble prediction: Mean of probabilities
            stream_probs = [torch.sigmoid(out) for out in outputs_list]
            avg_probs = torch.stack(stream_probs).mean(dim=0)

            all_preds.append(avg_probs.cpu().numpy())

    predictions = np.concatenate(all_preds).flatten()

    # Load sample submission to get IDs
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    submission_df = pd.read_csv(sample_sub_path)

    # Ensure lengths match
    if len(predictions) != len(submission_df):
        print(
            f"Warning: Prediction length {len(predictions)} does not match submission length {len(submission_df)}."
        )
        # In case test set size differs from sample_submission (unlikely in this setup but good practice)
        # We assume the order is correct as per test.csv
        submission_df = submission_df.iloc[: len(predictions)]

    submission_df["target"] = predictions

    # Save submission
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Run training
    best_model_path, test_ds, vocab_sizes, cont_dim = run_training(
        load_cached_data=True
    )

    # Generate submission
    generate_submission(best_model_path, test_ds, vocab_sizes, cont_dim)
