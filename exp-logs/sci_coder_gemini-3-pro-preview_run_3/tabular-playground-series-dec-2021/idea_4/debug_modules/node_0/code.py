import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import load_and_preprocess, DAEDataset, CoverTypeDataset
from library.models import DenoisingAutoencoder, ResNetClassifier
from library.trainers import train_dae, train_classifier


def main():
    print("=== Forest Cover Type Prediction Pipeline Demo ===")

    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    # loading cached data is fast. We will then subsample for the demo.
    print("\n[Step 1] Loading Data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_preprocess(
        load_cached_data=True
    )

    # Validate dimensions
    input_dim = X_train.shape[1]
    print(f"Original Train Shape: {X_train.shape}")
    print(f"Feature Dimension: {input_dim}")

    # Assertion: Ensure we have features
    assert input_dim > 0, "Input dimension must be positive"
    assert len(X_train) == len(y_train), "Mismatch in training features and labels"

    # OPTIMIZATION: Subsample data for rapid demonstration
    DEMO_SIZE = 2048  # Small enough for seconds, large enough for batching
    print(f"Subsampling data to {DEMO_SIZE} rows for demo execution...")

    X_train_sub = X_train[:DEMO_SIZE]
    y_train_sub = y_train[:DEMO_SIZE]
    X_val_sub = X_val[:DEMO_SIZE]
    y_val_sub = y_val[:DEMO_SIZE]
    X_test_sub = X_test[:DEMO_SIZE]
    test_ids_sub = test_ids[:DEMO_SIZE]

    # 3. Denoising Autoencoder (DAE) Setup & Training
    print("\n[Step 2] DAE Pretraining Demo...")

    # Create Datasets and Loaders
    # We use both train and test data for unsupervised learning in practice,
    # but here we just use the train subset for simplicity.
    dae_train_dataset = DAEDataset(X_train_sub, noise_prob=Config.SWAP_NOISE_PROB)
    dae_val_dataset = DAEDataset(X_val_sub, noise_prob=0.0)  # No noise for validation

    dae_train_loader = DataLoader(
        dae_train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )
    dae_val_loader = DataLoader(
        dae_val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Instantiate Model
    dae_model = DenoisingAutoencoder(
        input_dim=input_dim, hidden_dim=Config.HIDDEN_DIM, latent_dim=Config.LATENT_DIM
    )

    # Train DAE (1 Epoch for speed)
    dae_model = train_dae(
        dae_model,
        dae_train_loader,
        dae_val_loader,
        epochs=1,
        lr=Config.LR_PRETRAIN,
        device=device,
    )

    # Verification: Check if encoder produces correct shape
    dummy_input = torch.from_numpy(X_train_sub[:5]).float().to(device)
    with torch.no_grad():
        _, latent = dae_model(dummy_input)
    assert latent.shape == (
        5,
        Config.LATENT_DIM,
    ), f"Expected latent shape (5, {Config.LATENT_DIM}), got {latent.shape}"
    print("DAE verification passed.")

    # 4. Classifier Fine-Tuning
    print("\n[Step 3] Classifier Fine-Tuning Demo...")

    # Create Datasets and Loaders
    clf_train_dataset = CoverTypeDataset(X_train_sub, y_train_sub)
    clf_val_dataset = CoverTypeDataset(X_val_sub, y_val_sub)

    clf_train_loader = DataLoader(
        clf_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    clf_val_loader = DataLoader(
        clf_val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Instantiate Classifier using the trained encoder
    # Note: We pass the encoder from the DAE instance
    classifier_model = ResNetClassifier(
        encoder=dae_model.encoder,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    )

    # Train Classifier (1 Epoch for speed)
    classifier_model = train_classifier(
        classifier_model,
        clf_train_loader,
        clf_val_loader,
        epochs=1,
        lr=Config.LR_FINETUNE,
        patience=1,  # Minimal patience
        device=device,
    )

    # 5. Inference
    print("\n[Step 4] Generating Predictions...")

    classifier_model.eval()

    # Create Test Loader
    test_dataset = CoverTypeDataset(X_test_sub, targets=None)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    predictions = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            logits = classifier_model(x)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    predictions = np.array(predictions)

    # Verification: Check predictions shape and value range
    assert len(predictions) == len(test_ids_sub), "Prediction count mismatch"
    # Targets were shifted 0-6 internally, we need to shift back to 1-7 for submission
    final_predictions = predictions + 1

    assert np.all(final_predictions >= 1) and np.all(
        final_predictions <= 7
    ), "Predictions out of valid range [1, 7]"

    print(f"Generated {len(final_predictions)} predictions.")
    print(f"Sample predictions: {final_predictions[:10]}")

    # 6. Create Submission
    submission = pd.DataFrame(
        {Config.ID_COL: test_ids_sub, Config.TARGET_COL: final_predictions}
    )

    # Just print head to verify format, avoiding write to disk to keep demo clean
    print("\nSubmission DataFrame Head:")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
