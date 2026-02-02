import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import ChemicalDataset, get_transforms
from library.model import ShowAndTell
from library.trainer import Trainer
from library.utils import compute_levenshtein


def run_demonstration():
    print("=== Starting InChI Prediction Demonstration ===")

    # 1. Configuration Overrides for Speed
    # We override default config values to ensure the demo runs quickly
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    # Reduce model size for faster initialization and forward pass in demo
    Config.EMBED_DIM = 128
    Config.HIDDEN_SIZE = 256

    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Tokenizer Initialization and Verification
    print("\n[2] Initializing and verifying Tokenizer...")
    tokenizer = Tokenizer()
    # Build vocab (this will load from cache if available or build from metadata)
    # We force load_cached_data=False if we want to ensure it works from scratch,
    # but for this env, we rely on the logic in tokenizer.py.
    # Since metadata exists, it should work.
    tokenizer.build_vocab()

    # Verify Tokenizer Logic
    test_inchi = "InChI=1S/H2O/h1H2"
    encoded = tokenizer.encode(test_inchi)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_inchi}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded: {decoded}")

    # Assertions
    assert isinstance(encoded, torch.Tensor), "Encoded output should be a tensor"
    assert len(encoded) == Config.MAX_LEN, f"Encoded length should be {Config.MAX_LEN}"
    assert (
        decoded == test_inchi
    ), "Decoded string should match original (ignoring special tokens)"
    print("Tokenizer verification passed.")

    # 3. Dataset and DataLoader Setup
    print("\n[3] Setting up Datasets and DataLoaders...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Subsample for demonstration speed
    df_train_demo = df_train.head(Config.DEBUG_SAMPLE_SIZE).copy()
    df_val_demo = df_val.head(Config.DEBUG_SAMPLE_SIZE).copy()

    print(f"Train subset size: {len(df_train_demo)}")
    print(f"Val subset size: {len(df_val_demo)}")

    # Instantiate Datasets
    train_dataset = ChemicalDataset(
        df_train_demo, tokenizer, transform=get_transforms("train")
    )
    val_dataset = ChemicalDataset(
        df_val_demo, tokenizer, transform=get_transforms("valid")
    )

    # Verify Dataset Item
    img, label = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label shape: {label.shape}")

    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image dimensions"
    assert label.shape == (Config.MAX_LEN,), "Incorrect label dimensions"
    assert label.dtype == torch.long, "Label should be LongTensor"

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    print("DataLoaders initialized.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = ShowAndTell(
        vocab_size=len(tokenizer),
        sos_idx=tokenizer.sos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
        max_len=Config.MAX_LEN,
    )
    model = model.to(Config.DEVICE)

    # Verify Model Forward Pass (with dummy batch)
    dummy_imgs = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )
    dummy_caps = torch.randint(0, len(tokenizer), (2, Config.MAX_LEN)).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(dummy_imgs, dummy_caps)
        print(f"Model output shape: {outputs.shape}")
        # Expected: (batch, max_len, vocab_size)
        assert outputs.shape == (
            2,
            Config.MAX_LEN,
            len(tokenizer),
        ), "Model output shape mismatch"

        # Verify Inference Sample
        samples = model.sample(dummy_imgs)
        print(f"Inference sample shape: {samples.shape}")
        # Expected: (batch, max_len)
        assert samples.shape == (2, Config.MAX_LEN), "Inference sample shape mismatch"

    print("Model verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Starting Training Loop Demonstration...")

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        patience=Config.PATIENCE,
    )

    # Run fit (1 epoch as configured)
    trainer.fit(epochs=Config.EPOCHS)

    print("Training loop completed.")

    # 6. Prediction Demonstration
    print("\n[6] Generating Predictions on Test Subset...")

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    df_test_demo = df_test.head(10).copy()  # Predict on 10 images

    test_dataset = ChemicalDataset(
        df_test_demo, tokenizer, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    predictions = trainer.predict(test_loader)

    print(f"Generated {len(predictions)} predictions.")
    print("Sample Prediction:")
    print(f"Image ID: {df_test_demo.iloc[0]['image_id']}")
    print(f"Predicted InChI: {predictions[0]}")

    assert len(predictions) == len(df_test_demo), "Number of predictions mismatch"
    assert isinstance(predictions[0], str), "Prediction should be a string"
    assert predictions[0].startswith(
        "InChI="
    ), "Prediction should likely start with 'InChI='"

    # 7. Create Submission File
    print("\n[7] Creating Submission File...")
    submission_df = pd.DataFrame(
        {"image_id": df_test_demo["image_id"], "InChI": predictions}
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    assert os.path.exists(submission_path), "Submission file was not created"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
