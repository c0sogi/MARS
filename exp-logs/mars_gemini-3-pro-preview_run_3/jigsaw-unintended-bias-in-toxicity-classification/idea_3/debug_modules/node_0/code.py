import os
import sys
import torch
import pandas as pd
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.dataset import load_processed_train_data, ToxicityDataset
from library.model import MultiTaskRoBERTa
from library.trainer import Trainer


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # 1. Setup and Configuration Overrides for Speed
    print("Initializing demonstration...")
    set_seed(Config.SEED)

    # Suppress transformers logging
    transformers.logging.set_verbosity_error()

    # Override Config for a fast run
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loading
    print("Loading data subsets...")

    # Load Train Data (uses logic from library.dataset)
    # This calculates sample weights and handles caching logic
    train_df = load_processed_train_data(load_cached_data=False, debug=True)

    # Load Validation Data (Manual subsetting for speed)
    val_df = pd.read_csv(Config.VAL_PATH).iloc[:100]

    # Load Test Data (Manual subsetting for speed)
    test_df = pd.read_csv(Config.TEST_PATH).iloc[:100]

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # 3. Dataset and DataLoader Preparation
    print("Preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    train_dataset = ToxicityDataset(train_df, tokenizer, mode="train")
    val_dataset = ToxicityDataset(val_df, tokenizer, mode="val")
    test_dataset = ToxicityDataset(test_df, tokenizer, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 4. Model Initialization & Verification
    print("Initializing model...")
    device = Config.DEVICE
    model = MultiTaskRoBERTa(pretrained=True)
    model.to(device)

    # Verify Model Output Shape
    # Fetch one batch to check dimensions
    sample_batch = next(iter(train_loader))
    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Expected output: (batch_size, TOTAL_OUTPUTS)
    # TOTAL_OUTPUTS = 1 (Main) + 6 (Aux) = 7
    expected_shape = (input_ids.size(0), Config.TOTAL_OUTPUTS)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"
    print("Model output shape verified.")

    # 5. Training Setup
    print("Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        val_df=val_df,
        device=device,
    )

    # 6. Run Training
    print("Starting training loop (1 Epoch)...")
    trainer.fit(epochs=Config.EPOCHS)

    # Verify model was saved
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_save_path), "Model file was not saved."
    print("Training complete. Best model saved.")

    # 7. Inference
    print("Running inference on test set...")
    ids, predictions = trainer.predict(test_loader)

    # Verify predictions
    assert len(ids) == len(
        test_df
    ), "Mismatch in number of predictions vs test samples."
    assert len(predictions) == len(test_df), "Mismatch in number of prediction scores."
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions out of probability range [0, 1]."

    # 8. Create Submission
    print("Generating submission file...")
    submission_df = pd.DataFrame({"id": ids, "prediction": predictions})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Demonstration completed successfully.")
