import os
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import components from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import PhraseDataset
from library.model import SiameseBiEncoder
from library.engine import fit, generate_submission

if __name__ == "__main__":
    print("Starting demonstration script...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast demonstration (Optimize for Speed)
    Config.debug = True
    Config.debug_subset_size = 50  # Use only 50 samples
    Config.epochs = 2  # Train for only 2 epochs
    Config.train_batch_size = 8
    Config.val_batch_size = 8

    # Initialize directories and set random seeds
    Config.setup()
    set_seed(Config.seed)

    print(f"Configuration: Debug={Config.debug}, Device={Config.device}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("Loading tokenizer and datasets...")
    # Load tokenizer once to pass to datasets
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Instantiate Datasets
    # load_and_preprocess handles caching and debug truncation internally
    train_dataset = PhraseDataset(split="train", tokenizer=tokenizer)
    val_dataset = PhraseDataset(split="val", tokenizer=tokenizer)
    test_dataset = PhraseDataset(split="test", tokenizer=tokenizer)

    # Verify Dataset Logic
    print("Verifying dataset integrity...")
    # Check if debug truncation worked
    assert (
        len(train_dataset) == Config.debug_subset_size
    ), f"Train dataset size mismatch. Expected {Config.debug_subset_size}, got {len(train_dataset)}"

    # Check item structure
    sample_item = train_dataset[0]
    expected_keys = {
        "anchor_input_ids",
        "anchor_attention_mask",
        "target_input_ids",
        "target_attention_mask",
        "labels",
        "id",
    }
    assert expected_keys.issubset(
        sample_item.keys()
    ), f"Missing keys in dataset item. Found: {sample_item.keys()}"

    # Create DataLoaders
    # num_workers=0 to avoid overhead in this small demo
    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.val_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.val_batch_size, shuffle=False, num_workers=0
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = SiameseBiEncoder(model_name=Config.model_name)
    model.to(Config.device)

    # Verify Model Logic (Forward Pass)
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        # Fetch a single batch
        batch = next(iter(train_loader))

        # Move inputs to device
        anchor_ids = batch["anchor_input_ids"].to(Config.device)
        anchor_mask = batch["anchor_attention_mask"].to(Config.device)
        target_ids = batch["target_input_ids"].to(Config.device)
        target_mask = batch["target_attention_mask"].to(Config.device)

        # Perform forward pass
        outputs = model(
            anchor_input_ids=anchor_ids,
            anchor_attention_mask=anchor_mask,
            target_input_ids=target_ids,
            target_attention_mask=target_mask,
        )

        # Check output shape: should be [batch_size]
        assert outputs.shape == (
            anchor_ids.size(0),
        ), f"Model output shape mismatch. Expected {(anchor_ids.size(0),)}, got {outputs.shape}"

        # Check output type
        assert outputs.dtype == torch.float32, "Model output dtype should be float32"

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("Starting training loop...")

    # Setup Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Execute Training
    model = fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=Config.device,
        epochs=Config.epochs,
        patience=Config.early_stopping_patience,
        save_path=Config.model_save_path,
    )

    # -------------------------------------------------------------------------
    # 5. Inference & Submission
    # -------------------------------------------------------------------------
    print("Generating submission...")
    generate_submission(
        model=model,
        test_dataloader=test_loader,
        device=Config.device,
        output_path=Config.submission_path,
    )

    # Verify Submission File
    print("Verifying submission file...")
    assert os.path.exists(Config.submission_path), "Submission file was not created."

    df_sub = pd.read_csv(Config.submission_path)

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "score",
    ], f"Submission columns mismatch. Found {df_sub.columns}"

    # Check length (should match debug subset size)
    assert len(df_sub) == len(
        test_dataset
    ), f"Submission length mismatch. Expected {len(test_dataset)}, got {len(df_sub)}"

    # Check score validity
    assert (
        df_sub["score"].min() >= 0.0 and df_sub["score"].max() <= 1.0
    ), "Predicted scores are out of valid range [0, 1]"

    print("Demo execution completed successfully.")
