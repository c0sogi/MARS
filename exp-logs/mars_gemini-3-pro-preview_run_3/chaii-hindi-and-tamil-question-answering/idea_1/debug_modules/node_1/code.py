import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import classes and functions from the provided library files
from library.config import Config, set_seed
from library.utils import jaccard, split_sentences
from library.model import QATokenClassifier
from library.data_loader import prepare_data, SparseRetriever, QADataset
from library.trainer import Trainer


def main():
    print("Starting implementation demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup
    # --------------------------------------------------------------------------
    # Modify Config attributes to ensure the script runs quickly (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Initialize environment (directories, seeds)
    Config.setup()
    print(f"Configuration initialized. Working directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Utility Functions Verification
    # --------------------------------------------------------------------------
    print("\n--- Verifying Utility Functions ---")

    # Test Jaccard Similarity
    str1 = "apple banana orange"
    str2 = "apple banana"
    score = jaccard(str1, str2)
    # Intersection: {apple, banana} (2), Union: {apple, banana, orange} (3) -> 2/3 ~= 0.6667
    print(f"Jaccard Score ('{str1}', '{str2}'): {score:.4f}")
    assert abs(score - (2 / 3)) < 1e-5, "Jaccard calculation is incorrect."

    # Test Sentence Splitting
    text = "Hello world. How are you? I am fine!"
    sentences = split_sentences(text)
    print(f"Split Sentences: {sentences}")
    assert len(sentences) == 3, "Sentence splitting failed to produce 3 sentences."
    assert sentences[1].strip() == "How are you?", "Second sentence mismatch."

    # --------------------------------------------------------------------------
    # 3. Data Loading and Processing
    # --------------------------------------------------------------------------
    print("\n--- Verifying Data Pipeline ---")

    # Test Sparse Retriever
    retriever = SparseRetriever()
    context_pool = [
        "This is the first sentence.",
        "The answer lies in this specific sentence.",
        "This is the third sentence.",
    ]
    query = "Where is the answer?"
    retrieved_sent = retriever.retrieve(query, context_pool)
    print(f"Query: {query}")
    print(f"Retrieved: {retrieved_sent}")
    assert (
        "answer lies" in retrieved_sent
    ), "Retriever failed to select the relevant sentence."

    # Load and Process Datasets (using Debug subset)
    # We force load_cached_data=False to demonstrate the processing logic
    print("Processing datasets (Debug Mode)...")
    train_dataset, val_dataset, test_dataset = prepare_data(load_cached_data=False)

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Val Set Size:   {len(val_dataset)}")
    print(f"Test Set Size:  {len(test_dataset)}")

    assert len(train_dataset) > 0, "Training dataset is empty."
    assert isinstance(
        train_dataset, QADataset
    ), "Dataset is not an instance of QADataset."

    # Verify a single batch structure
    loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    assert "input_ids" in batch, "Batch missing input_ids."
    assert "labels" in batch, "Batch missing labels."
    print("Data loader batch structure verified.")

    # --------------------------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # --------------------------------------------------------------------------
    print("\n--- Verifying Model ---")

    model = QATokenClassifier(Config.MODEL_NAME)
    model.to(Config.DEVICE)

    # Perform a forward pass with the batch fetched above
    input_ids = batch["input_ids"].to(Config.DEVICE)
    attention_mask = batch["attention_mask"].to(Config.DEVICE)
    labels = batch["labels"].to(Config.DEVICE)

    outputs = model(input_ids, attention_mask, labels=labels)

    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits Shape: {outputs.logits.shape}")

    assert outputs.loss is not None, "Model did not return loss."
    assert outputs.logits.shape == (
        2,
        Config.MAX_LEN,
        Config.NUM_LABELS,
    ), f"Logits shape mismatch. Expected {(2, Config.MAX_LEN, Config.NUM_LABELS)}, got {outputs.logits.shape}"

    # --------------------------------------------------------------------------
    # 5. Training and Validation Loop
    # --------------------------------------------------------------------------
    print("\n--- Verifying Trainer ---")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    trainer = Trainer(model, tokenizer, Config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)

    # Train for 1 epoch
    print("Running training epoch...")
    avg_loss = trainer.train_epoch(loader, optimizer, scheduler, epoch=0)
    print(f"Average Train Loss: {avg_loss:.4f}")

    # Validate
    # Load raw validation dataframe for text comparison
    df_val = pd.read_csv(Config.VAL_DATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False
    )

    print("Running validation...")
    val_score = trainer.validate(val_loader, df_val)
    print(f"Validation Jaccard Score: {val_score:.4f}")

    # Save and Load Model
    print("Saving model...")
    trainer.save_model(Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not created."

    print("Loading model...")
    trainer.load_model(Config.MODEL_SAVE_PATH)

    # --------------------------------------------------------------------------
    # 6. Prediction and Submission
    # --------------------------------------------------------------------------
    print("\n--- Verifying Prediction ---")

    df_test = pd.read_csv(Config.TEST_DATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False
    )

    ids, predictions = trainer.predict(test_loader, df_test)

    print(f"Generated {len(predictions)} predictions.")
    print(f"Sample Prediction: ID={ids[0]}, Text='{predictions[0]}'")

    assert len(ids) == len(
        df_test
    ), "Number of predictions does not match test set size."

    # Create Submission File
    submission = pd.DataFrame({"id": ids, "PredictionString": predictions})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\nImplementation demonstration completed successfully.")


if __name__ == "__main__":
    main()
