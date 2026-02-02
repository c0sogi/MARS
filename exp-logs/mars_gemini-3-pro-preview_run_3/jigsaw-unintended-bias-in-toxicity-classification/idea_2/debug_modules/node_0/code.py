import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer

# Import from provided library files
from library.config import Config, seed_everything
from library.data import ToxicityDataset, DynamicPaddingCollator, get_dataloaders
from library.model import DistilBertWithBiasHead
from library.utils import JigsawEvaluator
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration Overrides for Speed
    print("=== Setting up Demonstration ===")
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    # Override Config for a fast demo run
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print("Configuration overridden for speed (Epochs=1, SampleSize=50).")

    # 2. Verify Data Pipeline Components
    print("\n=== Verifying Data Pipeline ===")

    # Load a tiny slice of training data directly for component testing
    df_train = pd.read_csv(Config.TRAIN_PATH).head(10)
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # Test Dataset
    dataset = ToxicityDataset(df_train, tokenizer, is_train=True)
    sample_item = dataset[0]

    print(f"Dataset length: {len(dataset)}")
    assert "input_ids" in sample_item
    assert "attention_mask" in sample_item
    assert "target" in sample_item
    print("ToxicityDataset: Item retrieval successful.")

    # Test Collator
    collator = DynamicPaddingCollator(tokenizer.pad_token_id)
    batch_list = [dataset[i] for i in range(4)]
    batch = collator(batch_list)

    assert batch["input_ids"].shape[0] == 4
    assert batch["target"].shape[0] == 4
    # Check dynamic padding: width should match max length in this specific batch
    max_len_in_batch = max(len(x["input_ids"]) for x in batch_list)
    assert batch["input_ids"].shape[1] == max_len_in_batch
    print(f"DynamicPaddingCollator: Batch shape verified {batch['input_ids'].shape}.")

    # 3. Verify Model Architecture
    print("\n=== Verifying Model Architecture ===")
    model = DistilBertWithBiasHead(
        model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES
    )
    model.to(Config.DEVICE)
    model.eval()

    # Move batch to device
    input_ids = batch["input_ids"].to(Config.DEVICE)
    attention_mask = batch["attention_mask"].to(Config.DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert logits.shape == (4, 1), f"Expected output shape (4, 1), got {logits.shape}"
    print("DistilBertWithBiasHead: Forward pass successful.")

    # 4. Verify Evaluation Metric (JigsawEvaluator)
    print("\n=== Verifying JigsawEvaluator ===")
    # Create synthetic data
    # 10 samples: 5 toxic, 5 non-toxic
    y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Predictions: reasonably good but not perfect
    y_pred = np.array([0.9, 0.8, 0.7, 0.4, 0.9, 0.1, 0.2, 0.6, 0.1, 0.05])

    # Synthetic identity dataframe
    identity_data = {
        "male": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "female": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        # Add other columns as zeros
    }
    for col in Config.IDENTITY_COLUMNS:
        if col not in identity_data:
            identity_data[col] = [0] * 10

    identity_df = pd.DataFrame(identity_data)

    evaluator = JigsawEvaluator(y_true, y_pred, identity_df)
    score, metrics = evaluator.get_final_metric()

    print(f"Calculated Score: {score:.4f}")
    print(f"Metrics breakdown: {metrics}")

    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
    assert "overall_auc" in metrics
    assert "subgroup_auc_mean" in metrics
    print("JigsawEvaluator: Metric calculation verified.")

    # 5. End-to-End Training Loop (Trainer)
    print("\n=== Running End-to-End Training (Debug Mode) ===")

    # Initialize Trainer with debug=True (uses Config.DEBUG_SAMPLE_SIZE)
    trainer = Trainer(debug=True)

    # Run Training
    # This will run for 1 epoch on 50 samples, validating on 50 samples
    trainer.train()

    # Run Prediction
    trainer.predict_and_submit()

    # 6. Verify Submission
    print("\n=== Verifying Submission File ===")
    if os.path.exists(Config.SUBMISSION_FILE):
        sub_df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission file found at {Config.SUBMISSION_FILE}")
        print(f"Submission shape: {sub_df.shape}")

        # Check columns
        assert "id" in sub_df.columns
        assert "prediction" in sub_df.columns

        # Check length (should match DEBUG_SAMPLE_SIZE because Trainer loads test in debug mode)
        assert (
            len(sub_df) == Config.DEBUG_SAMPLE_SIZE
        ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(sub_df)}"

        print("Submission file content verified.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
