import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import provided library components
from library.config import Config
from library.utils import seed_everything, jaccard, postprocess_qa_predictions
from library.data import get_data, get_folds, QADataset
from library.model import XLMROBERTAForQA
from library.engine import train_fn, eval_fn


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("Initializing demonstration script...")
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration (Debug Mode)
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples for speed
    Config.N_FOLDS = 2  # Use 2 folds, we will only run Fold 0
    Config.OUTPUT_DIR = "./working/demo_run"
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(
        f"Configuration set: Debug={Config.DEBUG_SAMPLE_SIZE}, Epochs={Config.EPOCHS}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\nLoading and processing data (Debug Mode)...")
    # get_data handles tokenization, sliding window, and caching
    # debug=True ensures we only process a small subset
    train_features, test_features = get_data(load_cached_data=False, debug=True)

    print(f"Processed Train Features Shape: {train_features.shape}")
    print(f"Processed Test Features Shape: {test_features.shape}")

    # Validate feature structure
    required_cols = [
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "example_id",
        "fold",
    ]
    for col in required_cols:
        assert col in train_features.columns, f"Missing column {col} in train features"

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader Preparation (Fold 0)
    # -------------------------------------------------------------------------
    fold_idx = 0
    print(f"\nPreparing data for Fold {fold_idx}...")

    # Split features into train and validation for this fold
    df_train_fold, df_val_fold = get_folds(train_features, fold_idx)

    # Instantiate PyTorch Datasets
    train_dataset = QADataset(df_train_fold, mode="train")
    val_dataset = QADataset(df_val_fold, mode="train")  # Validation set has targets

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple sequential execution in demo
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing XLM-RoBERTa model...")
    device = Config.DEVICE
    model = XLMROBERTAForQA(pretrained=True)
    model.to(device)

    # Verification: Check output shape with a dummy batch
    dummy_batch = next(iter(train_loader))
    d_ids = dummy_batch["input_ids"].to(device)
    d_mask = dummy_batch["attention_mask"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(d_ids, d_mask)

    # Assert logits shape: (batch_size, sequence_length)
    assert start_logits.shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LENGTH)
    assert end_logits.shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LENGTH)
    print("Model output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(Config.WARMUP_RATIO * num_train_steps),
        num_training_steps=num_train_steps,
    )

    print("\nStarting training...")
    # Run training for 1 epoch
    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"Epoch 1 Training Loss: {avg_train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 6. Evaluation and Metric Calculation
    # -------------------------------------------------------------------------
    print("\nStarting validation evaluation...")
    avg_val_loss, (val_start_logits, val_end_logits) = eval_fn(
        val_loader, model, device
    )
    print(f"Epoch 1 Validation Loss: {avg_val_loss:.4f}")

    # Post-processing: Convert logits to text
    # We need the raw examples (text) to map offsets back to strings.
    # Load metadata and filter for the current validation fold examples.
    raw_train = pd.read_csv(Config.TRAIN_CSV)
    raw_val = pd.read_csv(Config.VAL_CSV)
    raw_all = pd.concat([raw_train, raw_val], ignore_index=True)

    val_ids = df_val_fold["example_id"].unique()
    val_examples = raw_all[raw_all["id"].isin(val_ids)].to_dict("records")
    val_features_list = df_val_fold.to_dict("records")

    print(f"Post-processing {len(val_examples)} validation examples...")
    val_predictions = postprocess_qa_predictions(
        examples=val_examples,
        features=val_features_list,
        predictions=(val_start_logits, val_end_logits),
        n_best_size=Config.N_BEST_SIZE,
        max_answer_length=Config.MAX_ANSWER_LENGTH,
    )

    # Calculate Jaccard Score
    jaccard_scores = []
    for example in val_examples:
        ex_id = example["id"]
        ground_truth = example["answer_text"]
        prediction = val_predictions.get(ex_id, "")
        score = jaccard(ground_truth, prediction)
        jaccard_scores.append(score)

    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    print(f"Validation Jaccard Score: {mean_jaccard:.4f}")

    # -------------------------------------------------------------------------
    # 7. Inference on Test Set
    # -------------------------------------------------------------------------
    print("\nRunning inference on test set...")
    test_dataset = QADataset(test_features, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    _, (test_start_logits, test_end_logits) = eval_fn(test_loader, model, device)

    # Load raw test metadata
    raw_test = pd.read_csv(Config.TEST_CSV)
    # Filter raw test examples to match the debug subset
    test_ids = test_features["example_id"].unique()
    test_examples = raw_test[raw_test["id"].isin(test_ids)].to_dict("records")
    test_features_list = test_features.to_dict("records")

    test_predictions = postprocess_qa_predictions(
        examples=test_examples,
        features=test_features_list,
        predictions=(test_start_logits, test_end_logits),
        n_best_size=Config.N_BEST_SIZE,
        max_answer_length=Config.MAX_ANSWER_LENGTH,
    )

    # -------------------------------------------------------------------------
    # 8. Submission Generation
    # -------------------------------------------------------------------------
    submission_rows = []
    for ex_id, pred_text in test_predictions.items():
        submission_rows.append({"id": ex_id, "PredictionString": pred_text})

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")
    print("Sample predictions:")
    print(submission_df.head())

    # Final assertion to ensure file exists
    assert os.path.exists(submission_path), "Submission file was not created."
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
