import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import collections
import transformers

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import QADataset, load_and_process_data

# Suppress excessive transformer warnings
transformers.logging.set_verbosity_error()


class QAModel(nn.Module):
    """
    Question Answering Model using a pre-trained backbone (MuRIL) and a linear head.
    """

    def __init__(self, config_obj):
        super(QAModel, self).__init__()
        self.config = AutoConfig.from_pretrained(config_obj.model_checkpoint)
        self.model = AutoModel.from_pretrained(
            config_obj.model_checkpoint, config=self.config
        )
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Project to 2 dimensions (start_logits, end_logits)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        # Calculate loss for both start and end positions
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict(model, dataloader, device):
    """
    Runs inference on the test set and returns raw logits.
    """
    model.eval()
    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    return np.concatenate(all_start_logits), np.concatenate(all_end_logits)


def post_process_predictions(
    features_df, start_logits, end_logits, n_best_size=20, max_answer_length=50
):
    """
    Converts logits into text predictions using offset mappings and context.
    Aggregates predictions across sliding windows for each example.
    """
    # Load original test data to get the context text
    test_df = pd.read_csv(Config.test_data_path)
    example_id_to_context = dict(zip(test_df["id"], test_df["context"]))

    predictions = {}

    # Map features (windows) to their corresponding example_id
    example_to_features = collections.defaultdict(list)
    for idx, row in features_df.iterrows():
        example_to_features[row["example_id"]].append(idx)

    # Process each example
    for example_id, feature_indices in example_to_features.items():
        context_text = example_id_to_context.get(example_id, "")
        best_score = float("-inf")
        best_answer = ""

        # Iterate through all windows for this document
        for feature_idx in feature_indices:
            s_logits = start_logits[feature_idx]
            e_logits = end_logits[feature_idx]
            offsets = features_df.iloc[feature_idx]["offset_mapping"]
            seq_ids = features_df.iloc[feature_idx]["sequence_ids"]

            # Get top-k start and end indices
            start_indexes = np.argsort(s_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(e_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Validity checks
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    # Ensure tokens are part of the context (sequence_id == 1)
                    if seq_ids[start_index] != 1 or seq_ids[end_index] != 1:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    score = s_logits[start_index] + e_logits[end_index]

                    if score > best_score:
                        best_score = score
                        # Extract answer string using offsets
                        start_char = offsets[start_index][0]
                        end_char = offsets[end_index][1]
                        best_answer = context_text[start_char:end_char]

        predictions[example_id] = best_answer

    return predictions


def run_training():
    """
    Orchestrates the 5-fold cross-validation training and ensemble inference.
    """
    config = Config()
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    # Load and process data (cached)
    print("Loading and processing data...")
    train_features, val_features, test_features = load_and_process_data(
        config, tokenizer, load_cached_data=True
    )

    # Prepare Test Loader (used for inference in every fold)
    test_dataset = QADataset(test_features, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize arrays to store ensemble logits
    ensemble_start_logits = np.zeros((len(test_features), config.max_length))
    ensemble_end_logits = np.zeros((len(test_features), config.max_length))

    # Loop through folds
    for fold in range(config.n_folds):
        print(f"\n=== Training Fold {fold + 1}/{config.n_folds} ===")

        # Create train/val splits for this fold
        fold_train_data = train_features[train_features["fold"] != fold].reset_index(
            drop=True
        )
        fold_val_data = train_features[train_features["fold"] == fold].reset_index(
            drop=True
        )

        train_dataset = QADataset(fold_train_data, is_test=False)
        val_dataset = QADataset(fold_val_data, is_test=False)

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Initialize model
        model = QAModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_train_steps = len(train_loader) * config.epochs
        num_warmup_steps = int(num_train_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        best_val_loss = float("inf")
        best_model_state = None

        # Training loop
        for epoch in range(config.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device
            )
            val_loss = validate(model, val_loader, device)
            print(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

            # Simple early stopping / model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        # Load best model for inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Run inference on test set
        print(f"Running inference for Fold {fold + 1}...")
        start_logits, end_logits = predict(model, test_loader, device)

        # Accumulate logits for ensemble
        ensemble_start_logits += start_logits
        ensemble_end_logits += end_logits

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # Average logits across all folds
    ensemble_start_logits /= config.n_folds
    ensemble_end_logits /= config.n_folds

    # Post-process to get final strings
    print("\nGenerating final submission...")
    predictions = post_process_predictions(
        test_features, ensemble_start_logits, ensemble_end_logits
    )

    # Create submission file
    sample_sub = pd.read_csv(config.sample_submission_path)

    final_preds = []
    for pid in sample_sub["id"]:
        # Default to empty string if ID not found (should not happen)
        pred_str = predictions.get(pid, "")
        final_preds.append(pred_str)

    sample_sub["PredictionString"] = final_preds

    # Save to CSV (pandas handles quoting automatically)
    sample_sub.to_csv(config.submission_file, index=False)
    print(f"Submission saved to {config.submission_file}")
