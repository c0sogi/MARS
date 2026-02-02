import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from library.config import Config
from library.data_utils import (
    get_router_dataloader,
    get_generator_dataloader,
    process_router_data,
)
from library.router_model import TokenClassifier, train_router
from library.generator_model import Seq2SeqNormalizer, train_generator
from library.rule_based_norm import apply_rule


def train_router_pipeline(
    epochs: int = Config.ROUTER_EPOCHS,
    lr: float = Config.ROUTER_LR,
    batch_size: int = Config.ROUTER_TRAIN_BATCH_SIZE,
    debug_sample_size: int = None,
    load_cached_data: bool = True,
):
    """
    Orchestrates the training of the Router model.
    """
    Config.set_seed(Config.SEED)

    # Update Config for dynamic control
    if debug_sample_size is not None:
        Config.DEBUG_SAMPLE_SIZE = debug_sample_size

    # Note: The get_router_dataloader function reads batch_size from Config directly.
    # We patch the Config class to respect the argument.
    if batch_size is not None:
        Config.ROUTER_TRAIN_BATCH_SIZE = batch_size

    print(f"\n=== Starting Router Training Pipeline ===")
    print(f"Epochs: {epochs} | LR: {lr} | Batch Size: {Config.ROUTER_TRAIN_BATCH_SIZE}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # Load Data
    train_loader = get_router_dataloader(
        split="train", load_cached_data=load_cached_data
    )
    val_loader = get_router_dataloader(split="val", load_cached_data=load_cached_data)

    # Initialize Model
    model = TokenClassifier()

    # Train
    model = train_router(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=Config.DEVICE,
        save_dir=Config.ROUTER_CHECKPOINT_DIR,
    )

    return model


def train_generator_pipeline(
    epochs: int = Config.GEN_EPOCHS,
    lr: float = Config.GEN_LR,
    batch_size: int = Config.GEN_TRAIN_BATCH_SIZE,
    debug_sample_size: int = None,
    load_cached_data: bool = True,
):
    """
    Orchestrates the training of the Generator model.
    """
    Config.set_seed(Config.SEED)

    if debug_sample_size is not None:
        Config.DEBUG_SAMPLE_SIZE = debug_sample_size

    if batch_size is not None:
        Config.GEN_TRAIN_BATCH_SIZE = batch_size

    print(f"\n=== Starting Generator Training Pipeline ===")
    print(f"Epochs: {epochs} | LR: {lr} | Batch Size: {Config.GEN_TRAIN_BATCH_SIZE}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # Load Data
    train_loader = get_generator_dataloader(
        split="train", load_cached_data=load_cached_data
    )
    val_loader = get_generator_dataloader(
        split="val", load_cached_data=load_cached_data
    )

    # Initialize Model
    model = Seq2SeqNormalizer()

    # Train
    model = train_generator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=Config.DEVICE,
        save_dir=Config.GENERATOR_CHECKPOINT_DIR,
    )

    return model


class InferenceDataset(Dataset):
    """
    Simple dataset for batching generator inputs during inference.
    """

    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, padding=False
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def collate_inference(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {"input_ids": input_ids_padded, "attention_mask": attention_mask_padded}


def generate_submission(
    router_model: TokenClassifier,
    generator_model: Seq2SeqNormalizer,
    load_cached_data: bool = True,
):
    """
    Generates the submission file using the hybrid architecture.
    """
    print("\n=== Starting Submission Generation ===")
    Config.set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Load Test Data
    # We need the dataframe to access raw tokens and IDs
    df_test_grouped = process_router_data(
        split="test", load_cached_data=load_cached_data
    )

    # We also need the dataloader for the Router model
    router_loader = get_router_dataloader(
        split="test", load_cached_data=load_cached_data
    )

    # Tokenizer for alignment
    router_tokenizer = AutoTokenizer.from_pretrained(
        Config.ROUTER_MODEL_NAME, add_prefix_space=True
    )

    # 2. Run Router (Class Prediction)
    print("Running Router on Test Set...")
    router_model.to(device)
    router_model.eval()

    all_token_ids = []
    all_tokens = []
    all_pred_classes = []

    # Iterate through loader and dataframe in sync
    # Note: get_router_dataloader(split='test') has shuffle=False, so order is preserved.
    # df_test_grouped has one row per sentence. router_loader yields batches of sentences.

    df_iter = df_test_grouped.itertuples(index=False)

    with torch.no_grad():
        for batch in tqdm(router_loader, desc="Routing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = router_model(input_ids, attention_mask).logits
            preds = torch.argmax(logits, dim=2).cpu().numpy()

            # Process each sentence in the batch
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                # Get corresponding raw data
                try:
                    row = next(df_iter)
                except StopIteration:
                    break

                raw_tokens = row.tokens
                row_ids = row.token_ids

                # Re-tokenize to align predictions
                # We need word_ids to map subword predictions to word predictions
                encoding = router_tokenizer(
                    raw_tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=Config.ROUTER_MAX_LEN,
                    return_attention_mask=False,
                )
                word_ids = encoding.word_ids()

                # Map predictions
                sentence_preds = preds[i]
                aligned_preds = []

                prev_word_idx = None
                for j, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue

                    # We take the prediction of the first subword
                    if word_idx != prev_word_idx:
                        # Ensure we are within bounds of the raw tokens
                        if word_idx < len(raw_tokens):
                            class_id = sentence_preds[j]
                            class_label = Config.ID2CLASS[class_id]
                            aligned_preds.append(class_label)
                        prev_word_idx = word_idx

                # Fallback: if truncation occurred, fill remaining with PLAIN
                if len(aligned_preds) < len(raw_tokens):
                    aligned_preds.extend(
                        ["PLAIN"] * (len(raw_tokens) - len(aligned_preds))
                    )

                # Store
                all_token_ids.extend(row_ids)
                all_tokens.extend(raw_tokens)
                all_pred_classes.extend(aligned_preds)

    # 3. Hybrid Execution
    print("Executing Normalization Logic...")

    final_results = {}  # Map id -> normalized_text

    # Prepare batch for generator
    gen_inputs = []
    gen_indices = []  # Indices in the flat list to map back

    structured_set = Config.STRUCTURED_CLASSES
    unstructured_set = Config.UNSTRUCTURED_CLASSES

    for idx, (token, cls, tid) in enumerate(
        zip(all_tokens, all_pred_classes, all_token_ids)
    ):
        if cls == "PLAIN" or cls == "PUNCT":
            final_results[tid] = token
        elif cls in structured_set:
            # Deterministic Rule
            final_results[tid] = apply_rule(token, cls)
        elif cls in unstructured_set:
            # Queue for Generator
            # Input format: "[CLASS] raw_text"
            gen_inputs.append(f"[{cls}] {token}")
            gen_indices.append(tid)
        else:
            # Fallback
            final_results[tid] = token

    # 4. Run Generator
    if gen_inputs:
        print(f"Running Generator on {len(gen_inputs)} unstructured tokens...")
        generator_model.to(device)
        generator_model.eval()

        gen_tokenizer = AutoTokenizer.from_pretrained(Config.GENERATOR_MODEL_NAME)
        gen_dataset = InferenceDataset(
            gen_inputs, gen_tokenizer, max_len=Config.GEN_MAX_INPUT_LEN
        )
        gen_loader = DataLoader(
            gen_dataset,
            batch_size=Config.GEN_VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_inference,
        )

        gen_outputs = []
        with torch.no_grad():
            for batch in tqdm(gen_loader, desc="Generating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                generated_ids = generator_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=Config.GEN_MAX_TARGET_LEN,
                )

                decoded = gen_tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                gen_outputs.extend(decoded)

        # Map back
        for tid, output_text in zip(gen_indices, gen_outputs):
            final_results[tid] = output_text
    else:
        print("No unstructured tokens found.")

    # 5. Create Submission File
    print("Saving Submission...")

    # Ensure order matches original test IDs?
    # The submission format requires 'id' and 'after'.
    # We can just create a DataFrame from the dict.

    # To be safe with order, we use the original list of IDs
    submission_data = []
    for tid in all_token_ids:
        submission_data.append({"id": tid, "after": final_results.get(tid, "")})

    df_sub = pd.DataFrame(submission_data)

    # Post-processing: Handle quotes or special chars if needed?
    # The sample submission shows standard text.
    # We assume the models/rules produce clean text.

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_full_training(
    train_router: bool = True,
    train_generator: bool = True,
    do_inference: bool = True,
    debug_sample_size: int = None,
):
    """
    Main entry point to run the full pipeline.
    """
    router_model = None
    generator_model = None

    if train_router:
        router_model = train_router_pipeline(debug_sample_size=debug_sample_size)
    elif do_inference:
        # Load if not training but inferencing
        print("Loading Router from checkpoint...")
        router_model = TokenClassifier.from_pretrained(Config.ROUTER_CHECKPOINT_DIR)

    if train_generator:
        generator_model = train_generator_pipeline(debug_sample_size=debug_sample_size)
    elif do_inference:
        print("Loading Generator from checkpoint...")
        generator_model = Seq2SeqNormalizer.from_pretrained(
            Config.GENERATOR_CHECKPOINT_DIR
        )

    if do_inference:
        if router_model is None or generator_model is None:
            raise ValueError("Models must be trained or loaded for inference.")
        generate_submission(router_model, generator_model)
