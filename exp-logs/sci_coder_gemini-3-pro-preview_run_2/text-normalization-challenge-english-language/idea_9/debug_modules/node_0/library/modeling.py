import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Import from provided library
from library.config import Config
from library.utils import seed_everything, compute_accuracy
from library.dataset import (
    prepare_router_data,
    prepare_generator_data,
    TextNormalizationRouterDataset,
    TextNormalizationGeneratorDataset,
)
from library.normalization_rules import apply_rule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RouterModel(nn.Module):
    """
    Token Classification Model (DeBERTa-v3) for semantic class prediction.
    """

    def __init__(
        self, model_name=Config.ROUTER_MODEL_NAME, num_labels=Config.NUM_LABELS
    ):
        super(RouterModel, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, num_labels=Config.NUM_LABELS):
        instance = cls(model_name=path, num_labels=num_labels)
        return instance


class GeneratorModel(nn.Module):
    """
    Seq2Seq Model (ByT5) for context-aware text normalization.
    """

    def __init__(self, model_name=Config.GENERATOR_MODEL_NAME):
        super(GeneratorModel, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def generate(
        self, input_ids, attention_mask, max_length=Config.GENERATOR_MAX_TARGET_LEN
    ):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
        )

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path):
        instance = cls(model_name=path)
        return instance


def train_router(
    epochs=Config.ROUTER_EPOCHS,
    batch_size=Config.ROUTER_BATCH_SIZE,
    lr=Config.ROUTER_LR,
    load_cached_data=True,
):
    """
    Trains the Router model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Initializing Router Training...")

    # 1. Prepare Data
    train_df = prepare_router_data("train", load_cached_data=load_cached_data)
    val_df = prepare_router_data("val", load_cached_data=load_cached_data)

    tokenizer = AutoTokenizer.from_pretrained(Config.ROUTER_MODEL_NAME)

    train_dataset = TextNormalizationRouterDataset(train_df, tokenizer, is_test=False)
    val_dataset = TextNormalizationRouterDataset(val_df, tokenizer, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = RouterModel().to(device)
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=Config.ROUTER_WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 3. Training Loop
    best_val_accuracy = 0.0
    patience_counter = 0
    save_path = os.path.join(Config.CHECKPOINT_DIR, "router_best")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Router Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Router"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)

                # Filter out -100 labels
                active_loss = labels.view(-1) != -100
                active_logits = preds.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]

                val_preds.extend(active_logits.cpu().numpy())
                val_labels.extend(active_labels.cpu().numpy())

        val_accuracy = np.mean(np.array(val_preds) == np.array(val_labels))
        logger.info(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss}, Val Accuracy = {val_accuracy}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best accuracy! Saving model to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.ROUTER_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    return best_val_accuracy


def train_generator(
    epochs=Config.GENERATOR_EPOCHS,
    batch_size=Config.GENERATOR_BATCH_SIZE,
    lr=Config.GENERATOR_LR,
    load_cached_data=True,
):
    """
    Trains the Generator model (Seq2Seq).
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Initializing Generator Training...")

    # 1. Prepare Data
    # Generator data is already filtered for Path B classes and formatted
    train_df = prepare_generator_data("train", load_cached_data=load_cached_data)
    val_df = prepare_generator_data("val", load_cached_data=load_cached_data)

    if len(train_df) == 0:
        logger.warning(
            "No training data for generator found (check Path B filtering). Skipping training."
        )
        return 0.0

    tokenizer = AutoTokenizer.from_pretrained(Config.GENERATOR_MODEL_NAME)

    train_dataset = TextNormalizationGeneratorDataset(train_df, tokenizer)
    val_dataset = TextNormalizationGeneratorDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = GeneratorModel().to(device)
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=Config.GENERATOR_WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Generator Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Generator"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss}, Val Loss = {avg_val_loss}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best loss! Saving model to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.GENERATOR_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    return best_val_loss


def predict_submission(load_cached_data=True):
    """
    Generates the final submission file.
    1. Runs Router on Test Set.
    2. Routes tokens to Path A (Rules) or Path B (Generator).
    3. Combines results.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 1. Router Inference
    # ==========================================
    logger.info("Loading Router for inference...")
    router_path = os.path.join(Config.CHECKPOINT_DIR, "router_best")
    if not os.path.exists(router_path):
        logger.warning(
            "Router checkpoint not found. Using base model (Expect poor results)."
        )
        router_path = Config.ROUTER_MODEL_NAME

    router_model = RouterModel.from_pretrained(router_path).to(device)
    router_tokenizer = AutoTokenizer.from_pretrained(router_path)
    router_model.eval()

    logger.info("Loading Test Data...")
    # We use prepare_router_data to get the grouped dataframe (tokens list per sentence)
    test_grouped_df = prepare_router_data("test", load_cached_data=load_cached_data)

    # We need to map predictions back to token IDs.
    # The dataset returns input_ids but not word_ids mapping in __getitem__.
    # We will iterate manually over the dataframe to ensure alignment.

    logger.info("Running Router Inference...")
    all_token_ids = []
    all_pred_labels = []
    all_raw_tokens = []
    all_sentence_ids = []

    batch_size = Config.ROUTER_BATCH_SIZE * 2

    # Process in chunks
    for i in tqdm(range(0, len(test_grouped_df), batch_size), desc="Routing"):
        batch_df = test_grouped_df.iloc[i : i + batch_size]
        batch_tokens = batch_df["tokens"].tolist()
        batch_ids = batch_df["token_ids"].tolist()
        batch_sentence_ids = batch_df["sentence_id"].tolist()

        # Tokenize
        encodings = router_tokenizer(
            batch_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=Config.ROUTER_MAX_LEN,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = router_model(input_ids, attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2).cpu().numpy()

        # Align predictions to words
        for idx, (pred_seq, token_list, id_list, s_id) in enumerate(
            zip(preds, batch_tokens, batch_ids, batch_sentence_ids)
        ):
            word_ids = encodings.word_ids(batch_index=idx)

            # Map first subword label to token
            aligned_labels = []
            prev_word_idx = None

            for j, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != prev_word_idx:
                    # New token found
                    label_id = pred_seq[j]
                    label_str = Config.ID2LABEL[label_id]
                    aligned_labels.append(label_str)
                    prev_word_idx = word_idx

            # Safety check: length mismatch (truncation)
            if len(aligned_labels) < len(token_list):
                # Fill remaining with PLAIN
                diff = len(token_list) - len(aligned_labels)
                aligned_labels.extend(["PLAIN"] * diff)
            elif len(aligned_labels) > len(token_list):
                # Should not happen with word_ids logic, but truncate just in case
                aligned_labels = aligned_labels[: len(token_list)]

            all_token_ids.extend(id_list)
            all_pred_labels.extend(aligned_labels)
            all_raw_tokens.extend(token_list)
            all_sentence_ids.extend([s_id] * len(token_list))

    # Create DataFrame with predictions
    pred_df = pd.DataFrame(
        {
            "id": all_token_ids,
            "token": all_raw_tokens,
            "class": all_pred_labels,
            "sentence_id": all_sentence_ids,
        }
    )

    # ==========================================
    # 2. Path A: Deterministic Rules
    # ==========================================
    logger.info("Applying Path A (Deterministic Rules)...")

    # Initialize 'after' column
    pred_df["after"] = pred_df["token"]  # Default copy

    # Mask for Path A
    path_a_mask = pred_df["class"].isin(Config.PATH_A_CLASSES)

    # Apply rules
    # We can use apply, but let's iterate or vectorize where possible.
    # Since rules are python functions, apply is necessary.
    def apply_path_a(row):
        return apply_rule(row["token"], row["class"])

    pred_df.loc[path_a_mask, "after"] = pred_df[path_a_mask].apply(apply_path_a, axis=1)

    # ==========================================
    # 3. Path B: Neural Generator
    # ==========================================
    logger.info("Applying Path B (Neural Generator)...")

    path_b_mask = pred_df["class"].isin(Config.PATH_B_CLASSES)
    path_b_df = pred_df[path_b_mask].copy()

    if len(path_b_df) > 0:
        generator_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best")
        if not os.path.exists(generator_path):
            logger.warning(
                "Generator checkpoint not found. Skipping Path B refinement (using raw text)."
            )
        else:
            generator_model = GeneratorModel.from_pretrained(generator_path).to(device)
            generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)
            generator_model.eval()

            # Construct Inputs using Context
            # We need the full sentence context. Group pred_df by sentence_id
            sentence_map = pred_df.groupby("sentence_id")["token"].apply(list).to_dict()

            input_texts = []
            indices = []

            # Iterate over Path B tokens
            # We need the token index within the sentence to get context.
            # The 'id' column is "sentence_token".

            for row in tqdm(
                path_b_df.itertuples(), total=len(path_b_df), desc="Building Gen Inputs"
            ):
                s_id = row.sentence_id
                # Extract token_id from "123_5" -> 5
                t_id = int(row.id.split("_")[1])
                label = row._3  # class column

                ctx_tokens = sentence_map[s_id]
                seq_len = len(ctx_tokens)

                start = max(0, t_id - Config.CONTEXT_WINDOW_SIZE)
                end = min(seq_len, t_id + Config.CONTEXT_WINDOW_SIZE + 1)

                left_ctx = " ".join(ctx_tokens[start:t_id])
                target_token = ctx_tokens[t_id]
                right_ctx = " ".join(ctx_tokens[t_id + 1 : end])

                input_str = (
                    f"{label} {left_ctx} "
                    f"{Config.TARGET_START_TOKEN} {target_token} {Config.TARGET_END_TOKEN} "
                    f"{right_ctx}"
                )

                input_texts.append(input_str)
                indices.append(row.Index)  # Original dataframe index

            # Batch Inference
            gen_batch_size = Config.GENERATOR_BATCH_SIZE * 4
            generated_outputs = []

            for i in tqdm(
                range(0, len(input_texts), gen_batch_size), desc="Generating"
            ):
                batch_texts = input_texts[i : i + gen_batch_size]

                inputs = generator_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=Config.GENERATOR_MAX_INPUT_LEN,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = generator_model.generate(
                        inputs["input_ids"], inputs["attention_mask"]
                    )

                decoded = generator_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                generated_outputs.extend(decoded)

            # Assign back
            pred_df.loc[indices, "after"] = generated_outputs

    # ==========================================
    # 4. Save Submission
    # ==========================================
    logger.info("Saving Submission...")
    submission = pred_df[["id", "after"]]
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
