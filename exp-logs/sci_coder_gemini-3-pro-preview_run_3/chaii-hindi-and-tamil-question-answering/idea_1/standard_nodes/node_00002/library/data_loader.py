import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from library.config import Config
from library.utils import split_sentences


class SparseRetriever:
    """
    Retrieves the most relevant sentence from a context given a question
    using TF-IDF and cosine similarity.
    """

    def __init__(self):
        pass

    def retrieve(self, question, context_sentences):
        """
        Selects the best sentence from the context based on similarity to the question.

        Args:
            question (str): The question text.
            context_sentences (list): List of sentence strings.

        Returns:
            str: The selected sentence.
        """
        if not context_sentences:
            return ""

        # Combine question and sentences into a mini-corpus
        # Index 0 is the Question, Indices 1..N are the Sentences
        corpus = [question] + context_sentences

        try:
            # Fit TF-IDF on this specific sample
            # Use char n-grams to mitigate length bias and improve robustness for multilingual text
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(2, 4)
            ).fit_transform(corpus)
            vectors = vectorizer.toarray()

            query_vec = vectors[0].reshape(1, -1)
            candidate_vecs = vectors[1:]

            # Compute cosine similarity between Question and all Sentences
            similarities = cosine_similarity(query_vec, candidate_vecs).flatten()

            # Get index of the sentence with highest similarity
            best_idx = np.argmax(similarities)
            return context_sentences[best_idx]

        except (ValueError, IndexError):
            # Fallback for empty vocabulary or other edge cases
            return context_sentences[0] if context_sentences else ""


class QADataset(Dataset):
    """
    PyTorch Dataset for Question Answering Token Classification.
    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert encoding lists to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def find_gold_sentence(row):
    """
    Identifies the sentence in the context that contains the answer text.
    """
    context = row["context"]
    answer_text = row["answer_text"]

    sentences = split_sentences(context)

    # Filter sentences that contain the answer string
    candidates = [s for s in sentences if answer_text in s]

    if not candidates:
        return None

    # If multiple sentences contain the answer, we pick the first one.
    return candidates[0]


def process_data(df, tokenizer, mode="train"):
    """
    Converts a pandas DataFrame into a QADataset.

    Args:
        df (pd.DataFrame): The data.
        tokenizer: Hugging Face tokenizer.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        QADataset: The processed dataset.
    """
    questions = []
    sentences = []
    # Store answer texts to generate labels later
    answers = []

    retriever = SparseRetriever()

    # Filter and prepare text pairs
    for _, row in df.iterrows():
        question = row["question"]
        context = row["context"]

        target_sentence = None

        if mode in ["train", "val"]:
            target_sentence = find_gold_sentence(row)
            # Skip if answer cannot be found in any sentence (data noise/split issues)
            if target_sentence is None:
                continue
            answers.append(row["answer_text"])
        else:
            # For inference, retrieve the best sentence
            context_sentences = split_sentences(context)
            target_sentence = retriever.retrieve(question, context_sentences)

        questions.append(question)
        sentences.append(target_sentence)

    # Tokenize
    # We keep offset_mapping to map predictions back to characters if needed
    encodings = tokenizer(
        questions,
        sentences,
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LEN,
        return_offsets_mapping=True,
    )

    labels = None

    if mode in ["train", "val"]:
        labels = []
        offset_mappings = encodings["offset_mapping"]

        for i, offset in enumerate(offset_mappings):
            answer_text = answers[i]
            sentence = sentences[i]

            # Find character indices of answer in the sentence
            start_char = sentence.find(answer_text)
            end_char = start_char + len(answer_text)

            sequence_ids = encodings.sequence_ids(i)

            # Initialize labels with O (0)
            label_ids = [Config.LABELS_TO_IDS["O"]] * len(sequence_ids)

            # Identify the token span for the sentence (sequence_id == 1)
            token_indices = [
                idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not token_indices:
                labels.append(label_ids)
                continue

            found_start = False

            for idx in token_indices:
                # offset[idx] is (start, end) char index in the sentence
                token_start, token_end = offset[idx]

                # Check if token is inside the answer span
                if token_start < end_char and token_end > start_char:
                    if not found_start:
                        label_ids[idx] = Config.LABELS_TO_IDS["B-ANS"]
                        found_start = True
                    else:
                        label_ids[idx] = Config.LABELS_TO_IDS["I-ANS"]

            labels.append(label_ids)

    return QADataset(encodings, labels)


def prepare_data(load_cached_data=True):
    """
    Loads raw data, processes it, and returns PyTorch Datasets.
    Uses caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache = os.path.join(Config.WORKING_DIR, "train_data.pt")
    val_cache = os.path.join(Config.WORKING_DIR, "val_data.pt")
    test_cache = os.path.join(Config.WORKING_DIR, "test_data.pt")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading cached datasets from {Config.WORKING_DIR}...")
            train_dataset = torch.load(train_cache)
            val_dataset = torch.load(val_cache)
            test_dataset = torch.load(test_cache)
            return train_dataset, val_dataset, test_dataset

    # 2. Process from Scratch
    print("Processing datasets from scratch...")

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Apply Debugging Limits
    if Config.DEBUG:
        print(f"Debug mode: Limiting data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process Splits
    train_dataset = process_data(df_train, tokenizer, mode="train")
    val_dataset = process_data(df_val, tokenizer, mode="val")
    test_dataset = process_data(df_test, tokenizer, mode="test")

    # 3. Save to Cache
    print(f"Saving datasets to {Config.WORKING_DIR}...")
    torch.save(train_dataset, train_cache)
    torch.save(val_dataset, val_cache)
    torch.save(test_dataset, test_cache)

    return train_dataset, val_dataset, test_dataset
