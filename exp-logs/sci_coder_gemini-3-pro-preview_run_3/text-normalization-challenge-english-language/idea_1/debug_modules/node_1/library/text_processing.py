import os
import re
import logging
import pandas as pd
import numpy as np
from library.utils import setup_logger


class RegexTransducer:
    """
    A heuristic rule-based system for normalizing text patterns such as numbers,
    money, and dates. Used as a fallback for OOV tokens.
    """

    def __init__(self):
        self.logger = setup_logger("RegexTransducer")

        # Compiled Regex Patterns
        # Money: Matches $1, $1.50, $1,000.00
        self.re_money = re.compile(r"^\$([0-9,]+)(\.([0-9]{2}))?$")
        # Decimal: Matches 1.5, 3.1415 (requires at least one digit after dot)
        self.re_decimal = re.compile(r"^([0-9,]+)\.([0-9]+)$")
        # Ordinal: Matches 1st, 2nd, 3rd, 104th
        self.re_ordinal = re.compile(r"^([0-9,]+)(st|nd|rd|th)$", re.IGNORECASE)
        # Cardinal: Matches integers 100, -5, 1,000
        self.re_cardinal = re.compile(r"^-?[0-9,]+$")

        # Static data for number conversion
        self.ones = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        self.teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        self.tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]
        self.thousands = ["", "thousand", "million", "billion", "trillion"]

    def normalize(self, token):
        """
        Attempts to normalize a token using regex rules.
        Returns the normalized string if a match is found, otherwise returns None.
        """
        # Strip commas for pattern matching logic where appropriate,
        # but we handle them inside specific converters usually.

        # Money ($3.16 -> three dollars, sixteen cents)
        if self.re_money.match(token):
            try:
                return self._convert_money(token)
            except:
                pass

        # Ordinal (1st -> first)
        if self.re_ordinal.match(token):
            try:
                return self._convert_ordinal(token)
            except:
                pass

        # Decimal (3.14 -> three point one four)
        if self.re_decimal.match(token):
            try:
                return self._convert_decimal(token)
            except:
                pass

        # Cardinal (123 -> one hundred twenty-three)
        if self.re_cardinal.match(token):
            try:
                return self._convert_cardinal(token)
            except:
                pass

        return None

    def _num_to_words(self, n):
        """Converts an integer to its English word representation."""
        if n == 0:
            return "zero"

        words = []

        # Handle negative
        if n < 0:
            words.append("minus")
            n = -n

        def chunk_to_words(num):
            chunk_words = []
            if num >= 100:
                chunk_words.append(self.ones[num // 100])
                chunk_words.append("hundred")
                num %= 100

            if num >= 20:
                chunk_words.append(self.tens[num // 10])
                num %= 10
                if num > 0:
                    chunk_words.append(self.ones[num])
            elif num >= 10:
                chunk_words.append(self.teens[num - 10])
            elif num > 0:
                chunk_words.append(self.ones[num])
            return chunk_words

        chunk_count = 0
        while n > 0:
            chunk = n % 1000
            if chunk > 0:
                chunk_str = chunk_to_words(chunk)
                if chunk_count > 0:
                    chunk_str.append(self.thousands[chunk_count])
                words = chunk_str + words
            n //= 1000
            chunk_count += 1

        return " ".join(words)

    def _convert_cardinal(self, token):
        # Remove commas
        clean_token = token.replace(",", "")
        n = int(clean_token)
        return self._num_to_words(n)

    def _convert_ordinal(self, token):
        match = self.re_ordinal.match(token)
        number_str = match.group(1).replace(",", "")
        n = int(number_str)

        # Get cardinal words
        words = self._num_to_words(n).split()
        last_word = words[-1]

        # Transform last word to ordinal
        ordinal_map = {
            "one": "first",
            "two": "second",
            "three": "third",
            "five": "fifth",
            "eight": "eighth",
            "nine": "ninth",
            "twelve": "twelfth",
        }

        if last_word in ordinal_map:
            words[-1] = ordinal_map[last_word]
        elif last_word.endswith("y"):
            words[-1] = last_word[:-1] + "ieth"
        else:
            words[-1] = last_word + "th"

        return " ".join(words)

    def _convert_decimal(self, token):
        match = self.re_decimal.match(token)
        integer_part = match.group(1).replace(",", "")
        fractional_part = match.group(2)

        int_words = self._num_to_words(int(integer_part))

        frac_words = []
        for digit in fractional_part:
            if digit == "0":
                frac_words.append("zero")
            else:
                frac_words.append(self._num_to_words(int(digit)))

        return f"{int_words} point {' '.join(frac_words)}"

    def _convert_money(self, token):
        match = self.re_money.match(token)
        integer_part = match.group(1).replace(",", "")
        cents_part = match.group(3)  # Can be None

        n_dollars = int(integer_part)
        dollar_words = self._num_to_words(n_dollars)
        currency = "dollar" if n_dollars == 1 else "dollars"

        res = f"{dollar_words} {currency}"

        if cents_part:
            n_cents = int(cents_part)
            if n_cents > 0:
                cent_words = self._num_to_words(n_cents)
                cent_currency = "cent" if n_cents == 1 else "cents"
                res += f", {cent_words} {cent_currency}"

        return res


class TextNormalizer:
    """
    Hierarchical Text Normalization Model.
    Architecture:
    1. L2 Cache (Bigram): (prev_token, token) -> normalized_text
    2. L1 Cache (Unigram): token -> normalized_text
    3. Regex Transducer: heuristic rules
    4. Identity: token -> token
    """

    def __init__(self, cache_dir="./working/idea_1"):
        self.cache_dir = cache_dir
        self.logger = setup_logger("TextNormalizer")
        self.regex_engine = RegexTransducer()

        self.l1_lookup = {}  # Unigram
        self.l2_lookup = {}  # Bigram

    def fit(self, train_path, load_cached_data=True):
        """
        Builds lookup tables from training data.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        l1_path = os.path.join(self.cache_dir, "l1_stats.parquet")
        l2_path = os.path.join(self.cache_dir, "l2_stats.parquet")

        if load_cached_data and os.path.exists(l1_path) and os.path.exists(l2_path):
            self.logger.info("Loading cached statistics...")
            df_l1 = pd.read_parquet(l1_path)
            df_l2 = pd.read_parquet(l2_path)

            # Convert to dictionary for fast O(1) lookup
            # L1: token -> after
            self.l1_lookup = dict(zip(df_l1["before"], df_l1["after"]))

            # L2: (prev, token) -> after
            # We create a composite key string for storage/retrieval or use tuple
            # Using tuple in memory
            self.l2_lookup = dict(
                zip(zip(df_l2["prev_before"], df_l2["before"]), df_l2["after"])
            )

            self.logger.info(
                f"Loaded {len(self.l1_lookup)} unigrams and {len(self.l2_lookup)} bigrams."
            )
            return

        self.logger.info("Computing statistics from training data...")
        # Load training data
        df = pd.read_parquet(train_path)

        # 1. Unigram Statistics (L1)
        # Find most frequent 'after' for each 'before'
        self.logger.info("Aggregating Unigrams...")
        l1_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        # Sort by count desc and drop duplicates to keep top 1
        l1_best = l1_counts.sort_values(
            ["before", "count"], ascending=[True, False]
        ).drop_duplicates(["before"])
        self.l1_lookup = dict(zip(l1_best["before"], l1_best["after"]))

        # Save L1
        l1_best[["before", "after"]].to_parquet(l1_path, index=False)

        # 2. Bigram Statistics (L2)
        # We need previous token. Since data is grouped by sentence_id, we can shift.
        self.logger.info("Aggregating Bigrams...")
        df["prev_before"] = (
            df.groupby("sentence_id")["before"].shift(1).fillna("<START>")
        )

        l2_counts = (
            df.groupby(["prev_before", "before", "after"])
            .size()
            .reset_index(name="count")
        )

        # Filter low frequency bigrams to save memory and reduce noise (e.g., count > 1)
        # However, for this task, max accuracy is key, so we keep all unless memory is an issue.
        # Given 220GB RAM, we keep all.
        l2_best = l2_counts.sort_values(
            ["prev_before", "before", "count"], ascending=[True, True, False]
        )
        l2_best = l2_best.drop_duplicates(["prev_before", "before"])

        # Build Dict
        self.l2_lookup = dict(
            zip(zip(l2_best["prev_before"], l2_best["before"]), l2_best["after"])
        )

        # Save L2
        l2_best[["prev_before", "before", "after"]].to_parquet(l2_path, index=False)

        self.logger.info(
            f"Training complete. Learned {len(self.l1_lookup)} unigrams and {len(self.l2_lookup)} bigrams."
        )

    def predict_token(self, token, prev_token="<START>"):
        """
        Predicts normalized text for a single token using the hierarchy.
        """
        # 1. L2 Lookup
        l2_key = (prev_token, token)
        if l2_key in self.l2_lookup:
            return self.l2_lookup[l2_key]

        # 2. L1 Lookup
        if token in self.l1_lookup:
            return self.l1_lookup[token]

        # 3. Regex Fallback
        regex_pred = self.regex_engine.normalize(token)
        if regex_pred is not None:
            return regex_pred

        # 4. Identity
        return token

    def validate(self, val_path):
        """
        Validates the model on the validation set and prints accuracy.
        """
        self.logger.info("Starting validation...")
        df_val = pd.read_parquet(val_path)

        # Prepare context
        df_val["prev_before"] = (
            df_val.groupby("sentence_id")["before"].shift(1).fillna("<START>")
        )

        # Predict
        # Using list comprehension for speed over apply
        tokens = df_val["before"].tolist()
        prev_tokens = df_val["prev_before"].tolist()
        targets = df_val["after"].tolist()

        preds = []
        for t, pt in zip(tokens, prev_tokens):
            preds.append(self.predict_token(t, pt))

        # Calculate Accuracy
        correct = sum(p == t for p, t in zip(preds, targets))
        total = len(targets)
        acc = correct / total

        print(f"Validation Accuracy: {acc}")
        return acc

    def generate_submission(self, test_path, output_file):
        """
        Generates predictions for the test set and saves to CSV.
        """
        self.logger.info("Generating submission...")
        df_test = pd.read_parquet(test_path)

        # Prepare context
        df_test["prev_before"] = (
            df_test.groupby("sentence_id")["before"].shift(1).fillna("<START>")
        )

        tokens = df_test["before"].tolist()
        prev_tokens = df_test["prev_before"].tolist()
        ids = df_test["id"].tolist()

        preds = []
        for t, pt in zip(tokens, prev_tokens):
            preds.append(self.predict_token(t, pt))

        # Create submission DataFrame
        sub_df = pd.DataFrame({"id": ids, "after": preds})

        # Quote the 'after' column to handle special characters/delimiters correctly
        # Pandas to_csv handles quoting automatically, but we ensure the format matches sample.
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sub_df.to_csv(
            output_file, index=False, quoting=1
        )  # quote all non-numeric? usually minimal quoting is preferred but sample has quotes.
        # Default quoting in pandas is usually fine.

        self.logger.info(f"Submission saved to {output_file}")
