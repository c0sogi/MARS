import re
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from library.utils import clean_text, get_logger

# Configure logger
logger = get_logger("transformations")


class NumberConverter:
    """
    A helper class to convert numbers to their written English form.
    Handles cardinals, ordinals, and basic formatting.
    """

    def __init__(self):
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
        self.orders = ["", "thousand", "million", "billion", "trillion", "quadrillion"]

        self.ordinal_ones = [
            "",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
        ]
        self.ordinal_teens = [
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
        ]
        self.ordinal_tens = [
            "",
            "",
            "twentieth",
            "thirtieth",
            "fortieth",
            "fiftieth",
            "sixtieth",
            "seventieth",
            "eightieth",
            "ninetieth",
        ]

    def int_to_words(self, num: int) -> str:
        """Converts an integer to cardinal words."""
        if num == 0:
            return "zero"

        parts = []
        if num < 0:
            parts.append("minus")
            num = abs(num)

        for i, order in enumerate(self.orders):
            if num == 0:
                break

            chunk = num % 1000
            if chunk != 0:
                chunk_str = self._chunk_to_words(chunk)
                if i > 0:
                    parts.insert(0, f"{chunk_str} {order}")
                else:
                    parts.insert(0, chunk_str)
            num //= 1000

        return ", ".join(parts).strip()

    def _chunk_to_words(self, num: int) -> str:
        parts = []
        hundreds = num // 100
        remainder = num % 100

        if hundreds > 0:
            parts.append(f"{self.ones[hundreds]} hundred")

        if remainder > 0:
            if remainder < 10:
                parts.append(self.ones[remainder])
            elif remainder < 20:
                parts.append(self.teens[remainder - 10])
            else:
                ten = remainder // 10
                one = remainder % 10
                if one > 0:
                    parts.append(f"{self.tens[ten]} {self.ones[one]}")
                else:
                    parts.append(self.tens[ten])

        return " ".join(parts)

    def int_to_ordinal(self, num: int) -> str:
        """Converts an integer to ordinal words."""
        if num == 0:
            return "zeroth"  # Rare but possible

        # Get cardinal for everything except the last part
        cardinal_str = self.int_to_words(num)
        words = cardinal_str.split()
        last_word = words[-1]

        # Replace last word with ordinal
        ordinal_last = self._word_to_ordinal(last_word, num)
        words[-1] = ordinal_last
        return " ".join(words)

    def _word_to_ordinal(self, word: str, num: int) -> str:
        # Handle simple cases based on the number value
        remainder_100 = num % 100
        remainder_10 = num % 10

        if 11 <= remainder_100 <= 19:
            return self.ordinal_teens[remainder_100 - 10]

        if remainder_10 == 0:
            # 10, 20, 30...
            if remainder_100 == 10:
                return "tenth"
            return self.ordinal_tens[remainder_100 // 10]

        if remainder_10 in [1, 2, 3]:
            return self.ordinal_ones[remainder_10]

        # Regular cases (fourth, sixth, etc.)
        # Mapping specific irregular spelling
        if word == "five":
            return "fifth"
        if word == "eight":
            return "eighth"
        if word == "nine":
            return "ninth"
        if word == "twelve":
            return "twelfth"

        return word + "th"


class TransformationRegistry:
    """
    Registry of deterministic transformation functions.
    Supports finding the best transformation for a given input/output pair
    and applying transformations by name.
    """

    def __init__(self):
        self.funcs: Dict[str, Callable[[str], str]] = {}
        self.num_converter = NumberConverter()
        self._register_all()

        # Common mappings
        self.money_map = {"$": "dollar", "€": "euro", "£": "pound", "¥": "yen"}
        self.measure_map = {
            "kg": "kilograms",
            "g": "grams",
            "mg": "milligrams",
            "m": "meters",
            "km": "kilometers",
            "cm": "centimeters",
            "mm": "millimeters",
            "l": "liters",
            "ml": "milliliters",
            "s": "seconds",
            "h": "hours",
            "min": "minutes",
            "hz": "hertz",
            "mhz": "megahertz",
            "v": "volts",
            "a": "amps",
            "w": "watts",
            "%": "percent",
        }

    def register(self, name: str, func: Callable[[str], str]):
        """Registers a transformation function."""
        self.funcs[name] = func

    def apply(self, name: str, text: str) -> str:
        """Applies a named transformation to text."""
        if name not in self.funcs:
            # Fallback to plain if unknown
            return text
        try:
            return self.funcs[name](text)
        except Exception:
            # Fallback on error
            return text

    def find_best_transform(
        self, before: str, after: str, original_class: str = None
    ) -> str:
        """
        Reverse-engineers the transformation label.
        Tries all registered functions. If one matches 'after', returns its name.
        Uses 'original_class' as a heuristic to prioritize search.
        """
        before = clean_text(before)

        # 1. Optimization: Check PLAIN first (most common)
        if before == after:
            return "TRANS_PLAIN"

        # 2. Heuristic search based on class
        priority_prefixes = []
        if original_class:
            if original_class == "DATE":
                priority_prefixes = ["TRANS_DATE"]
            elif original_class == "CARDINAL":
                priority_prefixes = ["TRANS_CARDINAL"]
            elif original_class == "ORDINAL":
                priority_prefixes = ["TRANS_ORDINAL"]
            elif original_class == "MONEY":
                priority_prefixes = ["TRANS_MONEY"]
            elif original_class == "LETTERS":
                priority_prefixes = ["TRANS_LETTERS"]

        # Try priority functions first
        for name, func in self.funcs.items():
            if any(name.startswith(p) for p in priority_prefixes):
                try:
                    if func(before) == after:
                        return name
                except:
                    continue

        # Try all others
        for name, func in self.funcs.items():
            if name == "TRANS_PLAIN" or any(
                name.startswith(p) for p in priority_prefixes
            ):
                continue
            try:
                if func(before) == after:
                    return name
            except:
                continue

        # If no match found, we can't label it deterministically with our current library.
        # We return a special label indicating fallback or failure.
        return "TRANS_PLAIN"  # Fallback to identity

    def _register_all(self):
        """Registers all transformation logic."""

        # 1. Identity
        self.register("TRANS_PLAIN", lambda x: x)

        # 2. Cardinal
        self.register("TRANS_CARDINAL", self._trans_cardinal)

        # 3. Ordinal
        self.register("TRANS_ORDINAL", self._trans_ordinal)

        # 4. Date
        self.register("TRANS_DATE_YEAR", self._trans_date_year)  # 2012 -> twenty twelve
        self.register(
            "TRANS_DATE_FULL", self._trans_date_full
        )  # 2012-02-01 -> the first of february...

        # 5. Letters
        self.register("TRANS_LETTERS", self._trans_letters)  # IBM -> i b m

        # 6. Digits
        self.register("TRANS_DIGITS", self._trans_digits)  # 12 -> one two

        # 7. Money
        self.register("TRANS_MONEY", self._trans_money)

        # 8. Decimal
        self.register("TRANS_DECIMAL", self._trans_decimal)

        # 9. Measure
        self.register("TRANS_MEASURE", self._trans_measure)

        # 10. Verbatim / Symbols
        self.register(
            "TRANS_VERBATIM", lambda x: x
        )  # Often just copy, or specific symbols

    # --- Transformation Implementations ---

    def _trans_cardinal(self, text: str) -> str:
        # Remove commas
        text = text.replace(",", "")
        try:
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                return self.num_converter.int_to_words(int(text))
            # Handle float as cardinal? usually DECIMAL class.
            return text
        except:
            return text

    def _trans_ordinal(self, text: str) -> str:
        # Remove suffix (st, nd, rd, th)
        clean = re.sub(r"(st|nd|rd|th)$", "", text, flags=re.IGNORECASE)
        clean = clean.replace(",", "")
        try:
            return self.num_converter.int_to_ordinal(int(clean))
        except:
            return text

    def _trans_date_year(self, text: str) -> str:
        # 1998 -> nineteen ninety-eight
        # 2000 -> two thousand
        # 2010 -> twenty ten
        if not (text.isdigit() and len(text) == 4):
            return text

        year = int(text)
        if 2000 <= year < 2010:
            return self.num_converter.int_to_words(year)

        prefix = int(text[:2])
        suffix = int(text[2:])

        p_str = self.num_converter.int_to_words(prefix)
        if suffix == 0:
            s_str = "hundred"  # 1900 -> nineteen hundred
        elif suffix < 10:
            s_str = "oh " + self.num_converter.int_to_words(suffix)
        else:
            s_str = self.num_converter.int_to_words(suffix)

        return f"{p_str} {s_str}"

    def _trans_date_full(self, text: str) -> str:
        # Very basic implementation for YYYY-MM-DD
        # In a real scenario, this needs robust parsing
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
        if match:
            y, m, d = match.groups()
            months = [
                "",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ]
            try:
                m_idx = int(m)
                d_val = int(d)
                if 1 <= m_idx <= 12:
                    m_str = months[m_idx]
                    d_str = self.num_converter.int_to_ordinal(d_val)
                    y_str = self._trans_date_year(y)
                    return f"the {d_str} of {m_str} {y_str}"
            except:
                pass
        return text

    def _trans_letters(self, text: str) -> str:
        # U.S.A -> u s a
        # IBM -> i b m
        clean = text.replace(".", "")
        chars = list(clean)
        # Check if we should lowercase (usually yes for normalization tasks)
        # But sometimes case is preserved. Based on "en_train", usually lowercased if it's "LETTERS"
        return " ".join(c.lower() for c in chars)

    def _trans_digits(self, text: str) -> str:
        # 911 -> nine one one
        chars = []
        for c in text:
            if c.isdigit():
                chars.append(self.num_converter.int_to_words(int(c)))
            else:
                chars.append(c)
        return " ".join(chars)

    def _trans_money(self, text: str) -> str:
        # $3.50 -> three dollars, fifty cents
        # £1 -> one pound
        # Remove commas
        text = text.replace(",", "")

        # Find currency symbol
        currency = None
        for sym, name in self.money_map.items():
            if sym in text:
                currency = name
                text = text.replace(sym, "")
                break

        if not currency:
            return text  # Can't handle

        try:
            if "." in text:
                parts = text.split(".")
                major = int(parts[0])
                minor = int(parts[1])

                major_str = self.num_converter.int_to_words(major)
                minor_str = self.num_converter.int_to_words(minor)

                major_unit = currency if major == 1 else currency + "s"
                minor_unit = (
                    "cent" if minor == 1 else "cents"
                )  # Default to cents for all for simplicity
                if currency == "pound":
                    minor_unit = "penny" if minor == 1 else "pence"

                if minor == 0:
                    return f"{major_str} {major_unit}"
                else:
                    return f"{major_str} {major_unit}, {minor_str} {minor_unit}"
            else:
                val = int(text)
                val_str = self.num_converter.int_to_words(val)
                unit = currency if val == 1 else currency + "s"
                return f"{val_str} {unit}"
        except:
            return text

    def _trans_decimal(self, text: str) -> str:
        # 3.14 -> three point one four
        if "." not in text:
            return text
        try:
            parts = text.split(".")
            integer_part = parts[0]
            fractional_part = parts[1]

            int_str = self.num_converter.int_to_words(int(integer_part))
            frac_chars = [
                self.num_converter.int_to_words(int(c)) for c in fractional_part
            ]
            frac_str = " ".join(frac_chars)

            return f"{int_str} point {frac_str}"
        except:
            return text

    def _trans_measure(self, text: str) -> str:
        # 10kg -> ten kilograms
        # Separate number and unit
        match = re.match(r"([\d\.,]+)\s*([a-zA-Z%]+)", text)
        if match:
            num_str, unit_str = match.groups()
            unit_str = unit_str.lower()
            if unit_str in self.measure_map:
                expanded_unit = self.measure_map[unit_str]
                # Process number (could be decimal or int)
                if "." in num_str:
                    clean_num = self._trans_decimal(num_str)
                else:
                    clean_num = self._trans_cardinal(num_str)

                # Singularize if 1? (Usually measures are pluralized except 1, but map has plural)
                # Simple logic: if 1, remove 's' from map result if it ends in s
                if num_str == "1" and expanded_unit.endswith("s"):
                    expanded_unit = expanded_unit[:-1]

                return f"{clean_num} {expanded_unit}"
        return text

    # --- Caching and Batch Processing ---

    def generate_label_mapping(
        self, df: pd.DataFrame, working_dir: str, load_cached_data: bool = True
    ) -> Dict[str, int]:
        """
        Generates a mapping from transformation names to integer IDs.
        Iterates over the dataframe to find used transformations.

        Args:
            df: DataFrame containing 'before', 'after', and optionally 'class'.
            working_dir: Directory to store the cache.
            load_cached_data: Whether to try loading from cache.

        Returns:
            Dictionary mapping label string to int.
        """
        os.makedirs(working_dir, exist_ok=True)
        cache_path = os.path.join(working_dir, "label_mapping.json")

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading label mapping from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

        logger.info("Generating label mapping from data (this may take a while)...")

        unique_labels = set()
        unique_labels.add("TRANS_PLAIN")  # Ensure default exists

        # We iterate to find which transforms are actually useful
        # To save time, we can just register all available functions in the registry
        # But iterating ensures we cover data specifics if we had dynamic logic.
        # Since logic is static, we can just dump the registry keys.
        # However, to be safe and follow the 'Inverse Label Engineering' concept,
        # we should verify which ones appear.

        # Optimization: Just use all keys in registry.
        # This guarantees the model can predict any function we have code for.
        unique_labels.update(self.funcs.keys())

        label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}

        logger.info(f"Generated {len(label_map)} labels.")
        with open(cache_path, "w") as f:
            json.dump(label_map, f, indent=2)

        return label_map
