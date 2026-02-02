import re
import os
import pandas as pd
import numpy as np
from library.config import Config


class Normalizer:
    """
    A deterministic rule engine for normalizing text based on semiotic classes.
    """

    def __init__(self):
        # Basic Number Vocabulary
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
        self.groups = ["", "thousand", "million", "billion", "trillion"]

        # Ordinal Vocabulary
        self.ordinals_ones = [
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
        self.ordinals_teens = [
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
        self.ordinals_tens = [
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

        # Date/Time Vocabulary
        self.months = [
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

        # Currency Map
        self.currency_map = {
            "$": "dollars",
            "€": "euros",
            "£": "pounds",
            "¥": "yen",
            "USD": "dollars",
            "EUR": "euros",
            "GBP": "pounds",
            "JPY": "yen",
        }
        self.cents_map = {
            "$": "cents",
            "€": "cents",
            "£": "pence",
            "¥": "sen",  # Context dependent, but acceptable default
        }

        # Measure Map
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
            "ms": "milliseconds",
            "Hz": "hertz",
            "kHz": "kilohertz",
            "MHz": "megahertz",
            "GHz": "gigahertz",
            "V": "volts",
            "A": "amperes",
            "W": "watts",
            "J": "joules",
            "%": "percent",
            "in": "inches",
            "ft": "feet",
            "lb": "pounds",
            "oz": "ounces",
            "ha": "hectares",
            "ac": "acres",
            "mi": "miles",
            "mph": "miles per hour",
        }

    def _num2words(self, n):
        """Converts an integer to English words."""
        try:
            n = int(n)
        except ValueError:
            return ""

        if n == 0:
            return "zero"

        words = []

        if n < 0:
            words.append("minus")
            n = abs(n)

        for i, group_name in enumerate(self.groups):
            if n == 0:
                break

            chunk = n % 1000
            if chunk > 0:
                chunk_words = []
                h = chunk // 100
                t = chunk % 100

                if h > 0:
                    chunk_words.append(self.ones[h])
                    chunk_words.append("hundred")

                if t > 0:
                    if t < 10:
                        chunk_words.append(self.ones[t])
                    elif t < 20:
                        chunk_words.append(self.teens[t - 10])
                    else:
                        ten = t // 10
                        one = t % 10
                        chunk_words.append(self.tens[ten])
                        if one > 0:
                            chunk_words.append(self.ones[one])

                if group_name:
                    chunk_words.append(group_name)

                words = chunk_words + words

            n //= 1000

        return " ".join(words)

    def _ordinal2words(self, n):
        """Converts an integer to its ordinal word form."""
        try:
            n = int(n)
        except ValueError:
            return ""

        # Handle simple cases directly for speed
        if 1 <= n <= 9:
            return self.ordinals_ones[n]
        if 10 <= n <= 19:
            return self.ordinals_teens[n - 10]

        # For larger numbers, convert the prefix to cardinal and suffix to ordinal
        cardinal_part = self._num2words(n)
        words = cardinal_part.split()
        last_word = words[-1]

        # Replace last word with ordinal equivalent
        if last_word in self.ones and last_word != "":
            idx = self.ones.index(last_word)
            words[-1] = self.ordinals_ones[idx]
        elif last_word in self.teens:
            idx = self.teens.index(last_word)
            words[-1] = self.ordinals_teens[idx]
        elif last_word in self.tens and last_word != "":
            idx = self.tens.index(last_word)
            words[-1] = self.ordinals_tens[idx]
        elif last_word == "hundred":
            words[-1] = "hundredth"
        elif last_word == "thousand":
            words[-1] = "thousandth"
        elif last_word == "million":
            words[-1] = "millionth"
        elif last_word == "billion":
            words[-1] = "billionth"
        else:
            # Fallback for complex endings like "zero" -> "zeroth" or typos
            words[-1] = last_word + "th"

        return " ".join(words)

    def format_cardinal(self, text):
        # Remove commas
        clean_text = text.replace(",", "")
        if clean_text.isdigit() or (
            clean_text.startswith("-") and clean_text[1:].isdigit()
        ):
            return self._num2words(int(clean_text))
        return text

    def format_ordinal(self, text):
        # Remove "st", "nd", "rd", "th"
        clean_text = re.sub(r"(st|nd|rd|th)$", "", text, flags=re.IGNORECASE)
        clean_text = clean_text.replace(",", "")
        if clean_text.isdigit():
            return self._ordinal2words(int(clean_text))
        # Roman numerals could be here, but usually LETTERS class
        return text

    def format_decimal(self, text):
        # "1.5" -> "one point five"
        parts = text.replace(",", "").split(".")
        if len(parts) == 2:
            integer_part = (
                self._num2words(int(parts[0])) if parts[0] else "point"
            )  # Handle .5
            fractional_part_digits = list(parts[1])
            fractional_words = [self._num2words(int(d)) for d in fractional_part_digits]

            if parts[0] == "":
                return f"point {' '.join(fractional_words)}"
            return f"{integer_part} point {' '.join(fractional_words)}"
        return text

    def format_money(self, text):
        # "$3.16" -> "three dollars, sixteen cents"
        # Find currency symbol
        currency = None
        amount_str = text

        for sym in self.currency_map:
            if sym in text:
                currency = sym
                amount_str = text.replace(sym, "")
                break

        if not currency:
            # Fallback if no symbol found but class is MONEY
            # Assume generic number or look for suffix (not implemented for brevity)
            if re.match(r"^\d+(\.\d+)?$", text.replace(",", "")):
                return self.format_decimal(text)
            return text

        # Clean amount
        amount_str = amount_str.replace(",", "")

        try:
            if "." in amount_str:
                parts = amount_str.split(".")
                major = int(parts[0]) if parts[0] else 0
                minor = int(parts[1]) if parts[1] else 0

                # Handle "3.5" as 50 cents? Usually money is 2 decimals.
                # If "3.5", it's 5 cents or 50? In text norm, usually explicit.
                # If len is 1, treat as tens? e.g. .5 -> 50 cents.
                if len(parts[1]) == 1:
                    minor *= 10

                major_word = self._num2words(major)
                minor_word = self._num2words(minor)

                currency_name = self.currency_map.get(currency, "dollars")
                cents_name = self.cents_map.get(currency, "cents")

                if major == 1:
                    currency_name = currency_name[:-1]  # singular
                if minor == 1:
                    cents_name = cents_name[:-1]  # singular

                res = []
                if major > 0 or minor == 0:
                    res.append(f"{major_word} {currency_name}")
                if minor > 0:
                    res.append(f"{minor_word} {cents_name}")

                return ", ".join(res) if len(res) > 1 else res[0]
            else:
                major = int(amount_str)
                major_word = self._num2words(major)
                currency_name = self.currency_map.get(currency, "dollars")
                if major == 1:
                    currency_name = currency_name[:-1]
                return f"{major_word} {currency_name}"
        except ValueError:
            return text

    def format_date(self, text):
        # 2012 -> twenty twelve
        # 1990 -> nineteen ninety
        # 2000-2009 -> two thousand X
        if re.match(r"^\d{4}$", text):
            year = int(text)
            if 2000 <= year <= 2009:
                return self._num2words(year)
            else:
                century = year // 100
                decade = year % 100
                if decade == 0:
                    return f"{self._num2words(century)} hundred"  # 1900 -> nineteen hundred
                # 2010 -> twenty ten
                cen_word = self._num2words(century)
                dec_word = self._num2words(decade)
                # Fix for "05" -> "oh five"
                if decade < 10:
                    dec_word = f"oh {dec_word}"
                return f"{cen_word} {dec_word}"

        # YYYY-MM-DD or DD/MM/YYYY
        # Simple heuristic for "2012-02-14" -> "the fourteenth of february twenty twelve"
        match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
        if match:
            y, m, d = match.groups()
            y_word = self.format_date(y)
            m_word = self.months[int(m)] if 1 <= int(m) <= 12 else ""
            d_word = self._ordinal2words(int(d))
            return f"the {d_word} of {m_word} {y_word}"

        return text

    def format_time(self, text):
        # 12:30 -> twelve thirty
        parts = re.split(r"[:\.]", text)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            h = int(parts[0])
            m = int(parts[1])
            h_word = self._num2words(h)

            if m == 0:
                return f"{h_word} o'clock"

            if m < 10:
                m_word = f"oh {self._num2words(m)}"
            else:
                m_word = self._num2words(m)

            return f"{h_word} {m_word}"
        return text

    def format_measure(self, text):
        # "5kg" or "5 kg"
        # Split number and alpha
        match = re.match(r"(-?[\d\.,]+)\s*([a-zA-Z%]+)$", text)
        if match:
            num_str, unit = match.groups()
            # Normalize number
            if "." in num_str:
                num_word = self.format_decimal(num_str)
                is_plural = num_str != "1"
            else:
                num_word = self.format_cardinal(num_str)
                is_plural = num_str != "1"

            unit_word = self.measure_map.get(unit, unit)
            # Simple pluralization
            if is_plural and unit in self.measure_map and not unit_word.endswith("s"):
                # Check if already plural in map (most are) or needs s
                # My map has plurals. Singular logic:
                if unit_word.endswith("s"):
                    pass  # assume plural default
                else:
                    unit_word += "s"
            elif (
                not is_plural
                and unit_word.endswith("s")
                and unit != "s"
                and unit != "ms"
            ):
                # Naive singularization
                unit_word = unit_word[:-1]

            return f"{num_word} {unit_word}"
        return text

    def format_letters(self, text):
        # "USA" -> "u s a"
        # "A.B.C." -> "a b c"
        clean = text.replace(".", "")
        return " ".join(list(clean.lower()))

    def format_digit(self, text):
        # "123" -> "one two three"
        return " ".join([self._num2words(int(d)) for d in text if d.isdigit()])

    def format_fraction(self, text):
        # "1/2" -> "one half"
        parts = text.split("/")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            num = int(parts[0])
            den = int(parts[1])
            num_word = self._num2words(num)
            den_word = self._ordinal2words(den)
            if den == 2:
                den_word = "half"
            if den == 4:
                den_word = "quarter"

            if num > 1:
                if den == 2:
                    den_word = "halves"
                else:
                    den_word += "s"

            return f"{num_word} {den_word}"
        return text

    def format_electronic(self, text):
        # "foo.com" -> "foo dot com"
        t = text.lower()
        t = (
            t.replace(".", " dot ")
            .replace("@", " at ")
            .replace("/", " slash ")
            .replace("-", " dash ")
        )
        # Remove double spaces
        return " ".join(t.split())

    def format_telephone(self, text):
        # "555-1234" -> "five five five one two three four"
        # Usually silence punctuation
        digits = re.findall(r"\d", text)
        return " ".join([self._num2words(int(d)) for d in digits])

    def normalize(self, text, label):
        """
        Main dispatch function.
        """
        if label == "PLAIN":
            return text
        elif label == "PUNCT":
            return text
        elif label == "DATE":
            return self.format_date(text)
        elif label == "CARDINAL":
            return self.format_cardinal(text)
        elif label == "LETTERS":
            return self.format_letters(text)
        elif label == "VERBATIM":
            return text  # Often just symbols
        elif label == "MEASURE":
            return self.format_measure(text)
        elif label == "ORDINAL":
            return self.format_ordinal(text)
        elif label == "DECIMAL":
            return self.format_decimal(text)
        elif label == "MONEY":
            return self.format_money(text)
        elif label == "DIGIT":
            return self.format_digit(text)
        elif label == "ELECTRONIC":
            return self.format_electronic(text)
        elif label == "TELEPHONE":
            return self.format_telephone(text)
        elif label == "TIME":
            return self.format_time(text)
        elif label == "FRACTION":
            return self.format_fraction(text)
        elif label == "ADDRESS":
            # Address is highly complex, treat as sequence of Cardinals/Letters usually
            # Best effort: convert numbers, keep text
            return self.format_cardinal(text)  # Fallback
        else:
            return text


def apply_normalization(df, load_cached_data=True):
    """
    Applies normalization to a dataframe.
    Implements caching as required by the prompt.
    """
    CACHE_PATH = os.path.join(Config.WORKING_DIR, "normalization_cache.parquet")

    if load_cached_data and os.path.exists(CACHE_PATH):
        print(f"Loading cached normalization from {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    print("Computing normalization...")
    norm = Normalizer()

    # Ensure id column exists for submission
    if "id" not in df.columns and "sentence_id" in df.columns:
        df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)

    # We assume the dataframe has 'before' and predicted 'class' (or we use ground truth for dev)
    # If 'class' is missing (test set), this function might not be directly applicable without a model prediction step first.
    # This function assumes 'class' column exists.

    if "class" in df.columns:
        # Vectorized application is hard with complex rules, use apply
        # For speed, we might want to optimize, but Python loop is baseline
        df["after_predicted"] = df.apply(
            lambda row: norm.normalize(row["before"], row["class"]), axis=1
        )

    # Save cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_parquet(CACHE_PATH)

    return df
