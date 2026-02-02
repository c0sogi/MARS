import re
import os
import numpy as np
from library.config import Config


class NumberConverter:
    """
    Helper class to convert numbers to words without external dependencies like num2words.
    Supports Cardinals and Ordinals.
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
        self.thousands = [
            "",
            "thousand",
            "million",
            "billion",
            "trillion",
            "quadrillion",
        ]

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

    def int_to_words(self, n):
        """Converts an integer to cardinal words."""
        if n == 0:
            return "zero"

        words = []
        for i, chunk in enumerate(self._iterate_chunks(n)):
            if chunk > 0:
                chunk_words = self._chunk_to_words(chunk)
                if i > 0:
                    chunk_words += " " + self.thousands[i]
                words.append(chunk_words)

        return " ".join(reversed(words))

    def int_to_ordinal(self, n):
        """Converts an integer to ordinal words."""
        if n == 0:
            return "zeroth"

        # Handle simple cases directly to reuse logic
        cardinal_part = self.int_to_words(n)

        # Logic: Replace the last word of the cardinal representation with its ordinal equivalent
        words = cardinal_part.split()
        last_word = words[-1]

        # Check for special replacements
        ordinal_replacement = self._get_ordinal_replacement(last_word)
        if ordinal_replacement:
            words[-1] = ordinal_replacement
        else:
            # Fallback simple rule (add th), though _get_ordinal_replacement covers most
            words[-1] = last_word + "th"

        return " ".join(words)

    def _iterate_chunks(self, n):
        """Yields chunks of 1000."""
        while n > 0:
            yield n % 1000
            n //= 1000

    def _chunk_to_words(self, n):
        """Converts a number < 1000 to words."""
        words = []
        h = n // 100
        t = n % 100

        if h > 0:
            words.append(self.ones[h] + " hundred")

        if t > 0:
            if t < 10:
                words.append(self.ones[t])
            elif t < 20:
                words.append(self.teens[t - 10])
            else:
                ten = t // 10
                one = t % 10
                words.append(self.tens[ten])
                if one > 0:
                    words.append(self.ones[one])

        return " ".join(words)

    def _get_ordinal_replacement(self, word):
        """Maps specific cardinal words to ordinal words."""
        mapping = {
            "one": "first",
            "two": "second",
            "three": "third",
            "four": "fourth",
            "five": "fifth",
            "six": "sixth",
            "seven": "seventh",
            "eight": "eighth",
            "nine": "ninth",
            "ten": "tenth",
            "eleven": "eleventh",
            "twelve": "twelfth",
            "twenty": "twentieth",
            "thirty": "thirtieth",
            "forty": "fortieth",
            "fifty": "fiftieth",
            "sixty": "sixtieth",
            "seventy": "seventieth",
            "eighty": "eightieth",
            "ninety": "ninetieth",
            "hundred": "hundredth",
            "thousand": "thousandth",
            "million": "millionth",
            "billion": "billionth",
        }
        # Handle teens dynamically if needed, but they usually end in 'teen' -> 'teenth'
        if word in mapping:
            return mapping[word]
        if word.endswith("teen"):
            return word + "th"
        if word.endswith(
            "y"
        ):  # e.g. twenty-one is handled by splitting, but if logic fails
            return word[:-1] + "ieth"
        return word + "th"


# Instantiate global converter
_converter = NumberConverter()


def _clean_number(text):
    """Removes commas and converts to int/float."""
    clean_text = text.replace(",", "")
    try:
        if "." in clean_text:
            return float(clean_text)
        return int(clean_text)
    except ValueError:
        return None


def normalize_cardinal(text):
    """
    Normalizes CARDINAL class.
    Ex: "1,234" -> "one thousand two hundred thirty four"
    """
    # Handle negative
    prefix = ""
    if text.startswith("-"):
        prefix = "minus "
        text = text[1:]

    val = _clean_number(text)
    if val is None:
        return text  # Fallback

    # If float, delegate to decimal logic usually, but CARDINAL implies integer count often.
    # If it has decimal point, treat as decimal.
    if isinstance(val, float):
        return normalize_decimal(text)

    return prefix + _converter.int_to_words(val)


def normalize_ordinal(text):
    """
    Normalizes ORDINAL class.
    Ex: "1st" -> "first"
    """
    # Remove suffix
    match = re.match(r"(-?\d+)(st|nd|rd|th)?", text, re.IGNORECASE)
    if not match:
        # Roman numerals or words? Assuming digits for this strict function
        return text

    number_str = match.group(1)
    val = _clean_number(number_str)

    if val is None:
        return text

    prefix = ""
    if val < 0:
        prefix = "minus "
        val = abs(val)

    return prefix + _converter.int_to_ordinal(val)


def normalize_digit(text):
    """
    Normalizes DIGIT class.
    Ex: "2014" -> "two zero one four"
    """
    digit_map = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    words = []
    for char in text:
        if char in digit_map:
            words.append(digit_map[char])
        else:
            # Keep non-digits as is or ignore? Usually DIGIT class is pure digits.
            # If it's punctuation inside a digit string, we might preserve it?
            # Standard practice: "555-1234" -> "five five five one two three four"
            # We skip punctuation.
            pass

    return " ".join(words)


def normalize_letters(text):
    """
    Normalizes LETTERS class.
    Ex: "USA" -> "u s a"
    """
    # Remove dots? "U.S.A." -> "u s a"
    clean_text = text.replace(".", "")

    # Separate letters and lowercase
    # Note: Some datasets preserve case or use "capital" prefix.
    # Standard normalization for TTS often lowercases individual letters.
    words = [char.lower() for char in clean_text if char.isalpha()]

    # Handle 's handling for plural acronyms e.g. "CDs" -> "c d's"
    # If original ended in 's, we might want "c d's".
    # Simple heuristic: space out all letters.
    return " ".join(words)


def normalize_money(text):
    """
    Normalizes MONEY class.
    Ex: "$3.50" -> "three dollars, fifty cents"
    """
    # Detect currency
    currency_map = {
        "$": ("dollar", "cents"),
        "£": ("pound", "pence"),
        "€": ("euro", "cents"),
        "¥": ("yen", "sen"),
        "USD": ("dollar", "cents"),
        "GBP": ("pound", "pence"),
    }

    currency = ("dollar", "cents")  # Default

    # Strip non-numeric except . and ,
    # Find symbol
    clean_text = text
    for symbol, names in currency_map.items():
        if symbol in text:
            currency = names
            clean_text = text.replace(symbol, "")
            break

    # Clean number
    clean_text = clean_text.replace(",", "")

    try:
        if "." in clean_text:
            parts = clean_text.split(".")
            major_val = int(parts[0]) if parts[0] else 0
            minor_val = int(parts[1]) if parts[1] else 0

            # Pad minor val? e.g. 3.5 -> 50 cents?
            # Usually money is 2 decimals. "3.5" -> 3 dollars 50 cents.
            # "3.05" -> 3 dollars 5 cents.
            if len(parts[1]) == 1:
                minor_val *= 10

            # Truncate to 2 decimals if more?
            if len(parts[1]) > 2:
                minor_val = int(parts[1][:2])

        else:
            major_val = int(clean_text)
            minor_val = 0
    except ValueError:
        return text  # Fallback

    words = []

    # Major unit
    if major_val > 0 or (major_val == 0 and minor_val == 0):
        words.append(_converter.int_to_words(major_val))
        words.append(currency[0] if major_val == 1 else currency[0] + "s")

    # Minor unit
    if minor_val > 0:
        if words:
            words.append(",")  # Pause
        words.append(_converter.int_to_words(minor_val))
        words.append(currency[1])  # Plural/singular for cents? "1 cent", "2 cents"
        if minor_val == 1:
            # Fix plural if needed. "cents" -> "cent"
            if currency[1].endswith("s"):
                words[-1] = currency[1][:-1]
        # else keep plural

    return " ".join(words)


def normalize_decimal(text):
    """
    Normalizes DECIMAL class.
    Ex: "3.14" -> "three point one four"
    """
    clean_text = text.replace(",", "")
    if "." not in clean_text:
        return normalize_cardinal(text)

    parts = clean_text.split(".")

    # Integer part: Cardinal
    try:
        int_part = _converter.int_to_words(int(parts[0]))
    except ValueError:
        int_part = parts[0]  # Fallback

    # Fractional part: Digits
    frac_part = normalize_digit(parts[1])

    return f"{int_part} point {frac_part}"


def normalize_fraction(text):
    """
    Normalizes FRACTION class.
    Ex: "1/2" -> "one half"
    Ex: "3/4" -> "three quarters"
    """
    if "/" not in text:
        return text

    parts = text.split("/")
    if len(parts) != 2:
        return text

    try:
        num = int(parts[0])
        den = int(parts[1])
    except ValueError:
        return text

    num_word = _converter.int_to_words(num)

    # Denominator logic
    if den == 2:
        den_word = "half" if num == 1 else "halves"
    elif den == 4:
        den_word = "quarter" if num == 1 else "quarters"
    else:
        den_word = _converter.int_to_ordinal(den)
        if num > 1:
            den_word += "s"

    return f"{num_word} {den_word}"


def apply_rule(text, class_label):
    """
    Main dispatcher for rule-based normalization.

    Args:
        text (str): The raw token text.
        class_label (str): The predicted class label.

    Returns:
        str: The normalized text.
    """
    # Clean input slightly (strip whitespace)
    text = text.strip()

    if class_label == "PLAIN" or class_label == "PUNCT":
        return text

    elif class_label == "CARDINAL":
        return normalize_cardinal(text)

    elif class_label == "ORDINAL":
        return normalize_ordinal(text)

    elif class_label == "DIGIT":
        return normalize_digit(text)

    elif class_label == "LETTERS":
        return normalize_letters(text)

    elif class_label == "MONEY":
        return normalize_money(text)

    elif class_label == "DECIMAL":
        return normalize_decimal(text)

    elif class_label == "FRACTION":
        return normalize_fraction(text)

    # Fallback for structured classes not covered or if an unstructured class is passed by mistake
    # In the hybrid architecture, unstructured classes should go to the Generator.
    # If they arrive here, we return raw text to be safe.
    return text
