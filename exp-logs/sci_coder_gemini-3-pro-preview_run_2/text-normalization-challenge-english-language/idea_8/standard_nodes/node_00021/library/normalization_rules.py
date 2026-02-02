import re
from library.config import cfg

# ==========================================
# 1. Core Number Conversion Logic
# ==========================================


class IntegerToWords:
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
            "quintillion",
            "sextillion",
            "septillion",
            "octillion",
            "nonillion",
            "decillion",
        ]

    def _convert_chunk(self, n):
        """Converts a number < 1000 to words."""
        parts = []
        if n >= 100:
            parts.append(self.ones[n // 100])
            parts.append("hundred")
            n %= 100

        if n >= 20:
            parts.append(self.tens[n // 10])
            n %= 10

        if n >= 10:
            parts.append(self.teens[n - 10])
            n = 0
        elif n > 0:
            parts.append(self.ones[n])

        return parts

    def convert(self, n):
        """Converts an arbitrary integer to words."""
        if n == 0:
            return "zero"

        # Handle negative numbers
        prefix = ""
        if n < 0:
            prefix = "minus "
            n = abs(n)

        parts = []
        chunk_idx = 0

        while n > 0:
            chunk = n % 1000
            if chunk > 0:
                chunk_words = self._convert_chunk(chunk)
                if self.thousands[chunk_idx]:
                    chunk_words.append(self.thousands[chunk_idx])
                parts = chunk_words + parts

            n //= 1000
            chunk_idx += 1

        return prefix + " ".join(parts)


# Instantiate the converter
_converter = IntegerToWords()


def _make_ordinal(words):
    """Converts the last word of a cardinal string to ordinal."""
    tokens = words.split()
    if not tokens:
        return words

    last = tokens[-1]

    # Morphological rules for ordinal suffixes
    replacements = {
        "one": "first",
        "two": "second",
        "three": "third",
        "five": "fifth",
        "eight": "eighth",
        "nine": "ninth",
        "twelve": "twelfth",
    }

    if last in replacements:
        tokens[-1] = replacements[last]
    elif last.endswith("y"):
        tokens[-1] = last[:-1] + "ieth"
    else:
        tokens[-1] = last + "th"

    return " ".join(tokens)


# ==========================================
# 2. Semiotic Class Expanders
# ==========================================


def expand_cardinal(text):
    """
    Expands CARDINAL tokens (e.g., "1,234" -> "one thousand two hundred thirty four").
    """
    # Remove commas and other non-numeric formatting (except minus)
    clean_text = re.sub(r"[^\d-]", "", text)

    try:
        # Check if it's purely numeric
        if not clean_text or clean_text == "-":
            return text

        val = int(clean_text)
        return _converter.convert(val)
    except ValueError:
        # Fallback for Roman numerals or mixed text if any slip through
        return text


def expand_ordinal(text):
    """
    Expands ORDINAL tokens (e.g., "1st" -> "first").
    """
    # Remove ordinal suffixes (st, nd, rd, th) case-insensitively
    clean_text = re.sub(r"(st|nd|rd|th)$", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"[^\d-]", "", clean_text)

    try:
        if not clean_text or clean_text == "-":
            return text

        val = int(clean_text)
        cardinal_words = _converter.convert(val)
        return _make_ordinal(cardinal_words)
    except ValueError:
        return text


def expand_decimal(text):
    """
    Expands DECIMAL tokens (e.g., "3.14" -> "three point one four").
    """
    # Standard format: integer part + dot + fractional part
    if "." not in text:
        return expand_cardinal(text)

    parts = text.split(".")
    if len(parts) != 2:
        return text  # Unexpected format

    integer_part, fractional_part = parts

    # Expand integer part
    # Handle empty integer part (e.g., ".5")
    if integer_part == "" or integer_part == "-":
        if integer_part == "-":
            int_words = "minus zero"
        else:
            int_words = ""  # Implicit zero? Usually "point five"
    else:
        int_words = expand_cardinal(integer_part)

    # Expand fractional part (digit by digit)
    frac_words = expand_digit(fractional_part)

    result = []
    if int_words:
        result.append(int_words)
    result.append("point")
    if frac_words:
        result.append(frac_words)

    return " ".join(result)


def expand_digit(text):
    """
    Expands DIGIT tokens (e.g., "2012" -> "two zero one two").
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
        # Ignore non-digits (like hyphens in phone numbers if classified as DIGIT)

    return " ".join(words)


def expand_letters(text):
    """
    Expands LETTERS tokens (e.g., "FBI" -> "f b i", "A&M" -> "a and m").
    """
    # Special symbol mapping
    symbol_map = {"&": "and", "@": "at", "#": "number", "%": "percent"}

    words = []
    # Iterate through characters
    for char in text:
        if char.isalpha():
            words.append(char.lower())
        elif char in symbol_map:
            words.append(symbol_map[char])
        elif char.isdigit():
            # Sometimes letters have digits like "3D"
            words.append(expand_digit(char))
        # Ignore dots, hyphens, etc.

    return " ".join(words)


def expand_plain(text):
    """Returns text as is."""
    return text


def expand_punct(text):
    """Returns text as is."""
    return text


# ==========================================
# 3. Main Dispatcher
# ==========================================


def dispatch_rule(text, label):
    """
    Routes the input text to the appropriate normalization function based on the label.

    Args:
        text (str): The raw token text.
        label (str): The predicted semiotic class.

    Returns:
        str: The normalized text.
    """
    if label == "PLAIN":
        return expand_plain(text)
    elif label == "PUNCT":
        return expand_punct(text)
    elif label == "CARDINAL":
        return expand_cardinal(text)
    elif label == "ORDINAL":
        return expand_ordinal(text)
    elif label == "DIGIT":
        return expand_digit(text)
    elif label == "DECIMAL":
        return expand_decimal(text)
    elif label == "LETTERS":
        return expand_letters(text)
    else:
        # Fallback for unknown classes or if a neural class is accidentally passed here
        # In the hybrid system, this shouldn't happen for neural classes,
        # but safe default is identity.
        return text
