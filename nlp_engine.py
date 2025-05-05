"""
@Author : Aymen Brahim Djelloul
version : 0.1
date : 30.04.2025
License : MIT

"""

# IMPORTS
import re
import json
import sys
import difflib
import requests
import unicodedata
import os.path


class NlpEngine:

    def __init__(self, dev_mode: bool = False) -> None:

        # Initialize developer mode
        self.dev_mode = dev_mode

    @staticmethod
    def text_normalization(input_text: str, keep_punctuation: bool = True, lemmatize: bool = True) -> str:
        """
        Aggressively normalize and simplify text input for matching purposes.

        Parameters:
        -----------
        input_text : str
            The raw text to normalize
        keep_punctuation : bool, default=True
            Whether to preserve simple punctuation marks in the output
        lemmatize : bool, default=True
            Whether to apply basic rule-based lemmatization

        Returns:
        --------
        str
            Normalized text string

        Examples:
        ---------
        # >>> text_normalization("Café's are OPEN!! 😊  Visit us at café.com")
        'cafe are open ! ! visit u at cafe . com'

        # >>> text_normalization("I'm running and they're dancing", lemmatize=True)
        'i am run and they are danc'
        """
        try:
            if not input_text or not isinstance(input_text, str):
                return ""

            # 1. Lowercase
            text = input_text.lower()

            # 2. Handle common contractions
            contractions = {
                "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
                "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
                "what's": "what is", "who's": "who is", "where's": "where is",
                "don't": "do not", "doesn't": "does not", "didn't": "did not",
                "won't": "will not", "can't": "cannot", "couldn't": "could not",
                "shouldn't": "should not", "wouldn't": "would not", "isn't": "is not",
                "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
                "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
                "we'll": "we will", "they'll": "they will", "i'd": "i would", "you'd": "you would",
                "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
                "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
                "let's": "let us", "yea": "yes", "yaa": "yes", "ya": "yes", "noo": "no",
                "nah": "no", "yeah": "yes"
            }

            for contraction, expansion in contractions.items():
                text = re.sub(r'\b' + contraction + r'\b', expansion, text)

            # 3. Remove accents
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))

            # 4. Remove emojis & symbols (anything not ASCII printable)
            text = re.sub(r'[^\x00-\x7F]+', '', text)

            # 5. Handle punctuation
            if keep_punctuation:
                # Convert punctuation to spaces except for simple ones
                text = re.sub(r'[^\w\s.?!,;:-]', ' ', text)
                # Add spaces around remaining punctuation for better tokenization
                text = re.sub(r'([.?!,;:-])', r' \1 ', text)
            else:
                # Remove all punctuation
                text = re.sub(r'[^\w\s]', ' ', text)

            # 6. Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # 7. Enhanced rule-based lemmatization
            if lemmatize:
                words = text.split()
                lemmas = []

                for word in words:
                    # Skip very short words or punctuation
                    if len(word) <= 1 or not any(c.isalpha() for c in word):
                        lemmas.append(word)
                        continue

                    # Handle plurals and verb forms
                    if word.endswith("ies") and len(word) > 4:
                        word = word[:-3] + "y"  # categories → category
                    elif word.endswith("es") and len(word) > 3:
                        word = word[:-2]  # fixes → fix
                    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                        word = word[:-1]  # cars → car, but not pass → pas

                    # Handle verb forms
                    if word.endswith("ing") and len(word) > 5:
                        if word[-4] == word[-5]:  # running → run (double consonant)
                            word = word[:-4]
                        else:
                            word = word[:-3]  # dancing → danc
                        # Try to restore 'e' if it was likely removed
                        if word.endswith(('at', 'iv', 'iz', 'us', 'in', 'ov')):
                            word = word + 'e'  # racing → rac → race

                    elif word.endswith("ed") and len(word) > 4:
                        if word[-3] == word[-4] and word[-3] not in 'aeiou':  # stopped → stop
                            word = word[:-3]
                        else:
                            word = word[:-2]  # jumped → jump
                        # Try to restore 'e' if it was likely removed
                        if word.endswith(('at', 'iv', 'iz', 'us', 'in', 'ov')):
                            word = word + 'e'  # raced → rac → race

                    # Handle common suffixes
                    elif word.endswith("ly") and len(word) > 4:
                        word = word[:-2]  # quickly → quick
                    elif word.endswith("ment") and len(word) > 6:
                        word = word[:-4]  # development → develop
                    elif word.endswith("ness") and len(word) > 5:
                        word = word[:-4]  # happiness → happy

                    lemmas.append(word)

                text = " ".join(lemmas)

            return text

        except Exception:
            print(f"Warning: Normalization failed for '{input_text}'")
            print(traceback.format_exc())
            return input_text  # Return original text rather than failing completely

    @staticmethod
    def is_question(normalized_text: str) -> bool:
        """
        Detect whether the input is a question using:
        - Question mark
        - Question keywords
        - Exclusion of aggressive/non-question phrases
        """
        text = normalized_text.lower().strip()

        # Fast path: question mark check
        if text[-1] == "?":
            return True

        # Quick ignore: aggressive or rhetorical phrases
        if any(phrase in text for phrase in {
            "what the hell", "what the fuck", "what is this shit",
            "what a day", "what a mess", "how dare you"
        }):
            return False

        # Question-like starters
        question_starts = {
            "what", "how", "who", "where", "when", "why", "which", "whom", "define", "explain",
            "can", "could", "would", "should", "do", "does", "did", "is", "are", "was", "were", "will", "may", "might"
        }

        # Word-level analysis
        words = text.split()
        if words and words[0] in question_starts:
            return True

        # Basic auxiliary verb pattern (e.g., "can you", "is it", "should we")
        if re.match(r"^(can|could|would|should|do|does|did|is|are|was|were|will|may|might)\b", text):
            return True

        return False


if __name__ == "__main__":
    sys.exit()
