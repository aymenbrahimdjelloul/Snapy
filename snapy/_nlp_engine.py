"""
This code or file is part of 'Snapy Chatbot' project
copyright (c) 2025, Aymen Brahim Djelloul, All rights reserved.
use of this source code is governed by MIT License that can be found on the project folder.

@Author : Aymen Brahim Djelloul
Version : 0.2
Date : 26.06.2025
License : MIT

     \\ Snapy is a lightweight and simple chatbot with context tracking and online search capabilities,
        built for educational use and easy customization.

    \\ _nlp_engine is a module that contains Natural language processing functions and logic for Snapy bot

"""

# IMPORTS
import re
import json
import sys
import difflib
import requests
import unicodedata
import os.path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
import logging


@dataclass
class ProcessedText:
    """Container for processed text with metadata."""
    original: str
    normalized: str
    tokens: List[str]
    is_question: bool
    confidence: float = 0.0


class NlpEngine:
    """
    NlpEngine is a lightweight natural language processing engine designed
    for chatbot applications. It supports context tracking and an optional
    developer mode for debugging and extended logging.

    Attributes:
        dev_mode (bool): Enables verbose output for development purposes.
        context (List[str]): Stores the history of user inputs for context tracking.
        logger (logging.Logger): Logger instance for debugging and error tracking.
    """

    def __init__(self, dev_mode: bool = False, max_context_length: int = 10) -> None:
        """
        Initialize the NLP engine.

        Args:
            dev_mode: Enable developer mode for verbose logging
            max_context_length: Maximum number of context entries to retain
        """
        self.dev_mode = dev_mode
        self.context: List[str] = []
        self.max_context_length = max_context_length

        # Set up logging
        self.logger = self._setup_logger()

        # Precompiled regex patterns for better performance
        self._compile_patterns()

        if self.dev_mode:
            self.logger.info("NLP Engine initialized in developer mode")

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG if self.dev_mode else logging.WARNING)
        return logger

    def _compile_patterns(self) -> None:
        """Precompile regex patterns for better performance."""
        self.patterns = {
            'emoji_symbols': re.compile(r'[^\x00-\x7F]+'),
            'punctuation_space': re.compile(r'[^\w\s.?!,;:-]'),
            'punctuation_surround': re.compile(r'([.?!,;:-])'),
            'all_punctuation': re.compile(r'[^\w\s]'),
            'whitespace': re.compile(r'\s+'),
            'double_consonant': re.compile(r'(.)\1$'),
            'question_start': re.compile(r'^(can|could|would|should|do|does|did|is|are|was|were|will|may|might)\b'),
            'aggressive_phrases': re.compile(r'\b(what the hell|what the fuck|what is this shit|what a day|what a mess|how dare you)\b')
        }

    def add_to_context(self, text: str) -> None:
        """
        Add text to context history with automatic cleanup.

        Args:
            text: Text to add to context
        """
        if not text or not isinstance(text, str):
            return

        self.context.append(text.strip())

        # Maintain context length limit
        if len(self.context) > self.max_context_length:
            self.context.pop(0)

        if self.dev_mode:
            self.logger.debug(f"Added to context: '{text}' (context length: {len(self.context)})")

    def get_context(self, last_n: Optional[int] = None) -> List[str]:
        """
        Retrieve context history.

        Args:
            last_n: Number of last entries to return (None for all)

        Returns:
            List of context entries
        """
        if last_n is None:
            return self.context.copy()
        return self.context[-last_n:] if last_n > 0 else []

    def clear_context(self) -> None:
        """Clear all context history."""
        self.context.clear()
        if self.dev_mode:
            self.logger.debug("Context cleared")

    def process_text(self, input_text: str, **kwargs) -> ProcessedText:
        """
        Process text through normalization and analysis pipeline.

        Args:
            input_text: Raw text to process
            **kwargs: Additional arguments for text_normalization

        Returns:
            ProcessedText object with processed results
        """
        if not input_text or not isinstance(input_text, str):
            return ProcessedText(
                original="",
                normalized="",
                tokens=[],
                is_question=False,
                confidence=0.0
            )

        try:
            # Normalize text
            normalized = self.text_normalization(input_text, **kwargs)

            # Tokenize
            tokens = self.tokenize(normalized)

            # Analyze if it's a question
            is_question = self.is_question(normalized)

            # Add to context
            self.add_to_context(input_text)

            result = ProcessedText(
                original=input_text,
                normalized=normalized,
                tokens=tokens,
                is_question=is_question,
                confidence=1.0  # Could be enhanced with actual confidence scoring
            )

            if self.dev_mode:
                self.logger.debug(f"Processed text: {result}")

            return result

        except Exception as e:
            self.logger.error(f"Error processing text '{input_text}': {e}")
            return ProcessedText(
                original=input_text,
                normalized=input_text,
                tokens=[input_text],
                is_question=False,
                confidence=0.0
            )

    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization that preserves meaningful tokens.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Split on whitespace and filter empty strings
        tokens = [token for token in text.split() if token.strip()]

        if self.dev_mode:
            self.logger.debug(f"Tokenized '{text}' into {len(tokens)} tokens")

        return tokens

    def text_normalization(self, input_text: str, keep_punctuation: bool = True,
                          lemmatize: bool = True, aggressive: bool = False) -> str:
        """
        Aggressively normalize and simplify text input for matching purposes.

        Args:
            input_text: The raw text to normalize
            keep_punctuation: Whether to preserve simple punctuation marks
            lemmatize: Whether to apply basic rule-based lemmatization
            aggressive: Whether to apply more aggressive normalization

        Returns:
            Normalized text string

        Examples:
            >>> engine = NlpEngine()
            >>> engine.text_normalization("Café's are OPEN!! 😊  Visit us at café.com")
            'cafe are open ! ! visit us at cafe . com'

            >>> engine.text_normalization("I'm running and they're dancing", lemmatize=True)
            'i am run and they are danc'
        """
        try:
            if not input_text or not isinstance(input_text, str):
                return ""

            # 1. Lowercase
            text = input_text.lower()

            # 2. Handle common contractions
            text = self._expand_contractions(text)

            # 3. Remove accents
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))

            # 4. Remove emojis & symbols (anything not ASCII printable)
            text = self.patterns['emoji_symbols'].sub('', text)

            # 5. Handle punctuation
            if keep_punctuation:
                # Convert complex punctuation to spaces, keep simple ones
                text = self.patterns['punctuation_space'].sub(' ', text)
                # Add spaces around remaining punctuation for better tokenization
                text = self.patterns['punctuation_surround'].sub(r' \1 ', text)
            else:
                # Remove all punctuation
                text = self.patterns['all_punctuation'].sub(' ', text)

            # 6. Normalize whitespace
            text = self.patterns['whitespace'].sub(' ', text).strip()

            # 7. Enhanced rule-based lemmatization
            if lemmatize:
                text = self._lemmatize_text(text)

            # 8. Aggressive normalization (optional)
            if aggressive:
                text = self._aggressive_normalization(text)

            return text

        except Exception as e:
            self.logger.error(f"Normalization failed for '{input_text}': {e}")
            if self.dev_mode:
                self.logger.debug(traceback.format_exc())
            return input_text  # Return original text rather than failing completely

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
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
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)

        return text

    def _lemmatize_text(self, text: str) -> str:
        """Apply basic rule-based lemmatization."""
        words = text.split()
        lemmas = []

        for word in words:
            # Skip very short words or punctuation
            if len(word) <= 1 or not any(c.isalpha() for c in word):
                lemmas.append(word)
                continue

            lemmatized = self._lemmatize_word(word)
            lemmas.append(lemmatized)

        return " ".join(lemmas)

    def _lemmatize_word(self, word: str) -> str:
        """Lemmatize a single word using rule-based approach."""
        original_word = word

        # Handle plurals
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"  # categories → category
        elif word.endswith("es") and len(word) > 3 and not word.endswith("ses"):
            word = word[:-2]  # fixes → fix, but not processes → proces
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]  # cars → car, but not pass → pas

        # Handle verb forms
        if word.endswith("ing") and len(word) > 5:
            if len(word) > 6 and word[-4] == word[-5] and word[-4] not in 'aeiou':
                word = word[:-4]  # running → run (double consonant)
            else:
                word = word[:-3]  # dancing → danc
            # Try to restore 'e' if it was likely removed
            if word.endswith(('at', 'iv', 'iz', 'us', 'in', 'ov')):
                word = word + 'e'  # racing → rac → race

        elif word.endswith("ed") and len(word) > 4:
            if len(word) > 5 and word[-3] == word[-4] and word[-3] not in 'aeiou':
                word = word[:-3]  # stopped → stop
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
        elif word.endswith("tion") and len(word) > 6:
            word = word[:-4]  # information → informat
        elif word.endswith("able") and len(word) > 6:
            word = word[:-4]  # readable → read

        return word

    def _aggressive_normalization(self, text: str) -> str:
        """Apply aggressive normalization techniques."""
        # Remove repeated characters (e.g., "sooooo" → "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Handle common internet slang
        slang_replacements = {
            "u": "you", "ur": "your", "r": "are", "n": "and",
            "thru": "through", "tho": "though", "cuz": "because",
            "gonna": "going to", "wanna": "want to", "gotta": "got to"
        }

        words = text.split()
        normalized_words = []

        for word in words:
            if word in slang_replacements:
                normalized_words.append(slang_replacements[word])
            else:
                normalized_words.append(word)

        return " ".join(normalized_words)

    def is_question(self, normalized_text: str) -> bool:
        """
        Detect whether the input is a question using multiple heuristics.

        Args:
            normalized_text: Pre-normalized text to analyze

        Returns:
            True if the text appears to be a question
        """
        if not normalized_text:
            return False

        text = normalized_text.lower().strip()

        # Fast path: question mark check
        if text.endswith("?"):
            return True

        # Quick ignore: aggressive or non-question phrases
        if self.patterns['aggressive_phrases'].search(text):
            return False

        # Question-like starters
        question_starts = {
            "what", "how", "who", "where", "when", "why", "which", "whom",
            "define", "explain", "tell", "show", "describe"
        }

        # Word-level analysis
        words = text.split()
        if words and words[0] in question_starts:
            return True

        # Auxiliary verb pattern (e.g., "can you", "is it", "should we")
        if self.patterns['question_start'].match(text):
            return True

        # Check for question words anywhere in the sentence (weaker signal)
        question_words = {"what", "how", "who", "where", "when", "why", "which"}
        if any(word in question_words for word in words[:5]):  # Check first 5 words
            return True

        return False

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using difflib.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            # Normalize both texts
            norm1 = self.text_normalization(text1)
            norm2 = self.text_normalization(text2)

            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()

            return similarity

        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def extract_keywords(self, text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text by filtering out common stop words.

        Args:
            text: Text to extract keywords from
            min_length: Minimum length of keywords
            max_keywords: Maximum number of keywords to return

        Returns:
            List of keywords
        """
        # Common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those"
        }

        try:
            # Normalize text
            normalized = self.text_normalization(text, keep_punctuation=False)
            words = normalized.split()

            # Filter keywords
            keywords = []
            for word in words:
                if (len(word) >= min_length and
                    word.lower() not in stop_words and
                    word.isalpha()):
                    keywords.append(word)

            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for keyword in keywords:
                if keyword not in seen:
                    seen.add(keyword)
                    unique_keywords.append(keyword)

            return unique_keywords[:max_keywords]

        except Exception as e:
            self.logger.error(f"Error extracting keywords from '{text}': {e}")
            return []


if __name__ == "__main__":
    sys.exit(0)