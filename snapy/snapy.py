"""
This code is part of 'Snapy Chatbot' project
Copyright (c) 2025, Aymen Brahim Djelloul, All rights reserved.
Use of this source code is governed by MIT License.

@author : Aymen Brahim Djelloul
@version : 0.2
@date : 26.06.2025
@license : MIT

Snapy is a lightweight chatbot with context tracking and search capabilities,
built for educational use and easy customization.
"""

import os
import sys
import json
import random
import difflib
import textwrap
import datetime
import requests
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from functools import lru_cache, wraps
from pathlib import Path
from enum import Enum, auto
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ._nlp_engine import NlpEngine, ProcessedText
    from ._search_engine import SearchEngine

except ImportError:

    import traceback
    traceback.print_exc()


class ResponseType(Enum):
    """Enum for different response types"""
    EXACT_MATCH = auto()
    HIGH_CONFIDENCE = auto()
    MEDIUM_CONFIDENCE = auto()
    LOW_CONFIDENCE = auto()
    SEARCH_RESULT = auto()
    CONTEXT_BASED = auto()
    FALLBACK = auto()


class ChatMode(Enum):
    """Enum for chat modes"""
    STANDARD = auto()
    CONTEXT_AWARE = auto()
    LEARNING = auto()


@dataclass
class ResponseMetadata:
    """Metadata for bot responses"""
    response_type: ResponseType
    confidence: float
    processing_time: float
    matched_intent: Optional[str] = None
    search_query: Optional[str] = None
    context_used: bool = False
    fallback_reason: Optional[str] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class Config:
    """Enhanced configuration with validation and type safety"""

    # Core settings
    DEFAULT_LANG: str = "en"
    VERSION: str = "0.2"
    RELEASE_DATE: str = "26.06.2025"
    RESPONSE_TIMEOUT: int = 30

    # Paths (using pathlib for better path handling)
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    DATASET_URL: str = "https://raw.githubusercontent.com/aymenbrahimdjelloul/Snapy/refs/heads/main/datasets/dataset.json"
    DATASET_PATH: Path = field(default_factory=lambda: Path("../datasets/dataset.json"))
    CHAT_HISTORY_PATH: Path = field(default_factory=lambda: Path("../chat_history.json"))
    CACHE_DIR: Path = field(default_factory=lambda: Path("../cache"))
    LEARNING_DATA_PATH: Path = field(default_factory=lambda: Path("../learning_data.json"))

    # Network settings
    HEADERS: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Snapy/0.4 (Educational Chatbot)',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache'
    })

    # Confidence thresholds (more granular)
    EXACT_MATCH_CONFIDENCE: float = 1.0
    HIGH_CONFIDENCE: float = 0.85
    MEDIUM_CONFIDENCE: float = 0.65
    LOW_CONFIDENCE: float = 0.45
    SEARCH_THRESHOLD: float = 0.35
    FALLBACK_CONFIDENCE: float = 0.1

    # Context settings
    MAX_CONTEXT_LENGTH: int = 5
    CONTEXT_WEIGHT: float = 0.3
    CONTEXT_DECAY: float = 0.8

    # UI Configuration
    USE_TYPING_EFFECT: bool = True
    TYPING_SPEED: float = 0.001
    THINKING_ANIMATION_SPEED: float = 0.1
    MAX_LINE_LENGTH: int = 80

    # Performance settings
    MAX_CACHE_SIZE: int = 256
    MAX_HISTORY_SIZE: int = 1000
    SIMILARITY_CACHE_SIZE: int = 512
    ASYNC_TIMEOUT: int = 10

    # NLP settings
    USE_ADVANCED_NLP: bool = True
    ENABLE_KEYWORD_EXTRACTION: bool = True
    ENABLE_SENTIMENT_ANALYSIS: bool = True
    ENABLE_INTENT_CLASSIFICATION: bool = True

    # Learning settings
    ENABLE_LEARNING: bool = True
    LEARNING_THRESHOLD: float = 0.7
    MIN_LEARNING_CONFIDENCE: float = 0.5

    # Developer mode
    DEFAULT_DEV_MODE: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._ensure_directories()
        self._validate_thresholds()

    def _ensure_directories(self):
        """Create necessary directories"""
        for path in [self.DATASET_PATH.parent, self.CHAT_HISTORY_PATH.parent,
                    self.CACHE_DIR, self.LEARNING_DATA_PATH.parent]:
            path.mkdir(parents=True, exist_ok=True)

    def _validate_thresholds(self):
        """Ensure confidence thresholds are in correct order"""
        thresholds = [self.FALLBACK_CONFIDENCE, self.SEARCH_THRESHOLD, self.LOW_CONFIDENCE,
                     self.MEDIUM_CONFIDENCE, self.HIGH_CONFIDENCE, self.EXACT_MATCH_CONFIDENCE]
        if thresholds != sorted(thresholds):
            raise ValueError("Confidence thresholds must be in ascending order")


class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self):
        self.metrics = {
            'total_responses': 0,
            'avg_response_time': 0.0,
            'confidence_distribution': {},
            'response_types': {},
            'search_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._response_times = []

    def log_response(self, metadata: ResponseMetadata):
        """Log response metrics"""
        self.metrics['total_responses'] += 1
        self._response_times.append(metadata.processing_time)

        # Update averages
        if len(self._response_times) > 100:
            self._response_times.pop(0)  # Keep only recent times
        self.metrics['avg_response_time'] = sum(self._response_times) / len(self._response_times)

        # Track confidence distribution
        conf_bucket = f"{int(metadata.confidence * 10) * 10}%"
        self.metrics['confidence_distribution'][conf_bucket] = (
            self.metrics['confidence_distribution'].get(conf_bucket, 0) + 1
        )

        # Track response types
        response_type = metadata.response_type.name
        self.metrics['response_types'][response_type] = (
            self.metrics['response_types'].get(response_type, 0) + 1
        )

        if metadata.search_query:
            self.metrics['search_queries'] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return self.metrics.copy()


class NetworkManager:
    """Enhanced network operations with caching and retry logic"""

    def __init__(self, config: Config):
        self.config = config
        self._connection_cache = {}
        self._cache_timeout = 300  # 5 minutes
        self._session = requests.Session()
        self._session.headers.update(config.HEADERS)

    def is_connected(self, url: str = "https://8.8.8.8", timeout: int = 5) -> bool:
        """Check internet connectivity with caching"""
        current_time = time.time()

        # Check cache first
        if url in self._connection_cache:
            cached_time, cached_result = self._connection_cache[url]
            if current_time - cached_time < self._cache_timeout:
                return cached_result

        try:
            response = self._session.head(url, timeout=timeout)
            result = response.status_code < 400
        except (requests.ConnectionError, requests.Timeout, requests.RequestException):
            result = False

        # Cache the result
        self._connection_cache[url] = (current_time, result)
        return result

    def download_with_retry(self, url: str, max_retries: int = 3, timeout: int = 10) -> Optional[bytes]:
        """Download with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                response = self._session.get(url, timeout=timeout)
                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                if attempt < max_retries - 1:
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All download attempts failed for {url}: {e}")
        return None

    def __del__(self):
        """Cleanup session"""
        if hasattr(self, '_session'):
            self._session.close()


class DatasetManager:
    """Enhanced dataset management with caching and validation"""

    def __init__(self, config: Config, network_manager: NetworkManager, dev_mode: bool = False):

        self.config = config
        self.dev_mode = dev_mode
        self.network_manager = network_manager
        self._dataset_cache = None
        self._last_load_time = None
        self._dataset = self._load_dataset()
        self._intent_index = self._build_intent_index()

    def _build_intent_index(self) -> Dict[str, Dict]:
        """Build searchable index of intents"""
        index = {}
        for intent in self._dataset.get("intents", []):
            tag = intent.get("tag", "")
            if tag:
                index[tag] = intent
        return index

    def get_intent_by_tag(self, tag: str) -> Optional[Dict]:
        """Get intent by tag efficiently"""
        return self._intent_index.get(tag)

    def get_intents_by_category(self, category: str) -> List[Dict]:
        """Get intents by category"""
        return [
            intent for intent in self._dataset.get("intents", [])
            if intent.get("category", "").lower() == category.lower()
        ]

    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset with comprehensive fallback strategy"""
        dataset = None

        # Try local file first
        if self.config.DATASET_PATH.exists():
            try:
                with open(self.config.DATASET_PATH, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    if self._validate_dataset(dataset):

                        if self.dev_mode:
                            logger.info("Dataset loaded from local file")

                        return dataset

                    else:
                        logger.warning("Local dataset validation failed")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load local dataset: {e}")

        # Try online download
        if not dataset and self.network_manager.is_connected():
            content = self.network_manager.download_with_retry(self.config.DATASET_URL)
            if content:
                try:
                    dataset = json.loads(content.decode('utf-8'))
                    if self._validate_dataset(dataset):
                        # Save successful download
                        self._save_dataset(content)

                        if self.dev_mode:
                            logger.info("Dataset downloaded and cached")

                        return dataset
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse downloaded dataset: {e}")

        # Emergency fallback dataset
        if not dataset:
            logger.warning("Using emergency fallback dataset")
            return self._create_fallback_dataset()

        return dataset

    def _validate_dataset(self, dataset: Dict[str, Any]) -> bool:
        """Validate dataset structure"""
        required_keys = ['intents', 'version']
        if not all(key in dataset for key in required_keys):
            return False

        if not isinstance(dataset['intents'], list) or not dataset['intents']:
            return False

        # Validate intent structure
        for intent in dataset['intents']:
            if not isinstance(intent, dict):
                return False
            if 'patterns' not in intent or 'responses' not in intent:
                return False
            if not isinstance(intent['patterns'], list) or not isinstance(intent['responses'], list):
                return False

        return True

    def _save_dataset(self, content: bytes) -> None:
        """Save dataset to local cache"""
        try:
            with open(self.config.DATASET_PATH, 'wb') as f:
                f.write(content)

        except IOError as e:
            logger.error(f"Failed to save dataset: {e}")

    def _create_fallback_dataset(self) -> Dict[str, Any]:
        """Create minimal working dataset"""
        return {
            "version": "fallback-1.0",
            "updated": datetime.datetime.now().isoformat(),
            "intents": [
                {
                    "tag": "fallback",
                    "category": "system",
                    "patterns": ["*"],
                    "responses": [
                        "I'm sorry, I'm having trouble accessing my knowledge base right now.",
                        "I apologize, but I can't provide a good response at the moment.",
                        "Please try again later, I'm experiencing some difficulties."
                    ]
                },
                {
                    "tag": "greeting",
                    "category": "social",
                    "patterns": ["hello", "hi", "hey", "good morning", "good evening", "greetings"],
                    "responses": ["Hello!", "Hi there!", "Greetings!", "Nice to meet you!", "How can I help you today?"]
                },
                {
                    "tag": "goodbye",
                    "category": "social",
                    "patterns": ["bye", "goodbye", "see you", "farewell", "take care"],
                    "responses": ["Goodbye!", "See you later!", "Take care!", "Have a great day!"]
                }
            ]
        }

    def refresh(self) -> bool:
        """Force refresh dataset from source"""
        if not self.network_manager.is_connected():
            return False

        content = self.network_manager.download_with_retry(self.config.DATASET_URL)
        if content:
            try:
                new_dataset = json.loads(content.decode('utf-8'))
                if self._validate_dataset(new_dataset):
                    self._dataset = new_dataset
                    self._intent_index = self._build_intent_index()
                    self._save_dataset(content)
                    self._dataset_cache = None  # Clear cache
                    logger.info("Dataset refreshed successfully")
                    return True
            except json.JSONDecodeError:
                pass

        logger.error("Dataset refresh failed")
        return False

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._dataset

    @property
    def version(self) -> str:
        return self._dataset.get("version", "unknown")

    @property
    def updated_date(self) -> str:
        return self._dataset.get("updated", "unknown")

    @property
    def intent_count(self) -> int:
        return len(self._dataset.get("intents", []))


class ChatHistory:
    """Enhanced chat history with size limits and context analysis"""

    def __init__(self, config: Config):
        self.config = config
        self.history_path = config.CHAT_HISTORY_PATH
        self._memory_cache = []
        self._cache_dirty = False
        self._context_cache = {}

    def add_exchange(self, user_text: str, bot_text: str, metadata: Optional[ResponseMetadata] = None) -> None:
        """Add chat exchange with metadata"""
        timestamp = datetime.datetime.now().isoformat()

        user_msg = {
            "role": "user",
            "message": user_text,
            "timestamp": timestamp,
            "metadata": {"processed": True}
        }

        bot_msg = {
            "role": "bot",
            "message": bot_text,
            "timestamp": timestamp,
            "metadata": {
                "confidence": metadata.confidence if metadata else 0.0,
                "response_type": metadata.response_type.name if metadata else "unknown",
                "processing_time": metadata.processing_time if metadata else 0.0
            }
        }

        self._memory_cache.extend([user_msg, bot_msg])
        self._cache_dirty = True

        # Maintain size limit
        if len(self._memory_cache) > self.config.MAX_HISTORY_SIZE:
            excess = len(self._memory_cache) - self.config.MAX_HISTORY_SIZE
            self._memory_cache = self._memory_cache[excess:]

        # Clear context cache when history changes
        self._context_cache.clear()

        # Periodic save to disk
        if len(self._memory_cache) % 20 == 0:
            self._save_to_disk()

    def get_recent(self, count: int = 5) -> List[Dict]:
        """Get recent messages efficiently"""
        if not self._memory_cache:
            self._load_from_disk()

        return self._memory_cache[-count * 2:] if self._memory_cache else []

    def get_context(self, count: int = 3) -> str:
        """Get recent context as formatted string with caching"""
        cache_key = f"context_{count}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        recent = self.get_recent(count)
        context_parts = []

        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Bot"
            context_parts.append(f"{role}: {msg['message']}")

        context = "\n".join(context_parts)
        self._context_cache[cache_key] = context
        return context

    def get_user_patterns(self, limit: int = 50) -> List[str]:
        """Extract user message patterns for learning"""
        user_messages = [
            msg["message"] for msg in self._memory_cache[-limit:]
            if msg.get("role") == "user"
        ]
        return user_messages

    def get_conversation_topics(self) -> List[str]:
        """Extract conversation topics from history"""
        # This is a simplified implementation
        # In practice, you'd use NLP to extract topics
        topics = []
        for msg in self._memory_cache:
            if msg.get("role") == "user":
                # Simple keyword extraction
                words = msg["message"].lower().split()
                topics.extend([word for word in words if len(word) > 4])

        # Return most common topics
        from collections import Counter
        return [topic for topic, count in Counter(topics).most_common(10)]

    def clear(self) -> None:
        """Clear all history"""
        self._memory_cache.clear()
        self._context_cache.clear()
        self._cache_dirty = True
        if self.history_path.exists():
            self.history_path.unlink()

    def _load_from_disk(self) -> None:
        """Load history from disk"""
        if not self.history_path.exists():
            return

        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._memory_cache = data.get("messages", [])
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load chat history: {e}")
            self._memory_cache = []

    def _save_to_disk(self) -> None:
        """Save history to disk"""
        if not self._cache_dirty:
            return

        try:
            data = {
                "messages": self._memory_cache,
                "last_updated": datetime.datetime.now().isoformat(),
                "stats": {
                    "total_messages": len(self._memory_cache),
                    "conversation_topics": self.get_conversation_topics()
                }
            }
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._cache_dirty = False
        except IOError as e:
            logger.error(f"Failed to save chat history: {e}")

    def __len__(self) -> int:
        return len(self._memory_cache)

    def __del__(self):
        """Ensure data is saved on destruction"""
        if hasattr(self, '_cache_dirty') and self._cache_dirty:
            self._save_to_disk()


class ContextAnalyzer:
    """Analyze conversation context for better responses"""

    def __init__(self, config: Config, nlp_engine: Optional[NlpEngine] = None):
        self.config = config
        self.nlp_engine = nlp_engine

    def analyze_context(self, current_input: str, chat_history: ChatHistory) -> Dict[str, Any]:
        """Analyze current input in context of conversation history"""
        context = {
            "has_context": False,
            "context_keywords": [],
            "context_sentiment": "neutral",
            "follow_up_question": False,
            "topic_continuation": False,
            "context_weight": 0.0
        }

        if not self.nlp_engine:
            return context

        # Get recent context
        recent_messages = chat_history.get_recent(self.config.MAX_CONTEXT_LENGTH)
        if not recent_messages:
            return context

        context["has_context"] = True

        # Extract keywords from recent context
        context_text = " ".join([msg["message"] for msg in recent_messages])
        context_keywords = self.nlp_engine.extract_keywords(context_text)

        # Check for keyword overlap with current input
        current_keywords = self.nlp_engine.extract_keywords(current_input)
        overlap = set(context_keywords).intersection(set(current_keywords))

        if overlap:
            context["topic_continuation"] = True
            context["context_keywords"] = list(overlap)
            context["context_weight"] = min(len(overlap) / max(len(current_keywords), 1), 1.0)

        # Check if current input is a follow-up question
        if self.nlp_engine.is_question(current_input):
            # Simple heuristic: if it's a short question, it might be a follow-up
            if len(current_input.split()) <= 5:
                context["follow_up_question"] = True
                context["context_weight"] = max(context["context_weight"], 0.5)

        return context


class LearningManager:
    """Handles all learning from user conversations"""

    def __init__(self, config: Config, dev_mode: bool = False, nlp_engine: Optional[NlpEngine] = None):
        self.dev_mode = dev_mode

        self.config = config
        self.nlp_engine = nlp_engine
        self.learning_data_path = config.LEARNING_DATA_PATH
        self._learning_data = self._load_learning_data()
        self._log_initialization()

    def _log_initialization(self) -> None:
        """Log initialization details"""
        if self.dev_mode:
            logger.info("Initializing LearningManager with _config:")
            logger.info(f"  - ENABLE_LEARNING: {self.config.ENABLE_LEARNING}")
            logger.info(f"  - MIN_LEARNING_CONFIDENCE: {self.config.MIN_LEARNING_CONFIDENCE}")
            logger.info(f"  - Learning data path: {self.learning_data_path}")
            logger.info(f"  - Loaded {len(self._learning_data['learned_patterns'])} learned patterns")

    def _load_learning_data(self) -> Dict[str, Any]:
        """Load existing learning data"""
        if not self.learning_data_path.exists():
            if self.dev_mode:
                logger.debug("No existing learning data found, creating new structure")
            return {
                "version": 1,
                "learned_patterns": {},
                "response_mappings": {},
                "statistics": {
                    "total_learned": 0,
                    "last_learned": None
                }
            }

        try:
            with open(self.learning_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if self.config.dev_mode:
                    logger.debug(
                        f"Successfully loaded learning data with {len(data.get('learned_patterns', {}))} patterns")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load learning data: {e}")
            if self.config.dev_mode:
                logger.debug("Creating new learning data structure due to load error")
            return {
                "version": 1,
                "learned_patterns": {},
                "response_mappings": {},
                "statistics": {
                    "total_learned": 0,
                    "last_learned": None
                }
            }

    def _save_learning_data(self) -> None:
        """Save learning data to disk"""
        try:
            with open(self.learning_data_path, 'w', encoding='utf-8') as f:
                json.dump(self._learning_data, f, indent=2, ensure_ascii=False)
            if self.config.dev_mode:
                logger.debug(f"Saved learning data with {len(self._learning_data['learned_patterns'])} patterns")
        except IOError as e:
            logger.error(f"Failed to save learning data: {e}")
            if self.config.dev_mode:
                logger.debug("Learning data state at time of failure:", self._learning_data)

    def analyze_conversation(self, user_input: str, bot_response: str,
                             metadata: ResponseMetadata) -> Optional[Dict[str, Any]]:
        """Analyze conversation for learning opportunities"""
        if not self.config.ENABLE_LEARNING:
            if self.config.dev_mode:
                logger.debug("Learning disabled in _config, skipping analysis")
            return None

        # Only learn from high-confidence responses
        if metadata.confidence < self.config.MIN_LEARNING_CONFIDENCE:
            if self.config.dev_mode:
                logger.debug(
                    f"Confidence {metadata.confidence} below threshold {self.config.MIN_LEARNING_CONFIDENCE}, skipping")
            return None

        # Extract key information
        analysis = {
            "user_input": user_input,
            "bot_response": bot_response,
            "confidence": metadata.confidence,
            "response_type": metadata.response_type.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "learned": False
        }

        if self.config.dev_mode:
            logger.debug(f"Conversation analysis created:\n{json.dumps(analysis, indent=2)}")

        return analysis

    def learn_from_conversation(self, analysis: Dict[str, Any]) -> bool:
        """Learn from a conversation analysis"""
        if not analysis or analysis.get("learned", False):
            if self.config.dev_mode:
                logger.debug("Skipping learning - no analysis or already learned")
            return False

        user_input = analysis["user_input"]
        bot_response = analysis["bot_response"]

        # Normalize input
        normalized_input = self._normalize_input(user_input)
        if self.config.dev_mode:
            logger.debug(f"Normalized input: '{user_input}' -> '{normalized_input}'")

        # Check if this is worth learning
        if not self._is_worth_learning(normalized_input, bot_response):
            if self.config.dev_mode:
                logger.debug("Conversation not worth learning")
            return False

        # Add to learned patterns
        intent_tag = f"learned_{len(self._learning_data['learned_patterns']) + 1}"

        if intent_tag not in self._learning_data["learned_patterns"]:
            if self.config.dev_mode:
                logger.info(f"Creating new learned intent: {intent_tag}")
            self._learning_data["learned_patterns"][intent_tag] = {
                "patterns": [normalized_input],
                "responses": [bot_response],
                "category": "learned",
                "created": datetime.datetime.now().isoformat(),
                "last_used": datetime.datetime.now().isoformat(),
                "usage_count": 1
            }
        else:
            # Update existing learned pattern
            if self.config.dev_mode:
                logger.debug(f"Updating existing learned intent: {intent_tag}")
            learned_intent = self._learning_data["learned_patterns"][intent_tag]
            if normalized_input not in learned_intent["patterns"]:
                learned_intent["patterns"].append(normalized_input)
                if self.config.dev_mode:
                    logger.debug(f"Added new pattern to intent {intent_tag}")
            if bot_response not in learned_intent["responses"]:
                learned_intent["responses"].append(bot_response)
                if self.config.dev_mode:
                    logger.debug(f"Added new response to intent {intent_tag}")
            learned_intent["usage_count"] += 1
            learned_intent["last_used"] = datetime.datetime.now().isoformat()

        # Update statistics
        self._learning_data["statistics"]["total_learned"] += 1
        self._learning_data["statistics"]["last_learned"] = datetime.datetime.now().isoformat()

        # Mark as learned
        analysis["learned"] = True
        analysis["intent_tag"] = intent_tag

        # Save to disk
        self._save_learning_data()

        if self.config.dev_mode:
            logger.info(f"Successfully learned new pattern (tag: {intent_tag})")
            logger.debug(f"Updated learning data state:\n{json.dumps(self._learning_data, indent=2)}")

        return True

    def _normalize_input(self, text: str) -> str:
        """Normalize text for learning"""
        if self.nlp_engine:
            normalized = self.nlp_engine.text_normalization(text)
            if self.config.dev_mode:
                logger.debug(f"NLP normalization: '{text}' -> '{normalized}'")
            return normalized
        normalized = text.lower().strip()
        if self.config.dev_mode:
            logger.debug(f"Basic normalization: '{text}' -> '{normalized}'")
        return normalized

    def _is_worth_learning(self, normalized_input: str, response: str) -> bool:
        """Determine if a conversation is worth learning"""
        # Don't learn very short inputs
        if len(normalized_input.split()) < 3:
            if self.config.dev_mode:
                logger.debug(f"Input too short: '{normalized_input}'")
            return False

        # Don't learn very generic responses
        generic_responses = [
            "I don't know", "I'm not sure", "I can't help",
            "sorry", "I apologize", "please try again"
        ]
        if any(phrase.lower() in response.lower() for phrase in generic_responses):
            if self.config.dev_mode:
                logger.debug(f"Generic response detected: '{response}'")
            return False

        # Check similarity to existing patterns
        for intent_tag, intent in self._learning_data["learned_patterns"].items():
            for pattern in intent["patterns"]:
                similarity = self._calculate_similarity(normalized_input, pattern)
                if similarity > 0.8:  # Too similar to existing pattern
                    if self.config.dev_mode:
                        logger.debug(
                            f"Input too similar ({similarity:.2f}) to existing pattern in intent {intent_tag}\n"
                            f"Input: '{normalized_input}'\n"
                            f"Existing: '{pattern}'"
                        )
                    return False

        if self.config.dev_mode:
            logger.debug("Input passed all worth-learning checks")
        return True

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between texts"""
        if self.nlp_engine:
            similarity = self.nlp_engine.similarity(text1, text2)
            if self.config.dev_mode:
                logger.debug(f"NLP similarity between '{text1}' and '{text2}': {similarity:.2f}")
            return similarity

        # Fallback to basic similarity
        similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        if self.config.dev_mode:
            logger.debug(f"Basic similarity between '{text1}' and '{text2}': {similarity:.2f}")
        return similarity

    def get_learned_intents(self) -> List[Dict[str, Any]]:
        """Get all learned intents"""
        intents = list(self._learning_data["learned_patterns"].values())
        if self.config.dev_mode:
            logger.debug(f"Returning {len(intents)} learned intents")
        return intents

    def merge_with_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Merge learned intents with main dataset"""
        if not self._learning_data["learned_patterns"]:
            if self.dev_mode:
                logger.debug("No learned patterns to merge")
            return dataset

        if self.config.dev_mode:
            logger.info(f"Merging {len(self._learning_data['learned_patterns'])} learned patterns with dataset")

        # Create a copy of the dataset to modify
        merged = dataset.copy()

        # Add learned intents
        for tag, intent in self._learning_data["learned_patterns"].items():
            # Check if this intent already exists in dataset
            existing = next((i for i in merged["intents"] if i["tag"] == tag), None)

            if existing:
                # Update existing intent
                new_patterns = list(set(existing["patterns"] + intent["patterns"]))
                new_responses = list(set(existing["responses"] + intent["responses"]))

                if self.config.dev_mode:
                    logger.debug(
                        f"Updating existing intent {tag}\n"
                        f"Added {len(new_patterns) - len(existing['patterns'])} new patterns\n"
                        f"Added {len(new_responses) - len(existing['responses'])} new responses"
                    )

                existing["patterns"] = new_patterns
                existing["responses"] = new_responses
            else:
                # Add new intent
                if self.config.dev_mode:
                    logger.debug(f"Adding new intent {tag} with {len(intent['patterns'])} patterns")
                new_intent = {
                    "tag": tag,
                    "patterns": intent["patterns"],
                    "responses": intent["responses"],
                    "category": "learned"
                }
                merged["intents"].append(new_intent)

        if self.config.dev_mode:
            logger.info(f"Merged dataset now contains {len(merged['intents'])} intents")

        return merged

class ResponseGenerator:
    """Enhanced response generation with context awareness"""

    def __init__(self, config: Config, nlp_engine: Optional[NlpEngine] = None):
        self.config = config
        self.nlp_engine = nlp_engine
        self.context_analyzer = ContextAnalyzer(config, nlp_engine)
        self._similarity_cache = {}

    @lru_cache(maxsize=512)
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Enhanced similarity calculation with NLP integration"""
        if text1 == text2:
            return 1.0

        # Use NLP engine if available
        if self.nlp_engine:
            return self.nlp_engine.similarity(text1, text2)

        # Fallback to basic similarity
        seq_score = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        # Word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return seq_score

        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_score = len(intersection) / len(union) if union else 0

        # Weighted combination
        final_score = (seq_score * 0.6) + (jaccard_score * 0.4)
        return round(min(final_score, 1.0), 4)

    def find_best_match(self, processed_input: Union[str, ProcessedText], intents: List[Dict],
                       context: Optional[Dict] = None) -> Tuple[Optional[Dict], float, ResponseType]:
        """Find best matching intent with context consideration"""

        # Handle both string and ProcessedText inputs
        if isinstance(processed_input, str):
            normalized_input = processed_input.lower().strip()
            is_question = "?" in processed_input
        else:
            normalized_input = processed_input.normalized
            is_question = processed_input.is_question

        best_intent = None
        best_score = 0.0
        response_type = ResponseType.FALLBACK

        # Apply context boost if available
        context_boost = 0.0
        if context and context.get("topic_continuation"):
            context_boost = context.get("context_weight", 0.0) * self.config.CONTEXT_WEIGHT

        for intent in intents:
            if "fallback" in intent.get("tag", "").lower():
                continue

            current_best_score = 0.0

            for pattern in intent.get("patterns", []):
                pattern_lower = pattern.lower()

                # Exact match gets highest score
                if normalized_input == pattern_lower:
                    return intent, self.config.EXACT_MATCH_CONFIDENCE, ResponseType.EXACT_MATCH

                # Similarity matching
                score = self.calculate_similarity(normalized_input, pattern_lower)

                # Apply context boost
                if context_boost > 0:
                    # Check if intent is related to context keywords
                    intent_text = " ".join(intent.get("patterns", []) + intent.get("responses", []))
                    if any(keyword in intent_text.lower() for keyword in context.get("context_keywords", [])):
                        score += context_boost

                current_best_score = max(current_best_score, score)

            if current_best_score > best_score:
                best_score = current_best_score
                best_intent = intent

        # Determine response type based on score
        if best_score >= self.config.HIGH_CONFIDENCE:
            response_type = ResponseType.HIGH_CONFIDENCE
        elif best_score >= self.config.MEDIUM_CONFIDENCE:
            response_type = ResponseType.MEDIUM_CONFIDENCE
        elif best_score >= self.config.LOW_CONFIDENCE:
            response_type = ResponseType.LOW_CONFIDENCE
        else:
            response_type = ResponseType.FALLBACK

        # Boost context-based responses
        if context and context.get("has_context") and best_score > self.config.LOW_CONFIDENCE:
            response_type = ResponseType.CONTEXT_BASED

        return best_intent, best_score, response_type

    def generate_contextual_response(self, intent: Dict, context: Dict, user_input: str) -> str:
        """Generate response with contextual modifications"""
        responses = intent.get("responses", [])
        if not responses:
            return "I'm not sure how to respond to that."

        base_response = random.choice(responses)

        # Apply contextual modifications
        if context.get("follow_up_question"):
            # Add continuity phrases for follow-up questions
            continuity_phrases = [
                "Also, ", "Additionally, ", "Furthermore, ", "Moreover, "
            ]
            if not any(base_response.startswith(phrase.strip()) for phrase in continuity_phrases):
                base_response = random.choice(continuity_phrases) + base_response.lower()

        elif context.get("topic_continuation"):
            # Add topic continuation phrases
            continuation_phrases = [
                "Regarding that, ", "About that topic, ", "On that subject, "
            ]
            if len(base_response.split()) > 10:  # Only for longer responses
                base_response = random.choice(continuation_phrases) + base_response.lower()

        return base_response


class Snapy:
    """Enhanced main chatbot class with improved NLP integration"""

    def __init__(self, developer_mode: bool = None, chat_mode: ChatMode = ChatMode.STANDARD):
        self._config = Config()

        # Set developer mode
        if developer_mode is not None:
            self.developer_mode = developer_mode
        else:
            self.developer_mode = self._config.DEFAULT_DEV_MODE

        self.chat_mode = chat_mode

        # Initialize components
        self._network_manager = NetworkManager(self._config)
        self._dataset_manager = DatasetManager(self._config, self._network_manager, developer_mode)
        self._chat_history = ChatHistory(self._config)
        self._performance_monitor = PerformanceMonitor()

        # Initialize NLP engine
        self._nlp_engine = None
        if NlpEngine and self._config.USE_ADVANCED_NLP:

            try:
                self._nlp_engine = NlpEngine(
                    dev_mode=self.developer_mode,
                    max_context_length=self._config.MAX_CONTEXT_LENGTH
                )

                if developer_mode:
                    logger.info("Advanced NLP engine initialized")

            except Exception as e:
                logger.error(f"Failed to initialize NLP engine: {e}")
                self._nlp_engine = None

        # Initialize response generator
        self._response_generator = ResponseGenerator(self._config, self._nlp_engine)

        # Initialize search engine (always available if possible)
        self._search_engine = None
        if SearchEngine:
            try:
                self._search_engine = SearchEngine()

                if developer_mode:
                    logger.info("Search engine initialized")

            except Exception as e:
                logger.error(f"Failed to initialize search engine: {e}")
                self._search_engine = None

        # State tracking
        self._last_user_input = None
        self._last_bot_response = None
        self._last_metadata = None
        self._last_response_time = 0.0
        self._last_confidence = 0.0
        self._conversation_state = {}

        # Initialize learning manager
        self._learning_manager = LearningManager(self._config, developer_mode, self._nlp_engine)

        # Merge learned intents with dataset
        self._dataset_manager._dataset = self._learning_manager.merge_with_dataset(
            self._dataset_manager.dataset
        )

    def generate_response(self, user_input: str) -> Tuple[str, float]:
        """Generate response with comprehensive fallback strategy"""
        start_time = datetime.datetime.now()

        # Input validation and normalization
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you please say something?", 0.0

        self._last_user_input = user_input
        normalized_input = self._normalize_input(user_input)

        try:
            # First try to find a response from the dataset
            response, confidence = self._generate_standard_response(normalized_input)

            # If confidence is low and we're online, try searching
            if (confidence < self._config.MEDIUM_CONFIDENCE and
                self._network_manager.is_connected() and
                self._search_engine):

                if self.developer_mode:
                    logger.info(f"Low confidence ({confidence:.2f}), attempting online search")

                search_response, search_confidence = self._try_online_search(normalized_input)
                if search_response:
                    response = search_response
                    confidence = search_confidence

            # Finalize response
            response = self._format_response(response)
            self._finalize_response(user_input, response, confidence)

            # Calculate response time
            self._last_response_time = (datetime.datetime.now() - start_time).total_seconds()

            return response, confidence

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if self.developer_mode:
                logger.error(traceback.format_exc())

            fallback_response = "I apologize, but I encountered an error processing your request."
            self._finalize_response(user_input, fallback_response, 0.0)
            return fallback_response, 0.0

    def _try_online_search(self, normalized_input: str) -> Tuple[Optional[str], float]:
        """Attempt to get an answer from online search"""
        try:
            response = self._search_engine.get_answer(normalized_input)
            if response:
                return response, 0.7  # Medium confidence for search results
        except Exception as e:
            logger.error(f"Search engine error: {e}")
        return None, 0.0

    def _normalize_input(self, user_input: str) -> str:
        """Normalize user input"""
        if self._nlp_engine:
            return self._nlp_engine.text_normalization(user_input)
        return user_input.lower().strip()

    def _generate_standard_response(self, normalized_input: str) -> Tuple[str, float]:
        """Generate standard chatbot response"""
        intents = self._dataset_manager.dataset.get("intents", [])

        # Analyze context if available
        context = None
        if self.chat_mode == ChatMode.CONTEXT_AWARE and self._nlp_engine:
            context = self._response_generator.context_analyzer.analyze_context(
                normalized_input, self._chat_history
            )

        # Find best matching intent
        best_intent, confidence, response_type = self._response_generator.find_best_match(
            normalized_input, intents, context
        )

        # Generate response
        if confidence >= self._config.MEDIUM_CONFIDENCE and best_intent:
            if context and context.get("has_context"):
                response = self._response_generator.generate_contextual_response(best_intent, context, normalized_input)
            else:
                responses = best_intent.get("responses", [])
                response = random.choice(responses) if responses else ""

            return response, confidence

        # Fallback to dataset fallback intent
        fallback_intent = next(
            (intent for intent in intents if "fallback" in intent.get("tag", "").lower()),
            intents[0] if intents else None
        )

        if fallback_intent:
            responses = fallback_intent.get("responses", [])
            if responses:
                return random.choice(responses), self._config.FALLBACK_CONFIDENCE

        return "I'm not sure how to respond to that.", self._config.FALLBACK_CONFIDENCE

    def _format_response(self, response: str) -> str:
        """Format response text"""
        if not response:
            return "I'm sorry, I don't have a response for that."

        # Wrap long lines
        return '\n'.join(textwrap.wrap(response, width=self._config.MAX_LINE_LENGTH))

    def _finalize_response(self, user_input: str, response: str, confidence: float) -> None:
        """Finalize response and update state"""
        metadata = ResponseMetadata(
            response_type=ResponseType.SEARCH_RESULT if "search" in response.lower() else ResponseType.HIGH_CONFIDENCE,
            confidence=confidence,
            processing_time=self._last_response_time
        )

        # Analyze for learning
        if self._config.ENABLE_LEARNING and self.chat_mode == ChatMode.LEARNING:
            analysis = self._learning_manager.analyze_conversation(user_input, response, metadata)
            if analysis:
                self._learning_manager.learn_from_conversation(analysis)

        self._chat_history.add_exchange(user_input, response, metadata)
        self._last_bot_response = response
        self._last_metadata = metadata
        self._last_confidence = confidence
        self._performance_monitor.log_response(metadata)

    @property
    def is_connected(self) -> bool:
        """Check if connected to internet"""
        return self._network_manager.is_connected()

    @property
    def dataset_version(self) -> str:
        """Get dataset version"""
        return self._dataset_manager.version


if __name__ == "__main__":
    sys.exit(0)
