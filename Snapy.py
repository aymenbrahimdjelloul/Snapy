"""
@author : Aymen Brahim Djelloul
version : 0.1
Date : 30.04.2025
License : MIT



"""

# IMPORTS
import sys
import re
import os
import json
import time
import difflib
import textwrap
import random
import requests
import datetime
import traceback
from typing import Dict, List, Tuple, Optional, Union
import colorama
from colorama import Fore, Back, Style
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from nlp_engine import NlpEngine
from search_engine import SearchEngine


# DEFINE GENERALE CONFIGS
@dataclass
class Config:

    """
    Config class
    """

    DEFAULT_LANG: str = "en"
    VERSION: str = "0.1"
    RELEASE_DATE: str = "30.04.2025"
    RESPONSE_TIMEOUT: int = 60

    # DEFINE ASSETS PATHs
    DATASET_PATH: str = "datasets/dataset.json"
    CHAT_HISTORY_PATH: str = "chat_history.json"

    HEADERS: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/114.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    })

    # Confidence thresholds
    HIGH_CONFIDENCE: float = 1.0
    MEDIUM_CONFIDENCE: float = 0.7

    # UI Configuration
    TYPING_SPEED: float = 0.001
    THINKING_ANIMATION_SPEED: float = 0.1

    # Developer mode defaults
    DEFAULT_DEV_MODE: bool = False


class NetworkUtils:
    """Handles all network-related operations"""

    @staticmethod
    def is_connected(url: str = "https://8.8.8.8", timeout: int = 5) -> bool:
        """Check if this device is connected to the internet"""

        try:
            _ = requests.head(url, timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    @staticmethod
    def download_file(url: str, timeout: int = 10, headers: Dict = None) -> Optional[bytes]:
        """Download a file from a URL with error handling"""

        if headers is None:
            headers = Config().HEADERS

        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.content
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download from {url}: {str(e)}")
            return None


class DatasetManager:
    """Handles loading, refreshing and accessing the dataset"""

    def __init__(self, developer_mode: bool = False):
        self.developer_mode = developer_mode
        self._dataset = self._load_dataset()

    def _load_dataset(self) -> Dict:
        """Load the dataset with fallback mechanisms"""

        # Define dataset variable
        dataset: dict = None

        # Check if dataset exist locally
        if os.path.exists(Config.DATASET_PATH):
            try:
                # Load the dataset from local
                with open(Config.DATASET_PATH, "rb") as f:
                    dataset = json.loads(f.read())

            except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:

                if self.developer_mode:
                    print(f"{Fore.RED} ERROR : {e}")

        # Download the dataset online if not found locally
        if not dataset:

            try:

                # Try to download the latest dataset first
                content = NetworkUtils.download_file(Config.DATASET_URL)
                # Read the downloaded data as json
                dataset = json.loads(content.decode("UTF-8"))

                # If in developer mode, also save a local copy
                if self.developer_mode and not os.path.exists(os.path.dirname(Config.DATASET_PATH)):
                    os.makedirs(os.path.dirname(Config.DATASET_PATH), exist_ok=True)

                if self.developer_mode:
                    with open(Config.DATASET_PATH, "wb") as f:
                        f.write(content)

            except json.JSONDecodeError as e:

                # Print out the error
                if self.developer_mode:
                    print(f"{Fore.RED}ERROR : {e}")

        # If all fails, return a minimal working dataset
        if not dataset:
            raise ValueError("Dataset download failed, Snapy cannot run !")

        return dataset

    @property
    def dataset(self) -> Dict:
        """Get the current dataset"""
        return self._dataset

    @property
    def version(self) -> str:
        """Get the dataset version"""
        return self._dataset.get("version", "unknown")

    @property
    def updated_date(self) -> str:
        """Get the dataset last updated date"""
        return self._dataset.get("updated", "unknown")


class ChatHistory:
    """This class contain the chat history manager"""

    def __init__(self, history_path: str = Config.CHAT_HISTORY_PATH):
        self.history_path = history_path
        self.developer_mode = Config.DEFAULT_DEV_MODE

    def add_exchange(self, user_text: str, bot_text: str) -> None:
        """ This method will add the new chat"""

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        messages = [
            {"sender": "user", "message": user_text, "timestamp": timestamp},
            {"sender": "bot", "message": bot_text, "timestamp": timestamp}
        ]

        history = self._load_history()
        history["messages"].extend(messages)
        self._save_history(history)

    def get_recent(self, count: int = 5) -> List[Dict]:
        """ This method will get the recent chat history"""

        try:
            messages = self._load_history().get("messages", [])
            return messages[-count * 2:]

        except Exception:
            return {}

    def clear(self) -> None:
        """ This method will clear the chat history """

        if os.path.exists(self.history_path):
            os.remove(self.history_path)

    def _load_history(self) -> Dict:
        """ This method will load the chat history file"""

        try:

            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception:
            return {"messages": []}

    def _save_history(self, history: Dict) -> None:
        """ This method will save the chat history json file"""

        with open(self.history_path, 'w', encoding="utf-8") as f:
            json.dump(history, f, indent=2)


class Snapy:
    """Main chatbot class integrating all components"""

    def __init__(self, developer_mode: bool = Config.DEFAULT_DEV_MODE, search_mode: bool = False):

        # Track developer mode state
        self.developer_mode = developer_mode
        self.search_mode = search_mode

        # Update the default dev mode
        Config.DEFAULT_DEV_MODE = developer_mode

        # Define is connected online
        self.is_connected: bool = NetworkUtils.is_connected()

        # Initialize dependencies
        self.nlp_engine = NlpEngine(dev_mode=developer_mode)
        self.dataset_manager = DatasetManager(developer_mode)
        self.chat_history = ChatHistory()
        self.search_engine = SearchEngine()

        # Define dialogs dataset
        self.dataset = self.dataset_manager.dataset

        # Context tracking
        self.last_user_input = None
        self.last_bot_response = None
        self.last_confidence = 0.0

    def generate_response(self, user_input: str) -> Tuple[str, float]:
        """
        Generate a chatbot response for a given user input.

        Returns:
            Tuple[str, float]: Response text and confidence score.
        """

        # --- Init ---
        self.last_user_input = user_input
        normalized_message = self.nlp_engine.text_normalization(user_input.strip())
        response: Optional[str] = None
        confidence: float = 0.0

        # --- Empty Input Check ---
        if not normalized_message:
            return "Did you want to ask me something?", 0.0

        # --- Search Mode Override ---
        if self.search_mode:
            if not self.is_connected:
                return f"You are offline. I cannot search about '{user_input}'.", 0.0

            response = self.search_engine.get_answer(normalized_message)
            confidence = 0.6
            self._finalize_response(user_input, response, confidence)

            return self._adjust_response(response), confidence

        try:
            # --- STEP 1: Exact Match ---
            for intent in self.dataset["intents"]:
                if normalized_message in intent.get("patterns", []):
                    response = random.choice(intent.get("responses", ["I’m not sure how to respond to that."]))
                    confidence = 1.0
                    break

            # --- STEP 2: Similarity Match ---
            if not response:
                best_score = 0.0
                best_intent = None

                for intent in self.dataset["intents"]:
                    if "fallback" in intent.get("tag", "").lower():
                        continue
                    for pattern in intent.get("patterns", []):
                        sim = self._calculate_similarity(normalized_message, pattern)
                        if sim > best_score:
                            best_score = sim
                            best_intent = intent

                if best_score >= Config.MEDIUM_CONFIDENCE and best_intent:
                    response = random.choice(best_intent.get("responses", ["Let me think..."]))
                    confidence = best_score

            # --- STEP 3: Online Search Fallback ---
            if not response and self.is_connected and confidence < Config.MEDIUM_CONFIDENCE:
                response = self._adjust_response(self.search_engine.get_answer(normalized_message))
                confidence = 1.0

                # Print online search indicatior when dev mode
                if self.developer_mode:
                    print("Searching Online ...")

            # --- Final Fallback ---
            if not response:
                fallback_intent = self.dataset["intents"][0]
                response = random.choice(fallback_intent.get("responses", ["I’m not sure I understand."]))
                confidence = 0.1

        except Exception:
            if self.developer_mode:
                print("[DEV] Error in response generation:\n" + traceback.format_exc())
            response = "Oops, something went wrong while processing your request."
            confidence = 0.0

        # --- Finalize ---
        self._finalize_response(user_input, response, confidence)
        return self._adjust_response(response), confidence

    def _finalize_response(self, user_input: str, response: str, confidence: float) -> None:
        """Helper to finalize response state."""
        self.chat_history.add_exchange(user_input, response)
        self.last_bot_response = response
        self.last_confidence = confidence

    @staticmethod
    def _adjust_response(_text: str, max_line_length: int = 80) -> str:
        """
        Wrap the response text to a given line length by inserting newlines.
        """
        return '\n'.join(textwrap.wrap(_text, width=max_line_length))

    @staticmethod
    def _calculate_similarity(text_1: str, text_2: str) -> float:
        """
        Calculate the similarity between two texts using multiple methods

        Args:
            text_1: First text string
            text_2: Second text string

        Returns:
            float: Composite similarity score between 0.0 and 1.0
        """

        # 1. Sequence matcher for overall string similarity
        sequence_score = difflib.SequenceMatcher(None, text_1, text_2).ratio()

        # 2. Word-based similarity (handles different word orders)
        text1_words = text_1.split()
        text2_words = text_2.split()

        text1_word_set = set(text1_words)
        text2_word_set = set(text2_words)

        # Exact word match (Jaccard)
        if text1_word_set and text2_word_set:
            intersection = text1_word_set.intersection(text2_word_set)
            union = text1_word_set.union(text2_word_set)
            word_score = len(intersection) / len(union) if union else 0
        else:
            word_score = 0

        # Fuzzy word match tolerance
        fuzzy_matches = 0
        for word in text1_words:
            close = difflib.get_close_matches(word, text2_words, n=1, cutoff=0.8)
            if close:
                fuzzy_matches += 1
        fuzzy_word_score = fuzzy_matches / len(text1_words) if text1_words else 0

        # 3. Key terms matching (words longer than 3 chars have more weight)
        key_text1_words = [w for w in text1_words if len(w) > 3]
        key_text2_words = [w for w in text2_words if len(w) > 3]

        if key_text1_words and key_text2_words:
            key_intersection = set(key_text1_words).intersection(set(key_text2_words))
            key_score = len(key_intersection) / max(len(key_text1_words), len(key_text2_words))
        else:
            key_score = 0

        # 4. Length difference penalty
        len_diff = abs(len(text_1) - len(text_2)) / max(len(text_1), len(text_2)) if max(len(text_1),
                                                                                         len(text_2)) > 0 else 0
        length_penalty = 1 - len_diff

        # Final composite score with fuzzy match support
        composite_score = (
                sequence_score * 0.40 +
                word_score * 0.3 +
                fuzzy_word_score * 0.30 +
                key_score * 0.1 +
                length_penalty * 0.1
        )

        return round(min(composite_score, 1.0), 4)

    def refresh_dataset(self) -> bool:
        """Refresh the dataset from source"""
        return self.dataset_manager.refresh()

    @property
    def dataset_version(self) -> str:
        """Get the current dataset version"""
        return self.dataset_manager.version

    @property
    def dataset_updated(self) -> str:
        """Get the dataset last updated date"""
        return self.dataset_manager.updated_date


class SnapyUI:
    """User interface for Snapy chatbot"""

    def __init__(self, snapy: Snapy):
        self.snapy = snapy
        colorama.init(autoreset=True)

    @staticmethod
    def clear_screen():
        """Clear the terminal screen based on OS"""
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def get_terminal_width() -> int:
        """Get terminal width, with fallback for unsupported platforms"""
        try:
            return os.get_terminal_size().columns
        except (AttributeError, OSError):
            return 80

    @staticmethod
    def print_with_typing_effect(text: str, speed: float = Config.TYPING_SPEED) -> None:
        """Print text with a typing effect"""

        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(speed)

        print()

    def print_header(self):
        """Print a styled header for Snapy CLI"""
        term_width = self.get_terminal_width()
        header = f" | Snapy {' ' * (term_width - 15)} | V {Config.VERSION}"
        print(f"{Fore.CYAN}{Style.BRIGHT}{header}{Style.RESET_ALL}")

    def print_help(self):
        """Print help information"""
        term_width = self.get_terminal_width()

        print(f"\n{Fore.YELLOW}{'=' * term_width}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}{'AVAILABLE COMMANDS':^{term_width}}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * term_width}{Style.RESET_ALL}")

        commands = [
            ("help", "Display this help message"),
            ("clear", "Clear the chat history from screen"),
            ("exit/quit/bye", "Exit the program"),
            ("dev", "Toggle developer mode"),
            ("refresh", "Refresh the dataset"),
            ("history", "Show recent chat history"),
            ("about", "Show information about Snapy"),
            ("reset", "Clear chat history file"),
            ("search <query>", "Search the web for information"),
            ("version", "Display version information")
        ]

        for cmd, desc in commands:
            print(f"{Fore.GREEN}{cmd:20}{Style.RESET_ALL} - {desc}")

        print(f"{Fore.YELLOW}{'=' * term_width}{Style.RESET_ALL}\n")

    def print_about(self):
        """Print about information"""
        term_width = self.get_terminal_width()

        print(f"\n{Fore.CYAN}{'=' * term_width}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'ABOUT SNAPY':^{term_width}}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * term_width}{Style.RESET_ALL}")

        about_text = [
            f"Snapy v{Config.VERSION} - A pattern-matching chatbot with web search capabilities",
            f"Released on: {Config.RELEASE_DATE}",
            f"Current dataset version: {self.snapy.dataset_version}",
            f"Dataset last updated: {self.snapy.dataset_updated}",
            "",
            "Created by Aymen Brahim Djelloul",
            "Licensed under MIT License",
            "",
            "Repository: https://github.com/aymenbrahimdjelloul/Snapy"
        ]

        for line in about_text:
            print(f"{line}")

        print(f"{Fore.CYAN}{'=' * term_width}{Style.RESET_ALL}\n")

    @staticmethod
    def print_thinking_animation():
        """Show 'thinking' animation"""

        print(f"{Fore.BLUE}Snapy: {Style.RESET_ALL}", end="")
        sys.stdout.flush()
        for _ in range(3):
            for c in "...":
                sys.stdout.write(c)
                sys.stdout.flush()
                time.sleep(Config.THINKING_ANIMATION_SPEED)
            sys.stdout.write("\b\b\b   \b\b\b")
            sys.stdout.flush()

        # Clear the thinking animation
        sys.stdout.write("\r" + " " * 40 + "\r")

    def show_recent_history(self):
        """Display recent chat history"""
        print(f"\n{Fore.YELLOW}===== RECENT CHAT HISTORY ====={Style.RESET_ALL}")

        recent_messages = self.snapy.chat_history.get_recent()  # Get last 5 exchanges

        if not recent_messages:
            print(f"{Fore.YELLOW}No chat history found.{Style.RESET_ALL}\n")
            return

        # Display messages in pairs (user + bot)
        for i in range(0, len(recent_messages), 2):
            if i < len(recent_messages):
                user_msg = recent_messages[i]
                print(f"{Fore.GREEN}You  ({user_msg['timestamp']}): {Style.RESET_ALL}{user_msg['message']}")

            if i + 1 < len(recent_messages):
                bot_msg = recent_messages[i + 1]
                print(f"{Fore.BLUE}Snapy ({bot_msg['timestamp']}): {Style.RESET_ALL}{bot_msg['message']}\n")

        print(f"{Fore.YELLOW}{'=' * self.get_terminal_width()}{Style.RESET_ALL}\n")

    def run(self):
        """Start the chatbot interface"""
        # Clear screen and show welcome
        self.clear_screen()
        self.print_header()

        # Display developer mode status if enabled
        if self.snapy.developer_mode:
            print(f"{Fore.YELLOW}Developer Mode: ON{Style.RESET_ALL}")

        # Welcome message
        print(f"\n{Fore.BLUE}Snapy: {Style.RESET_ALL}", end="")
        self.print_with_typing_effect("Hello there! I'm Snapy. How can I help you today?")

        # Main interaction loop
        try:
            while True:
                # User input with styled prompt
                print(f"{Fore.GREEN}You  : {Style.RESET_ALL}", end="")
                user_input = input()

                # Process special commands
                lower_input = user_input.lower().strip()

                # Handle commands
                if self._handle_commands(lower_input):
                    continue

                # Process normal input with "thinking" animation
                self.print_thinking_animation()

                # Get the start generating time
                s_time: float = time.perf_counter()

                # Generate and display response
                response = self.snapy.generate_response(user_input)

                # Display response with typing effect
                print(f"{Fore.BLUE}Snapy: {Style.RESET_ALL}", end="")
                self.print_with_typing_effect(response[0])

                # Show response time and confidence in developer mode
                if self.snapy.developer_mode:
                    confidence = getattr(self.snapy, 'last_confidence', 'N/A')
                    print(f"{Fore.MAGENTA}[Confidence: {confidence:.2f}]{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}Respond in : {time.perf_counter() - s_time:.4f} s\n")

                else:
                    print()  # Empty line for spacing

        except KeyboardInterrupt:
            print(f"\n\n{Fore.BLUE}Snapy: {Style.RESET_ALL}Session ended by user. Goodbye!")
            return

        # except Exception as e:
        #     # More detailed error handling
        #     error_details = traceback.format_exc()
        #
        #     print(f"\n{Fore.RED}ERROR: {Style.RESET_ALL}{str(e)}")
        #
        #     # In developer mode, show stack trace
        #     if self.snapy.developer_mode:
        #         print(f"\n{Fore.RED}=== STACK TRACE ==={Style.RESET_ALL}")
        #         print(error_details)
        #
        #     print(f"\n{Fore.YELLOW}Please report this issue if it persists.{Style.RESET_ALL}")
        #     return

    def _handle_commands(self, lower_input: str) -> bool:
        """
        Handle special commands

        Args:
            lower_input: Lowercase version of user input
            original_input: Original user input with case preserved

        Returns:
            bool: True if a command was handled, False otherwise
        """
        # Exit commands
        if lower_input in ['exit', 'quit', 'bye', 'goodbye', 'q']:
            print(f"\n{Fore.BLUE}Snapy: {Style.RESET_ALL}", end="")
            self.print_with_typing_effect("Goodbye! Have a great day!")
            sys.exit(0)

        # Help command
        elif lower_input == 'help':
            self.print_help()
            return True

        # About command
        elif lower_input == 'about':
            self.print_about()
            return True

        # Clear screen command
        elif lower_input == 'clear':
            # Clear the console screen
            self.clear_screen()
            # Clear the history chat
            self.snapy.chat_history.clear()

            # Print the header
            self.print_header()
            print(f"\n{Fore.GREEN}System: {Style.RESET_ALL}Chat screen cleared.\n")
            return True

        # Toggle developer mode
        elif lower_input == 'dev':
            self.snapy.developer_mode = not self.snapy.developer_mode
            mode_status = "enabled" if self.snapy.developer_mode else "disabled"
            print(f"\n{Fore.GREEN}System: {Style.RESET_ALL}"
                  f"Developer mode {Fore.YELLOW}{mode_status}{Style.RESET_ALL}.\n")
            return True

        # Toggle developer mode
        elif lower_input == 'search':
            self.snapy.search_mode = not self.snapy.search_mode
            mode_status = "enabled" if self.snapy.search_mode else "disabled"
            print(f"\n{Fore.GREEN}System: {Style.RESET_ALL}Search mode {Fore.YELLOW}{mode_status}{Style.RESET_ALL}.\n")
            return True

        # Refresh dataset
        elif lower_input == 'refresh':
            print(f"\n{Fore.GREEN}System: {Style.RESET_ALL}Refreshing dataset...")
            success = self.snapy.refresh_dataset()

            if success:
                print(f"{Fore.GREEN}System: {Style.RESET_ALL}"
                      f"Dataset refreshed successfully (version: {self.snapy.dataset_version}).\n")

            else:
                print(f"{Fore.RED}System: {Style.RESET_ALL}Failed to refresh dataset.\n")
            return True

        # Show chat history
        elif lower_input == 'history':
            self.show_recent_history()
            return True

        # Show version info
        elif lower_input == 'version':
            print(f"\n{Fore.GREEN}Snapy version: {Style.RESET_ALL}{Config.VERSION}")
            print(f"{Fore.GREEN}Dataset version: {Style.RESET_ALL}{self.snapy.dataset_version}")
            print(f"{Fore.GREEN}Release date: {Style.RESET_ALL}{Config.RELEASE_DATE}\n")
            return True

        # Not a command
        return False


def main():
    """Entry point for the application"""

    # Initialize the chatbot
    snapy = Snapy()

    # Start the UI
    ui = SnapyUI(snapy)
    ui.run()


if __name__ == "__main__":
    main()
