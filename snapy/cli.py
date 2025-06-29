"""
This code or file is part of 'Snapy Chatbot' project
copyright (c) 2025, Aymen Brahim Djelloul, All rights reserved.
use of this source code is governed by MIT License that can be found on the project folder.

@author: Aymen Brahim Djelloul
Version: 0.2
Date: 26.06.2025
License: MIT License
"""

# IMPORTS
import os
import sys
import time
import shutil
import datetime
import traceback
from typing import Optional, Tuple, List, Dict, Callable
from .snapy import Snapy, Config

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class Colors:
    """Color constants for terminal output"""
    if COLORAMA_AVAILABLE:
        HEADER, TITLE, PROMPT = Fore.MAGENTA, Fore.CYAN + Style.BRIGHT, Fore.GREEN
        RESPONSE, WARNING, ERROR = Fore.BLUE, Fore.YELLOW, Fore.RED
        DEBUG, RESET = Fore.MAGENTA, Style.RESET_ALL
    else:
        HEADER = TITLE = PROMPT = RESPONSE = WARNING = ERROR = DEBUG = RESET = ""


class Interface:
    """Enhanced User Interface for Snapy chatbot"""

    def __init__(self, snapy: Snapy) -> None:
        self.snapy = snapy
        self.terminal_width = self._get_terminal_width()
        self.session_start = datetime.datetime.now()
        self.message_count = 0

        # Command registry with handlers
        self.commands = {
            'help': (self._show_help, "Display available commands", False),
            'clear': (self._clear_chat, "Clear chat history and screen", False),
            'exit': (self._exit_program, "Exit program (aliases: quit, bye)", False),
            'quit': (self._exit_program, "", False),
            'bye': (self._exit_program, "", False),
            'dev': (self._toggle_dev_mode, "Toggle developer mode", False),
            'history': (self._show_history, "Show recent chat history", False),
            'about': (self._show_about, "Show Snapy information", False),
            'search': (self._toggle_search, "Toggle web search mode", False),
            'version': (self._show_version, "Display version information", False),
            'debug': (self._show_debug, "Show debug information", True),
            'stats': (self._show_stats, "Show session statistics", False),
            'save': (self._save_session, "Save current session", False)
        }

    def _get_terminal_width(self) -> int:
        """Get terminal width with fallback"""
        try:
            return min(shutil.get_terminal_size().columns, 120)  # Cap at 120 for readability
        except (AttributeError, OSError):
            return 80

    def _print_with_effect(self, text: str, speed: float = None,
                          color: str = Colors.RESET, end: str = "\n") -> None:
        """Print text with optional typing effect and color"""
        if speed is None:
            speed = Config.TYPING_SPEED

        print(color, end='')
        if Config.USE_TYPING_EFFECT and len(text) < 200:  # Only for shorter text
            for char in text:
                print(char, end='', flush=True)
                time.sleep(speed)
        else:
            print(text, end='')
        print(Colors.RESET + end, end='', flush=True)

    def _clear_screen(self) -> None:
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _display_header(self, title: str = "") -> None:
        """Display application header"""
        self._clear_screen()
        print()

        # Create border
        border = f"╭{'─' * (self.terminal_width - 4)}╮"
        print(f"{Colors.HEADER}{border.center(self.terminal_width)}{Colors.RESET}")

        # Display title or logo
        display_text = title or f"Snapy {Colors.WARNING}V{Config.VERSION}"
        print(f"{Colors.TITLE}{display_text.center(self.terminal_width)}{Colors.RESET}")

        # Bottom border
        border = f"╰{'─' * (self.terminal_width - 4)}╯"
        print(f"{Colors.HEADER}{border.center(self.terminal_width)}{Colors.RESET}")

        # Show mode indicators
        modes = []
        if self.snapy.developer_mode:
            modes.append(f"{Colors.WARNING}DEV")

        # if self.snapy.search_mode:
        #     modes.append(f"{Colors.WARNING}SEARCH")

        if modes:
            print(f"  {' | '.join(modes)}{Colors.RESET}")

    def _show_thinking(self) -> None:
        """Animated thinking indicator"""
        print(f"{Colors.RESPONSE}Snapy: {Colors.RESET}", end="", flush=True)
        for _ in range(3):
            for dot in "...":
                print(dot, end="", flush=True)
                time.sleep(Config.THINKING_ANIMATION_SPEED)
            print("\b\b\b   \b\b\b", end="", flush=True)
        print("\r" + " " * 40 + "\r", end="", flush=True)

    def _get_user_input(self) -> str:
        """Get validated user input"""
        while True:
            try:
                self._print_with_effect("You  : ", color=Colors.PROMPT, end="")
                user_input = input().strip()
                if user_input:
                    self.message_count += 1
                    return user_input
                print(f"{Colors.ERROR}Please enter a message.{Colors.RESET}")
            except (EOFError, KeyboardInterrupt):
                self._exit_program()

    def _handle_command(self, command: str) -> bool:
        """Execute user commands"""
        cmd = command.lower().strip()
        if cmd in self.commands:
            handler, _, dev_only = self.commands[cmd]
            if dev_only and not self.snapy.developer_mode:
                self._print_with_effect("Command requires developer mode.", color=Colors.ERROR)
                return True
            handler()
            return True
        return False

    def _process_input(self, user_input: str) -> Tuple[str, float]:
        """Process user input and generate response"""
        self._show_thinking()
        start_time = time.perf_counter()
        response = self.snapy.generate_response(user_input)
        response_time = time.perf_counter() - start_time
        return response[0], response_time

    def _display_response(self, response: str, response_time: float) -> None:
        """Display bot response with optional metadata"""
        self._print_with_effect(f"Snapy: {response}", color=Colors.RESPONSE)

        if self.snapy.developer_mode:
            confidence = getattr(self.snapy, '_last_confidence', 0)
            debug = f"[Confidence: {confidence:.2f} | Time: {response_time:.3f}s]"
            print(f"{Colors.DEBUG}{debug}{Colors.RESET}\n")

    # Command handlers
    def _show_help(self) -> None:
        """Display help information"""
        print(f"\n{Colors.WARNING}{'COMMANDS':^{self.terminal_width}}{Colors.RESET}")
        print(f"{Colors.WARNING}{'-' * self.terminal_width}{Colors.RESET}")

        for cmd, (_, desc, dev_only) in sorted(self.commands.items()):
            if not desc or (dev_only and not self.snapy.developer_mode):
                continue
            print(f"{Colors.PROMPT}{cmd:<12}{Colors.RESET} - {desc}")
        print()

    def _clear_chat(self) -> None:
        """Clear chat history and screen"""
        self._clear_screen()
        self.snapy._chat_history.clear()
        self._display_header()
        self._print_with_effect("Chat cleared.", color=Colors.PROMPT)

    def _toggle_dev_mode(self) -> None:
        """Toggle developer mode"""
        self.snapy.developer_mode = not self.snapy.developer_mode
        status = "enabled" if self.snapy.developer_mode else "disabled"
        self._print_with_effect(f"Developer mode {status}.", color=Colors.WARNING)

    def _toggle_search(self) -> None:
        """Toggle web search mode"""
        self.snapy.search_mode = not self.snapy.search_mode
        status = "enabled" if self.snapy.search_mode else "disabled"
        self._print_with_effect(f"Web search {status}.", color=Colors.WARNING)

    def _show_history(self) -> None:
        """Display recent chat history"""
        self._display_header("CHAT HISTORY")
        recent = self.snapy._chat_history.get_recent(10)

        if not recent:
            self._print_with_effect("No chat history available.", color=Colors.WARNING)
            return

        for i in range(0, len(recent), 2):
            if i < len(recent):
                msg = recent[i]
                print(f"{Colors.PROMPT}You  ({msg['timestamp']}): {Colors.RESET}{msg['message']}")
            if i + 1 < len(recent):
                msg = recent[i + 1]
                print(f"{Colors.RESPONSE}Snapy ({msg['timestamp']}): {Colors.RESET}{msg['message']}\n")

        self._wait_key()

    def _show_about(self) -> None:
        """Display about information"""
        self._display_header("ABOUT SNAPY")

        info = [
            ("Description", "Lightweight AI Chatbot with preprocessed dataset and web search"),
            ("Version", Config.VERSION),
            ("Dataset", self.snapy.dataset_version),
            ("Release", Config.RELEASE_DATE),
            ("Developer", f"{Colors.TITLE}Aymen Brahim Djelloul{Colors.RESET}"),
            ("Repository", "https://github.com/aymenbrahimdjelloul/Snapy")
        ]

        for label, value in info:
            print(f"  {Colors.HEADER}• {label}:{Colors.RESET} {value}")

        print()
        self._wait_key()

    def _show_version(self) -> None:
        """Display version information"""
        info = [
            f"Snapy Version: {Config.VERSION}",
            f"Dataset Version: {self.snapy.dataset_version}",
            f"Release Date: {Config.RELEASE_DATE}",
            f"Python: {sys.version.split()[0]}"
        ]
        for item in info:
            self._print_with_effect(item, color=Colors.PROMPT)

    def _show_debug(self) -> None:
        """Display debug information"""
        self._display_header("DEBUG INFO")

        info = [
            f"Python: {sys.version.split()[0]}",
            f"Platform: {sys.platform}",
            f"Terminal: {self.terminal_width} cols",
            f"Colors: {'Yes' if COLORAMA_AVAILABLE else 'No'}",
            f"History: {len(self.snapy._chat_history)} messages",
            f"Session: {(datetime.datetime.now() - self.session_start).total_seconds():.0f}s"
        ]

        for item in info:
            print(f"  {Colors.DEBUG}• {item}{Colors.RESET}")

        print()
        self._wait_key()

    def _show_stats(self) -> None:
        """Display session statistics"""
        duration = datetime.datetime.now() - self.session_start
        stats = [
            f"Session Duration: {duration.total_seconds():.0f}s",
            f"Messages Sent: {self.message_count}",
            f"History Size: {len(self.snapy._chat_history)}",
            f"Developer Mode: {'On' if self.snapy.developer_mode else 'Off'}",
            f"Search Mode: {'On' if self.snapy.search_mode else 'Off'}"
        ]

        for stat in stats:
            self._print_with_effect(stat, color=Colors.PROMPT)

    def _save_session(self) -> None:
        """Save current session to file"""
        try:
            os.makedirs("sessions", exist_ok=True)
            filename = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
            filepath = os.path.join("sessions", filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Snapy Session - {datetime.datetime.now()}\n")
                f.write(f"Duration: {datetime.datetime.now() - self.session_start}\n")
                f.write(f"Messages: {self.message_count}\n\n")

                for msg in self.snapy._chat_history.get_recent(100):
                    role = "You" if msg.get('role') == 'user' else "Snapy"
                    f.write(f"{role} ({msg['timestamp']}): {msg['message']}\n")

            self._print_with_effect(f"Session saved to: {filepath}", color=Colors.PROMPT)
        except Exception as e:
            self._print_with_effect(f"Save failed: {str(e)}", color=Colors.ERROR)

    def _exit_program(self) -> None:
        """Exit program gracefully"""
        farewells = ["Goodbye!", "See you later!", "Until next time!", "Take care!"]
        msg = farewells[datetime.datetime.now().second % len(farewells)]
        self._print_with_effect(f"Snapy: {msg}", color=Colors.RESPONSE)
        sys.exit(0)

    def _wait_key(self, prompt: str = "Press Enter to continue...") -> None:
        """Wait for user input"""
        print(f"\n{Colors.WARNING}{prompt}{Colors.RESET}", end="")
        input()

    def _handle_error(self, error: Exception) -> None:
        """Handle and log errors"""
        self._print_with_effect(f"Error: {str(error)}", color=Colors.ERROR)

        if self.snapy.developer_mode:
            traceback.print_exc()

        # Log to file
        try:
            os.makedirs("logs", exist_ok=True)
            log_file = f"logs/error_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
            with open(log_file, "w") as f:
                traceback.print_exception(type(error), error, error.__traceback__, file=f)
            self._print_with_effect(f"Error logged: {log_file}", color=Colors.WARNING)
        except:
            pass  # Fail silently if logging fails

    def run(self) -> None:
        """Main interaction loop"""
        self._display_header()

        # Welcome message
        welcome = [
            "Hello! I'm Snapy. How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! Ready to assist you."
        ]
        msg = welcome[datetime.datetime.now().second % len(welcome)]
        self._print_with_effect(f"Snapy: {msg}", color=Colors.RESPONSE)

        # Main loop
        while True:
            try:
                user_input = self._get_user_input()

                if self._handle_command(user_input):
                    continue

                response, response_time = self._process_input(user_input)
                self._display_response(response, response_time)

            except KeyboardInterrupt:
                self._exit_program()

            except Exception as e:
                self._handle_error(e)


def main() -> int:
    """Application entry point"""

    try:
        snapy = Snapy(developer_mode=False)
        ui = Interface(snapy)
        ui.run()

        return 0

    except Exception:

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(0)
