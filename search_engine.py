"""
@Author : Aymen Brahim Djelloul
Version : 0.1
Date : 06.05.2025
License : MIT

     \\ Snapy is a lightweight and simple chatbot with context tracking and online search capabilities,
        built for educational use and easy customization.
        
    \\ This 'search_engine' module contain the online searching
     and online response generating logic for Snapy bot

"""

# IMPORTS
import sys
import requests
import re
import time
import json
import os
from hashlib import md5
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup


@dataclass
class _Config:

    VERSION: str = "0.1-beta"
    PARSER: str = "html.parser"  # Fallback to default parser if lxml not available
    TIMEOUT: int = 8
    CACHE_DIR: str = ".search_cache"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

    # Search engine configurations
    SEARCH_ENGINES = {
        "DuckDuckGo": {
            "url": "https://html.duckduckgo.com/html/?q={}",
            "selector": ".result__snippet"
        },
        "Qwant": {
            "url": "https://lite.qwant.com/?q={}&t=web",
            "selector": ".result-snippet"
        },
        "Brave": {
            "url": "https://search.brave.com/search?q={}",
            "selector": ".snippet-description"
        }
    }


class SearchEngine:
    """ This class contain the whole search engine logic"""

    def __init__(self, dev_mode: bool = False, cache_enabled: bool = True):
        """
        Initialize the search engine

        Args:
            dev_mode: Enable verbose logging
            cache_enabled: Enable result caching
        """
        self.dev_mode = dev_mode
        self.session = requests.Session()
        self.cache_enabled = cache_enabled

        # Create cache directory if enabled
        if cache_enabled:
            os.makedirs(_Config.CACHE_DIR, exist_ok=True)

        if self.dev_mode:
            print(f"Initialized search engine with {len(_Config.SEARCH_ENGINES)} sources")
            print(f"Cache: {'Enabled' if cache_enabled else 'Disabled'}")

    def get_answer(self, query: str) -> str:
        """
        Get the best answer for a query

        Args:
            query: The search query

        Returns:
            Best answer string
        """
        # Check cache first
        if self.cache_enabled:
            cached = self._get_from_cache(query)
            if cached:
                if self.dev_mode:
                    print(f"[Cache] Using cached result for: {query}")
                return cached

        # Get snippets from all sources
        all_snippets = self._get_all_snippets(query)

        if not all_snippets:
            return "No relevant results found."

        # Process and select best result
        best_result = self._select_best_result(all_snippets, query)

        # Cache the result
        if self.cache_enabled:
            self._save_to_cache(query, best_result)

        return best_result

    def _get_all_snippets(self, query: str) -> List[Dict]:
        """
        Query all search engines and collect snippets

        Args:
            query: Search query

        Returns:
            List of snippet dictionaries with text and source
        """
        processed_query = self._get_query(query)
        all_snippets = []

        # Query each search engine
        for name, engine in _Config.SEARCH_ENGINES.items():
            try:
                # Format URL with query
                url = engine["url"].format(processed_query.replace(" ", "+"))

                # Make request
                response = self.session.get(
                    url,
                    headers=_Config.HEADERS,
                    timeout=_Config.TIMEOUT
                )

                # Check response
                if response.status_code != 200:
                    if self.dev_mode:
                        print(f"[{name}] HTTP Error: {response.status_code}")
                    continue

                # Extract snippets
                html = response.text
                snippets = self._extract_snippets(html, engine["selector"])

                # Add source information
                engine_snippets = [{"text": s, "source": name} for s in snippets]
                all_snippets.extend(engine_snippets)

                if self.dev_mode:
                    print(f"[{name}] Found {len(snippets)} snippets.")

            except Exception as e:
                if self.dev_mode:
                    print(f"[{name}] Error: {str(e)}")

        return all_snippets

    def _extract_snippets(self, html: str, selector: str) -> List[str]:
        """
        Extract snippets using BeautifulSoup

        Args:
            html: Raw HTML content
            selector: CSS selector

        Returns:
            List of snippet texts
        """
        snippets = []

        try:
            # Parse HTML
            soup = BeautifulSoup(html, _Config.PARSER)

            # Find elements matching selector
            elements = soup.select(selector)

            # Extract text
            for element in elements:
                text = element.get_text(strip=True)
                if text:
                    # Clean text
                    text = self._clean_text(text)
                    snippets.append(text)

        except Exception as e:
            if self.dev_mode:
                print(f"Extraction error: {str(e)}")

            # Try fallback with regex if BeautifulSoup fails
            try:
                pattern = r'<[^>]*' + selector.replace(".", "") + r'[^>]*>(.*?)</[^>]*>'
                matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)

                for match in matches:
                    text = self._clean_html(match)
                    if text:
                        snippets.append(text)
            except Exception:
                raise AttributeError(f"Cannot fetch snippets ! {e}")

        return snippets

    def _select_best_result(self, snippets: List[Dict], query: str) -> str:
        """
        Select the best result from all snippets

        Args:
            snippets: List of snippet dictionaries
            query: Original query for relevance scoring

        Returns:
            Best snippet text
        """
        # Remove duplicates
        unique_snippets = []
        seen_texts = set()

        for snippet in snippets:
            text = snippet["text"]
            if text not in seen_texts and len(text) > 20:  # Filter short snippets
                seen_texts.add(text)
                unique_snippets.append(snippet)

        if not unique_snippets:
            return "No useful content found."

        # Score snippets
        scored_snippets = []
        for snippet in unique_snippets:
            score = self._calculate_score(snippet["text"], query)
            scored_snippets.append({
                "text": snippet["text"],
                "source": snippet["source"],
                "score": score
            })

        # Sort by score
        ranked = sorted(scored_snippets, key=lambda x: x["score"], reverse=True)

        if self.dev_mode and ranked:
            print(f"Best result [{ranked[0]['source']}] with score {ranked[0]['score']:.2f}")

        return ranked[0]["text"] if ranked else "No relevant content found."

    def _calculate_score(self, text: str, query: str) -> float:
        """Calculate relevance score for text vs query with improved accuracy"""

        # Improved fallback scoring
        text_lower = text.lower()
        query_lower = query.lower()

        # Extract meaningful terms (exclude common stop words)
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'on', 'in',
                      'at', 'to', 'for', 'with', 'by', 'of', 'about', 'like', 'that',
                      'this', 'what', 'who', 'when', 'where', 'how', 'why', 'which'}

        # Get meaningful query terms (ignoring stop words)
        query_terms = [term for term in query_lower.split() if term not in stop_words]
        if not query_terms and query_lower.split():  # If all terms were stop words, use original terms
            query_terms = query_lower.split()

        # Exact phrase matching (highest importance)
        # Check for complete phrase match with boundaries
        exact_match_score = 0.0
        if query_lower in text_lower:
            # Full exact match
            exact_match_score = 3.0
        elif len(query_terms) > 1:
            # Check for exact phrase matches with word boundaries
            words_pattern = r'\b' + r'\s+'.join(re.escape(term) for term in query_terms) + r'\b'
            if re.search(words_pattern, text_lower):
                exact_match_score = 2.5

        # Term frequency scoring (high importance)
        # Count how many query terms appear in the text and how many times
        term_count = 0
        matched_terms = 0

        for term in query_terms:
            if term in text_lower:
                matched_terms += 1
                # Count occurrences of this term
                term_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))

        # Calculate term score based on percentage of matched terms and frequency
        term_percentage = matched_terms / len(query_terms) if query_terms else 0
        term_frequency = min(term_count / (len(query_terms) * 2), 1.5) if query_terms else 0
        term_score = (term_percentage * 2.5) + term_frequency

        # Content quality scoring (medium importance)
        # Prefer snippets of reasonable length and with proper sentences
        length = len(text)

        # Optimal length is between 100-500 characters
        if length < 50:  # Too short to be useful
            length_score = length / 100  # Linear increase up to 0.5
        elif 50 <= length <= 150:  # Good short answer
            length_score = 0.8
        elif 150 < length <= 500:  # Ideal length
            length_score = 1.0
        elif 500 < length <= 1000:  # Getting too long
            length_score = 1.0 - ((length - 500) / 1000)
        else:  # Too long
            length_score = 0.5

        # Check for sentence structure (proper punctuation, capitalization)
        has_sentence_structure = bool(re.search(r'[A-Z][^.!?]*[.!?]', text))
        sentence_bonus = 0.5 if has_sentence_structure else 0

        # Content quality score combines length and structure
        quality_score = length_score + sentence_bonus

        # Position score (lower importance)
        # Generally, answers at the beginning of snippets are better
        position_score = 0.0
        if query_terms and any(text_lower.startswith(term) for term in query_terms):
            position_score = 0.5

        # Final score combines all factors with appropriate weights
        final_score = (
                (exact_match_score * 1.0) +  # Weight: 1.0
                (term_score * 0.8) +  # Weight: 0.8
                (quality_score * 0.6) +  # Weight: 0.6
                (position_score * 0.4)  # Weight: 0.4
        )

        if self.dev_mode:
            print(f"Scoring (text sample: '{text[:50]}...'):")
            print(f"  - Exact match: {exact_match_score:.1f}")
            print(f"  - Term score: {term_score:.1f} ({matched_terms}/{len(query_terms)} terms)")
            print(f"  - Quality score: {quality_score:.1f}")
            print(f"  - Final score: {final_score:.1f}")

        return final_score

    @staticmethod
    def _clean_html(raw: str) -> str:
        """Remove HTML tags"""
        return re.sub(r'<[^>]+>', ' ', raw).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove unusual Unicode characters
        text = re.sub(r'[\u200b-\u200f]', '', text)
        return text.strip()

    @staticmethod
    def _get_query(text: str) -> str:
        """Normalize query"""
        return text.strip()

    @staticmethod
    def _get_cache_key(query: str) -> str:
        """Generate cache key for query"""
        return md5(query.encode()).hexdigest()

    def _get_from_cache(self, query: str) -> Optional[str]:
        """Get result from cache if available"""

        cache_file = os.path.join(_Config.CACHE_DIR, f"{self._get_cache_key(query)}.json")

        # Check if cache file exists
        if not os.path.exists(cache_file):
            return None

        # Check if cache is still valid (less than 24 hours old)
        if time.time() - os.path.getmtime(cache_file) > 86400:
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("result")
        except Exception:
            return None

    def _save_to_cache(self, query: str, result: str) -> bool:
        """Save result to cache"""
        if not self.cache_enabled:
            return False

        cache_file = os.path.join(_Config.CACHE_DIR, f"{self._get_cache_key(query)}.json")

        try:
            data = {
                "query": query,
                "result": result,
                "timestamp": time.time()
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)

            return True
        except Exception:
            return False

    @property
    def version(self) -> str:
        """ This method returns the search engine current version"""
        return _Config.VERSION


if __name__ == "__main__":
    sys.exit()

    # query = "What is dsg farts ?"
    # engine = SearchEngine(cache_enabled=False)
    # print(engine.get_answer(query))
