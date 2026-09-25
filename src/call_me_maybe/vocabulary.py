from typing import Dict, DefaultDict, Set
import json
from collections import defaultdict

from .trie import VocabularyTrie


class VocabularyManager:
    """Loads vocabulary and maps token IDs to clean string representations."""

    def __init__(self, vocab_file_path: str) -> None:
        """Load vocabulary from file."""
        self.vocab_path = vocab_file_path
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: DefaultDict[str, Set[int]] = defaultdict(set)
        self.trie = VocabularyTrie()

        self.quote_ids: Set[int] = set()
        self.valid_string_body_ids: Set[int] = set()
        self.valid_string_all_ids: Set[int] = set()
        self.number_start_ids: Set[int] = set()
        self.number_body_ids: Set[int] = set()
        self.boolean_ids: Set[int] = set()
        self.delimiter_ids: Set[int] = set()

        self._load_and_build()

    def _clean_token_string(self, token_str: str) -> str:
        """Converts tokenizer space markers (e.g., 'Ġ') to standard spaces."""
        cleaned = token_str.replace("Ġ", " ")
        return cleaned

    def _load_and_build(self) -> None:
        """Reads vocab JSON file and populates mapping tables and Trie."""
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            raw_vocab: Dict[str, int] = json.load(f)

        for raw_token, token_id in raw_vocab.items():
            clean_str = self._clean_token_string(raw_token)
            self.id_to_token[token_id] = clean_str

            # Map both raw and cleaned token variants to token IDs
            self.token_to_id[clean_str].add(token_id)
            self.token_to_id[raw_token].add(token_id)
            self.trie.insert(clean_str, token_id)
            self.trie.insert(raw_token, token_id)

            # --- Precompute Categories ---
            if clean_str == '"':
                self.quote_ids.add(token_id)

            if not any(c in clean_str for c in ['\n', '\r', '\x00']):
                self.valid_string_body_ids.add(token_id)

            clean_stripped = clean_str.strip()
            if clean_stripped:
                if clean_stripped in ['-', '.'] or \
                        clean_stripped.replace('.', '', 1).isdigit():
                    self.number_start_ids.add(token_id)
                if clean_stripped.replace('.', '', 1).isdigit():
                    self.number_body_ids.add(token_id)
                if clean_stripped in ['true', 'false']:
                    self.boolean_ids.add(token_id)
                if clean_stripped in [',', '}']:
                    self.delimiter_ids.add(token_id)

        self.valid_string_all_ids = self.quote_ids | self.valid_string_body_ids

    def token_ids_that_prefix(self, text: str) -> Set[int]:
        """Return token IDs whose complete token string is a prefix of text."""
        return self.trie.get_token_ids_that_prefix(text)
