from typing import Dict
import json 

from .trie import VocabularyTrie


class VocabularyManager:
    """Loads vocabulary and maps token IDs to clean string representations."""

    def __init__(self, vocab_file_path: str) -> None:
        self.vocab_path = vocab_file_path
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {} # REMOVE?? Not sure I'll use
        self.trie = VocabularyTrie()
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

