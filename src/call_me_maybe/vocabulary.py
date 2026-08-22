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
        # TODO: might lead to tokens being overwritten depending on the
        # tokenizer/where its called - REMOVE??
        cleaned = token_str.replace("Ġ", " ")
        return cleaned

    def _load_and_build(self) -> None:
        """Reads vocab JSON file and populates mapping tables and Trie."""
        # FIXME: Should this really be a private method?
        # FIXME: Add try/except here or to whoever calls this
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            raw_vocab: Dict[str, int] = json.load(f)

        for raw_token, token_id in raw_vocab.items():
            clean_str = self._clean_token_string(raw_token)
            self.id_to_token[token_id] = clean_str
            self.token_to_id[clean_str] = token_id
            self.trie.insert(clean_str, token_id)

