from typing import Dict, List, Set, Optional


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.token_ids: Set[int] = set()


class VocabularyTrie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_str: str, token_id: int) -> None:
        """Inserts a token string and its ID into the Trie."""
        
        current = self.root
        current.token_ids.add(token_id)

        for char in token_str:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
            current.token_ids.add(token_id)

    def get_tokens_for_prefix(self, prefix: str) -> Set[int]:
        """Returns all token IDs whose string representations start with 
                the prefix.
        """
        current = self.root
        for char in prefix:
            if char not in current.children:
                return set()
            current = current.children[char]
        return current.token_ids
