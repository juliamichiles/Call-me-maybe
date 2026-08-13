#!/usr/bin/env python3

from llm_sdk import Small_LLM_Model
from src.call_me_maybe.vocab import VocabManager
import time

def test_vocab() -> None:
    
    start = time.perf_counter()
    print(f"   took {time.perf_counter() - start:.2f}s")

    print("1. Creating model...")
    model = Small_LLM_Model()
    print(f"   took {time.perf_counter() - start:.2f}s")

    print("2. Getting vocab path...")
    vocab_path = model.get_path_to_vocab_file()
    print(f"   took {time.perf_counter() - start:.2f}s")
    
    print("3. Building VocabManager...")
    vocab_mgr = VocabManager(vocab_path)
    print(f"   took {time.perf_counter() - start:.2f}s")

    print(f"Total vocabulary size: {len(vocab_mgr.id_to_token)}")

    print("4. Testing Trie...") 
    # Test searching for tokens starting with '{'
    curly_tokens = vocab_mgr.trie.get_tokens_for_prefix("{")
    print(f"Tokens starting with '{{': {len(curly_tokens)}")

    # Print a few example tokens starting with '{"'
    json_start_tokens = vocab_mgr.trie.get_tokens_for_prefix('{"')
    example_strings = [vocab_mgr.id_to_token[tid] for tid in list(json_start_tokens)[:5]]
    print(f"Sample tokens starting with '{{\"': {example_strings}")

    print(f"   took {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    test_vocab()

