#!/usr/bin/env python3
import json
from llm_sdk import Small_LLM_Model


def explore_sdk() -> None:
    
    # Initialize SDK
    model = Small_LLM_Model()

    # Inspect vocab path and load vocab map
    vocab_path = model.get_path_to_vocab_file()
    print(f"Vocab path: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    print(f"Loaded {len(vocab_data)} tokens from vocabulary.")

    # Test encoding
    prompt = "What is the sum of 2 and 3?"
    tokens_ids = model.encode(prompt)

    # If model.encode return a Tensor, convert to list[int] if necessary
    if hasattr(tokens_ids, "tolist"):
        tokens_ids_list = tokens_ids.tolist()
    else:
        tokens_ids_list = list(tokens_ids)

    print(
            f"Prompt encoded to {len(tokens_ids_list)} "
            " token IDs: {tokens_ids_list[:10]}..."
    )

    # Query logits for next token
    logits = model.get_logits_from_input_ids(tokens_ids_list)
    print(f"Returned {len(logits)} logit scores for vocabulary items.")


if __name__ == "__main__":
    explore_sdk()
