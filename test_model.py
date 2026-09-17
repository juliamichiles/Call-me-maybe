#!/usr/bin/env python3
import time
from llm_sdk import Small_LLM_Model


def test_model():
    
    model = Small_LLM_Model()

    token_lengths = [10, 50, 100, 200, 500, 1000]

    # Create a sufficiently long piece of text once.
    text = "hello " * 2000
    all_input_ids = model.encode(text).tolist()[0]

    print("Testing model inference time:\n")

    for length in token_lengths:
        input_ids = all_input_ids[:length]

        start = time.perf_counter()
        model.get_logits_from_input_ids(input_ids)
        end = time.perf_counter()

        print(
            f"{length:4d} tokens -> "
            f"{end - start:.3f}s"
        )

if __name__ == "__main__":
    test_model()
