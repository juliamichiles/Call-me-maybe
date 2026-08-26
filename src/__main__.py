from pathlib import Path
import argparse
import json
import sys

from llm_sdk import Small_LLM_Model

from call_me_maybe.io import load_functions_definition, load_input_prompts
from call_me_maybe.vocabulary import VocabularyManager
from call_me_maybe.pipeline import Generation
from call_me_maybe.errors import CallMeError

def main() -> None:
    
    parser = argparse.ArgumentParser(
            description="Constrained Decoding Function Caller"
    )
    parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json",
            help="Path to functions definition JSON file",
    )
    parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
            help="Path to input prompts JSON file",
    )
    parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json",
            help="Path to output results JSON file",
    )

    args = parser.parse_args()

    try:

        functions = load_functions_definition(args.functions_definition)
        prompts = load_input_prompts(args.input)

        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
        vocab_mgr = VocabularyManager(vocab_path)
        try:
            VocabularyManager.tokens_for_prefix.cache_clear()
        except Exception:
            try:
                vocab_mgr.tokens_for_prefix.cache_clear()
            except Exception:
                pass
        
        generator = Generation()
        results = []

        for prompt_obj in prompts:
            call_result = generator.gen_function_call(
                    model=model,
                    prompt_txt=prompt_obj.prompt,
                    functions=functions,
                    vocab_mgr=vocab_mgr
            )
            results.append(call_result)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    except CallMeError as e:
        print(f"Application error: {e}", file=sys.stderr)
        sys.exit(1)
    # except Exception as e:
        # FIXME: more specific errors
        # print(f"Unexpected error: {e}", file=sys.stderr)
        # sys.exit(1)

if __name__ == "__main__":
    main()
