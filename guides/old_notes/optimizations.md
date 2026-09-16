Short answer — the two runtime crashes you saw were caused by mismatched types in the vocabulary/state-machine code (now fixed): token_to_id needed to be a mapping string -> set[int] and the state machine must merge those sets into the candidate set (use update instead of add). Those fixes remove the KeyError('!') and the TypeError: unhashable type: 'set'.

Why your run is still slow (high-level)

The code runs a fairly expensive candidate-selection + validation loop on every token generation step. The heavy parts are:
many Trie lookups (get_tokens_for_prefix) for each remaining target fragment,
iterating over/merging sets of token IDs repeatedly,
calling _is_candidate_valid for every candidate token (string operations + regex checks),
repeated short-string computations (overlap detection / remainder slicing) each step.
These are algorithmic hotspots (work per generation step can be large), so you need algorithmic caching and cheaper checks rather than micro-optimizations.
Concrete, prioritized optimization opportunities (where and how)

High impact — cache Trie prefix lookups
Problem: get_tokens_for_prefix(remainder) is called repeatedly for the same remainder fragments across steps.
Fix: add an LRU cache of prefix -> tuple(token_ids). Example wrapper (put in VocabularyManager or as a thin wrapper around the trie method):
Python example: from functools import lru_cache class VocabularyManager: ... @lru_cache(maxsize=4096) def tokens_for_prefix(self, prefix: str) -> tuple[int, ...]: return tuple(self.trie.get_tokens_for_prefix(prefix))
Use tuple for cacheable, hashable values; convert back to set when merging (candidate_ids.update(tuple)).
Expected: large reduction in repeated tree traversal, especially for repeated prefixes.
High impact — avoid repeated work in state_machine._get_candidate_tokens
Problems:
Overlap calculation (for every i) and remainder prefix loops are repeated every token.
candidate set is recomputed from scratch every time.
Fixes:
Cache candidate_ids per remainder (or per remainder + selected_function + current_state): a small LRU keyed by remainder string (or remainder + state) that returns the candidate set. Merge quickly rather than rebuild.
When you compute prefix_sub loop, use candidate_ids.update(token_set) (you already changed) — don't call add on a set.
Precompute the set of structural tokens (e.g., punctuation, comma, braces, quote, colon, space) and reuse them instead of scanning full vocab.
Expected: significantly fewer set unions and string ops per step.
Medium impact — avoid validating every candidate with expensive checks
Problem: _is_candidate_valid(combined) can run regex and string logic for each candidate token.
Fixes:
Fast pre-filter: check the candidate token's first n characters against the next expected literal characters in the target before calling the full validator.
Batch validation: if many candidates share the same prefix, validate once for that prefix and apply to all.
Memoize _is_candidate_valid for recently seen combined strings (LRU cache with small maxsize).
Expected: reduces expensive regex invocations.
Medium impact — optimize VocabularyManager and Trie usage
Suggestions:
Only insert unique strings into the trie. Currently both raw_token and cleaned token are inserted; if they are identical (or one maps to the other) avoid duplicate insertion.
Clean tokens once at load time (you already have _clean_token_string; ensure it's called once per token and stored).
In the Trie, ensure nodes store token IDs at insert time so get_tokens_for_prefix can return stored token IDs without a full subtree traversal. If not present, augment nodes to hold the set of token IDs under that node.
Expected: faster prefix queries and less traversal overhead.
Low/medium — reduce allocations & repeated string operations
Move compiled regexes to module-level constants (compile once).
When checking overlaps, avoid creating many substring objects — use indices when possible.
Use local variable bindings for often-accessed items (e.g., get_tokens_for_pref = self.vocab_mgr.trie.get_tokens_for_prefix), which you already do in parts.
Architectural — reduce candidate universe in STRING state
Currently, STRING state returns the entire id_to_token.keys() which can be huge. If you can restrict allowed tokens during string generation:
Limit to tokens that can start a JSON string (quote, escape sequences) or tokens that appear in your functions' parameter value samples, or
Allow unconstrained strings but only let the model produce tokens (no validation loop) until the closing quote is emitted.
Expected: large performance wins when the vocab is big.
Instrumentation & profiling (must do before/after)
Use cProfile or pyinstrument to measure hot functions. Example: python -m cProfile -o prof.out -m src Then run snakeviz or pstats to inspect.
Add simple timing (time.perf_counter) around:
vocabulary loading
Trie lookups
_get_candidate_tokens
_is_candidate_valid
Expected: tells you which optimization yields the best return.
Files / places to focus (from repo)

src/call_me_maybe/vocabulary.py — load and store token mappings; good place to add prefix cache wrapper and to avoid duplicate trie inserts.
src/call_me_maybe/trie.py — ensure efficient storage per node and consider caching common prefix results.
src/call_me_maybe/state_machine.py — main hot loop (_get_candidate_tokens, get_allowed_token_ids, _is_candidate_valid). Add caching and pre-filters here.
src/call_me_maybe/pipeline.py — check how often get_allowed_token_ids is called per model step; reduce calls if possible (e.g., reuse allowed_ids when buffer didn't change in a meaningful way).
