B. Medium-term / high-ROI performance improvements Apply these changes after the immediate fix — they will make the program much faster and eliminate the heavy CPU usage.

Per-instance caches (do this now)
Make the tokens_for_prefix and cached_candidates_for_remainder caches per-instance (not class-level). That ensures a fresh cache for every run and you won't need to clear caches manually. I gave sample code earlier — in short:
In VocabularyManager.init, create self._tokens_for_prefix = lru_cache(maxsize=4096)(self.trie.get_tokens_for_prefix) and implement tokens_for_prefix(fn) to return tuple(self._tokens_for_prefix(prefix)).
In JSONStateMachine.init, create a bound cached helper for remainder -> tuple(ids).
Benefit: no more global stale cache issues and better locality.
Memoize _is_candidate_valid results (small LRU)
_is_candidate_valid is expensive (string concat + regexes). Add an LRU keyed by (len(current_buffer), candidate_str) or (current_buffer_suffix, candidate_str). You can do a tiny per-instance cache with functools.lru_cache on a helper function that takes current_buffer_tail and candidate_str. That reduces repeated recalculation across many candidate tokens that share the same candidate_str or same buffer tail.
Pre-filter cheaply before regex
For number checks, check first character(s) of candidate_str: if candidate_str[0] is digit/space/dot/minus/comma/brace, accept quickly, else only then do the regex. Avoid expensive re.match for cheap rejects/accepts.
Reduce candidate universe in STRING state
Returning all ids when in STRING is expensive. Options:
Limit to tokens that can start readable string content: quote, space, alphabetic subwords. Or allow the model to generate unconstrained for the body of a string until it emits a closing quote (i.e., stop validating tokens in STRING state except to enforce the closing quote).
Another approach: allow only tokens that appear in function param names or typical words in your small dataset while debugging.
Use more compact token representations for comparisons
When comparing prefixes/suffixes, operate on normalized short suffixes (last 32 chars) rather than the entire buffer every time to reduce string allocations. For _matches_prefix you already normalize by removing whitespace, but limiting length helps.
Profile and iterate
Run a short profiling run to confirm hotspots: python -m cProfile -o prof.out -m src --functions_definition ... (--input ...) Then use snakeviz or pstats to inspect which functions use most CPU (likely _get_candidate_tokens and _is_candidate_valid). Optimize those first.
C. Testing & next steps

Apply the short-term fast-path changes first and re-run to confirm generation completes without the empty-token error.
Then implement per-instance caches and memoized validator. Re-run and measure the runtime improvement.
If you want, I can prepare the exact patch (a commit) that:
adds the fast-path changes to get_allowed_token_ids,
converts tokens_for_prefix to a per-instance lru_cache in VocabularyManager,
makes the remainder-candidates cache per-instance in JSONStateMachine,
and adds a small LRU memoization wrapper for _is_candidate_valid.
Which would you like me to do next?

I can prepare the minimal crash-avoiding patch now (fast-paths) and commit it, or
prepare the full performance patch (per-instance caches + validator memo) and commit that in one go.
If you prefer to apply patches yourself, I can paste the exact diff for get_allowed_token_ids and the cache changes.
