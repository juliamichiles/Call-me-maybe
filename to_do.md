## Next:
- [ ] Stopped hallucinating numbers, but its still generating malformed JSON
- [ ] Not handling getting parameters from LLM well
- [ ] Now hallucinates numbers endlessly...
- [ ] Generating invalid JSON, but much faster
- [ ] (see tests)
- [ ] write orchestrator that will call state_machine and trie

## Questions:
- What does the LLM even do with the prompt after encoding it? How is it used?

## Environment and Makefile:
- [ ] Ensure we actually used all dependencies on uv
- Ensure Makefile actually respects subject requirements
- [X] Update Makefile as project grows
    - [X]  add linters
- [ ] can have custom max_line_len in .flake8 file? Don't think so...

## Testing:
- [ ] write a "main" for state_machine:
    - print data structures
    - print state and buffer content for each iteration
    - try to actually see it working


## SRC:
- [ ] actually write stuff on __init__ + proper prettier imports
- [ ] raise custom errors instead of printing them directly on IO?
- [ ] io.py is currently NOT object oriented?
- [X] Added untested and full of errors version of state_machine
- [ ] Test state_machine using pytest

## Before submission:
- [ ] Ensure subject hasn't changed
- [ ] uv.lock exists
- [ ] pyproject.toml exists
- [ ] make run works
- [ ] uv run python -m src works
- [ ] Output JSON is valid and compliant
- [ ] No output/ folder committed

## Structure:
- [ ] Do we really want to keep call_me_maybe subdir inside src?
- [ ] Add call_me to ALL files name, bc why not? as in call_me_schema
- [ ] Main file is too long
- [ ] Some files have almost nothing in them, maybe merge them? like pipeline + io

## General:
- [ ] Ensure all docstrings contains what subject requires and are PEP-whatever
    compliant
- [ ] Maybe state machine is too strict abt input file format, as in:
    - only accepts double quotes for dicts - would single quotes be valid JSON?
    - parameter types have limited names and must be lowercase
- [ ] Taking SO LONG to run, optimize, do something abt it
