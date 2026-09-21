## Idea:
- keep advance_deterministic to force some of the structural JSON chars
- add a similar logic as the one used to select function to select param values:
    - parse user input and use LLM to select some "candidate" parameters from the 
    prompt, later use the logic described above to select the precomputed candidate
    - see possible_solution.md

## Next:
- [ ] more tests with more files
- [ ] add more try/except blocks + add them on imports (for pydantic and other external stuff)
- [ ] pipeline is big and messy, maybe there's some redundant stuff, check it!
- [ ] visualization has issues:
    - [ ] Last token section blinks, I don't like its
    - [ ] make text pink always, regarding of terminal colors
    - [ ] I'm not sure math/stats is actually correct
    - [ ] Add a "final graph" with performance profiling for the whole process
- [ ] run this on 42's pc bc apparently WSL-Ubuntu is slowing the program down 
- [ ] somehow make the Makefile handle flags
- [ ] linters
- [ ] clean files, remove unused stuff
- [ ] README

## Fixed:
- [X] visualization:
    - [X] its printing some invalid tokens I don't even know were they're comming
        from, bc output file is normal and valid
    - [X] I don't like the yellow shade for contraints
    - [X] I'm not sure I like this visualization at all actually
    - [X] add time report        
- [X] Import time and inspect where are the most expensive parts 
- [X] check how debug prints actually impact run time 
- [X] implement simple time-cheap visualization
- [X] Stopped hallucinating numbers, but its still generating malformed JSON
- [X] Not handling getting parameters from LLM well
- [X] Now hallucinates numbers endlessly...
- [X] Generating invalid JSON, but much faster
- [X] (see tests)
- [X] write orchestrator that will call state_machine and trie

## Questions:
- What does the LLM even do with the prompt after encoding it? How is it used?

## Environment and Makefile:
- [ ] Ensure we actually used all dependencies on uv
- Ensure Makefile actually respects subject requirements
- [X] Update Makefile as project grows
    - [X]  add linters
- [ ] can have custom max_line_len in .flake8 file? Don't think so...

## Testing:
- [ ] write more input files, invalid files etc.
- [X] write a "main" for state_machine:
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
