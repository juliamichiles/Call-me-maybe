PYTHON = uv run python

INPUT = data/input/function_calling_tests.json
FUNCTIONS = data/input/functions_definition.json
OUTPUT = data/output/function_calling_results.json

all: run

install:
	uv sync

run:
	$(PYTHON) -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	$(PYTHON) -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	rm -rf data/output/*.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

re: clean run

.PHONY: all install run debug clean lint lint-strict re
