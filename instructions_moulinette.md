from the main project root:

uv run python -m src \
    --input moulinette/data/input/function_calling_tests.json \
    --functions_definition moulinette/data/input/functions_definition.json

Then grade:

cd moulinette
uv run python -m moulinette grade_student_answers \
    --set private \
    --student_answer_path ../data/output/function_calling_results.json
