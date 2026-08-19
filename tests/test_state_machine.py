import pytest
from src.call_me_maybe.schemas import FunctionDefinition, ParameterProperty, ReturnSpec
from src.call_me_maybe.state_machine import JSONStateMachine, CurrentState
from src.call_me_maybe.errors import CallMeError

# 1. Pytest Fixtures: Reusable setup data
@pytest.fixture
def sample_functions():
    """Returns a list of sample FunctionDefinitions for testing."""
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers together.",
            parameters={
                "a": ParameterProperty(type="number", description="First number"),
                "b": ParameterProperty(type="number", description="Second number"),
            },
            returns=ReturnSpec(type="number"),
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet a person.",
            parameters={
                "name": ParameterProperty(type="string", description="Name")
            },
            returns=ReturnSpec(type="string"),
        ),
    ]


@pytest.fixture
def mock_vocab_manager():
    """Mock VocabularyManager mapping token IDs to token strings."""
    class MockVocabManager:
        def __init__(self):
            self.id_to_token = {
                1: '{"name": "',
                2: 'fn_add_numbers',
                3: '", "parameters": {',
                4: '"a": ',
                5: "5",
                6: ', "b": ',
                7: "10",
                8: "}}",
            }

        def get_all_tokens(self):
            return self.id_to_token

    return MockVocabManager()


# 2. Pytest Test Cases
def test_initial_state(sample_functions, mock_vocab_manager):
    """Test state machine initializes in START state."""
    sm = JSONStateMachine("Sum 5 and 10", sample_functions, mock_vocab_manager)
    assert sm.get_current_state() == CurrentState.START
    assert not sm.is_complete()


def test_function_selection_transition(sample_functions, mock_vocab_manager):
    """Test transition when a function name token is fed."""
    sm = JSONStateMachine("Sum 5 and 10", sample_functions, mock_vocab_manager)
    
    # Update with header token
    sm.update('{"name": "fn_add_numbers"')
    
    assert sm.selected_function is not None
    assert sm.selected_function.name == "fn_add_numbers"
    assert sm.get_current_state() == CurrentState.CHOOSE_FUNCTION


def test_full_generation_flow(sample_functions, mock_vocab_manager):
    """Test full sequence of token updates reaching END state."""
    sm = JSONStateMachine("Sum 5 and 10", sample_functions, mock_vocab_manager)

    tokens = [
        '{"name": "',
        'fn_add_numbers',
        '", "parameters": {',
        '"a": ',
        '5',
        ', "b": ',
        '10',
        '}}'
    ]

    for tok in tokens:
        sm.update(tok)

    assert sm.is_complete()
    assert sm.get_current_state() == CurrentState.END
    
    # Test result dictionary extraction
    res = sm.get_result_dict()
    assert res == {
        "name": "fn_add_numbers",
        "parameters": {"a": 5, "b": 10}
    }


def test_invalid_json_result_raises_error(sample_functions, mock_vocab_manager):
    """Test that get_result_dict raises ValueError if JSON is incomplete."""
    sm = JSONStateMachine("Sum 5 and 10", sample_functions, mock_vocab_manager)
    sm.update('{"name": "fn_add_numbers"')

    with pytest.raises(CallMeError):
        sm.get_result_dict()


