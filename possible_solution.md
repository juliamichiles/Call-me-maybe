Possible solution:

https://chatgpt.com/s/t_6a9abe6117248191906e538110057984

----
12/09/26 - new possible state machine feedback:
https://chatgpt.com/s/t_6aa5532dc4c88191ac925f587135a59e

12/09/26 - pipeline after modifying state machine:
https://chatgpt.com/s/t_6aa5532dc4c88191ac925f587135a59e


Yes — **the architecture you're proposing is compatible with the subject**, with one important distinction:

> The LLM should still be the component that *selects/extracts* the parameter values. Your constrained decoder should then make sure that whatever the LLM generates is a valid value according to the schema.

That is actually quite aligned with what the subject asks you to demonstrate.

The key requirements are that constrained decoding must enforce **both JSON structure and schema**, and that the function choice itself must be made using the LLM rather than heuristics.  The subject also explicitly expects argument extraction from natural-language prompts and says the implementation will be tested with different function sets/prompts.  

### Your idea, conceptually

I would structure it roughly like this:

```text
User prompt
    │
    ▼
LLM
    │
    ├── constrained function selection
    │       ↓
    │   fn_greet
    │
    └── constrained parameter extraction
            ↓
        "john"
            │
            ▼
       JSON generation
            │
            ▼
{
    "name": "fn_greet",
    "parameters": {
        "name": "john"
    }
}
```

Your existing:

```python
while self.advance_deterministic():
    pass
```

is a good mechanism for the **grammar/structural side**.

For example, after generating:

```json
{"name":
```

the state machine knows that the next thing must be a value.

But at that point, instead of allowing *every possible JSON string*, you can have another LLM-driven selection state that determines what value is appropriate.

---

## The part I'd change in your idea

I would **not** have the LLM first produce a completely unrestricted "candidate parameter" and then simply select that candidate deterministically.

That risks turning your system into:

```text
LLM → guess value → constrained decoder just accepts it
```

rather than:

```text
LLM logits → constrained decoder → valid next token
```

The latter is what the subject is really emphasizing.

The subject describes constrained decoding as intervening **at every token-selection step**, identifying valid tokens and setting invalid-token logits to negative infinity. 

So I'd make your candidate idea a little more integrated.

### For example

Suppose:

```text
Prompt:
"Greet John"
```

and:

```json
{
    "name": "fn_greet",
    "parameters": {
        "name": {
            "type": "string"
        }
    }
}
```

After selecting `fn_greet`, your state machine reaches:

```text
SELECT_PARAMETER_VALUE
```

The LLM sees something conceptually like:

```text
Greet John

Generate the value for parameter "name".
```

Then its logits might favor tokens corresponding to:

```text
John
```

Your decoder constrains those logits according to the fact that the state expects a JSON string.

So the generation becomes:

```text
"
J
o
h
n
"
```

with your state machine ensuring that every token remains legal JSON/schema-wise.

---

# Where your "candidate" idea is useful

There **is** a very reasonable optimization/design where you first extract candidates from the prompt.

For example:

```text
"Greet John"
       │
       ▼
candidate extraction
       │
       ▼
["John"]
       │
       ▼
LLM/constrained selection
       │
       ▼
"John"
```

This can be useful, especially for numbers.

For:

```text
"What is the sum of 265 and 345?"
```

you might identify:

```text
candidate numbers = [265, 345]
```

and then the constrained generation chooses between them for:

```text
a → 265
b → 345
```

However, **I would be careful about implementing candidate extraction using regex as the primary semantic mechanism.**

The subject explicitly says:

> "The function to call should be chosen using the LLM, not with heuristics or any other sort of medieval magic." 

It doesn't explicitly prohibit heuristics for parameter extraction, but the overall goal is natural-language → function call, and the performance requirement explicitly includes **argument extraction**. 

So I wouldn't build something like:

```python
if "sum" in prompt:
    ...
if re.search(r"\d+", prompt):
    ...
```

That would become fragile very quickly, especially because the evaluator can change the functions and prompts.

---

# Your existing function-selection mechanism

Your current logic:

```python
if self.current_state == State.SELECT_FUNCTION:
    fn_names = [fn_def.name for fn_def in self.functions]

    for fn_name in fn_names:
        if fn_name.startswith(self.fn_name_buffer):
            target = fn_name[len(self.fn_name_buffer):]

            for token_id, token_str in self.vocab_mgr.id_to_token.items():
                if target.startswith(token_str):
                    allowed_ids.add(token_id)
```

is conceptually good **as constrained decoding**.

The important distinction is:

```text
LLM decides which function is likely
              ↓
state machine restricts possible tokens
              ↓
LLM chooses among those tokens
```

rather than:

```text
Python decides which function is correct
              ↓
state machine outputs it
```

The latter would violate the explicit requirement that function selection be done using the LLM.

---

# I'd use the same pattern for parameters

You could have states along these lines:

```text
START
  ↓
{
  ↓
"name":
  ↓
SELECT_FUNCTION
  ↓
,
  ↓
"parameters":
  ↓
{
  ↓
SELECT_PARAMETER
  ↓
:
  ↓
SELECT_PARAMETER_VALUE
  ↓
...
```

And then `get_allowed_token_ids()` becomes state-dependent:

```python
if self.current_state == State.SELECT_FUNCTION:
    return self.allowed_function_tokens()

if self.current_state == State.SELECT_PARAMETER:
    return self.allowed_parameter_tokens()

if self.current_state == State.SELECT_PARAMETER_VALUE:
    return self.allowed_value_tokens()
```

The important part is that **`allowed_value_tokens()` depends on the parameter's schema type**.

For example:

```text
number
    ↓
only tokens that can continue a valid JSON number

string
    ↓
only tokens that can continue a JSON string

boolean
    ↓
true / false

...
```

This directly corresponds to the subject's requirement that schema constraints be enforced during generation. It even explicitly gives numbers as an example: the decoder should restrict generation to values satisfying the integer/float requirement. 

And importantly, the subject says `number, string, boolean, etc.` — so I would **not design your state machine around only three hard-coded types**. The exact supported schema should be derived from the input function definitions and your implementation should be extensible. The supplied examples are explicitly not exhaustive.  

---

## One thing I'd strongly recommend

Separate these two responsibilities:

### 1. Semantic decision

```text
"What does the user want?"
```

LLM.

Examples:

```text
"Greet Shrek"
        ↓
fn_greet

"Reverse hello"
        ↓
fn_reverse_string
```

and:

```text
"Greet Shrek"
        ↓
name = "Shrek"
```

### 2. Syntactic/schema enforcement

```text
"Is fn_greet a legal function name?"
"Is this a legal JSON string?"
"Is this number valid?"
"Can this token legally occur here?"
```

State machine / constrained decoder.

That separation will make your project **much easier to explain during evaluation**.

---

## And yes, `advance_deterministic()` fits nicely

I'd actually keep your current idea of:

```python
while self.advance_deterministic():
    pass
```

because you're effectively saying:

> "If the next transition doesn't require an LLM decision, don't waste a model inference step on it."

For example:

```text
{                     deterministic
 ↓
"name"                deterministic
 ↓
:                     deterministic
 ↓
SELECT_FUNCTION       ← LLM
 ↓
,                     deterministic
 ↓
"parameters"          deterministic
 ↓
:                     deterministic
 ↓
{                     deterministic
 ↓
SELECT_PARAMETER      ← LLM
 ↓
:                     deterministic
 ↓
SELECT_VALUE          ← LLM
```

That's a sensible state-machine architecture.

It also matches the subject's description that generation proceeds token-by-token and that constrained decoding is applied during token selection. 

### So, in short

**Yes, your idea is viable.** I'd phrase it as:

> **Use the LLM for semantic selection/extraction, and use the state machine/constrained decoder to restrict every generated token to the set of values that can still produce valid JSON conforming to the selected function's schema.**

Your "precomputed candidates" idea can be added as an **optimization/assistance mechanism**, but I wouldn't make regex/candidate extraction the thing that *decides* the parameter. The LLM should remain responsible for the semantic choice, just as it is required to be for function selection.

And this approach gives you a very clean story for the README's required **"Algorithm explanation"** and **"Design decisions"** sections, which the subject explicitly requires. 

