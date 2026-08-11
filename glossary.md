# LLM / Constrained Decoding — Mini Glossary

## Core concepts

### Vocabulary

The complete set of **tokens that the model knows**, each associated with a numerical **token ID**.

In this project, the vocabulary file lets us map between token IDs and their string representations. This is important because the decoder needs to determine which tokens are valid at each generation step.

**Think:** *What tokens exist at all?*

---

### Token

A small piece of text from the model's vocabulary. A token is not necessarily a complete word; it can be a word, part of a word, punctuation, or a piece containing a space.

The LLM generates **one token at a time**.

**Think:** *One piece of text the model can generate.*

---

### Token ID / Input ID

The numerical ID corresponding to a token.

For example:

```text
token:       "hello"
token ID:    4281
```

The model works with these numerical IDs rather than directly with human-readable text.

**Think:** *The number representing a token.*

---

### Tokenizer

The component that converts text into tokens and token IDs.

```text
Text
  ↓
Tokenizer
  ↓
Tokens
  ↓
Token IDs
```

In this project, `encode()` performs this conversion. The subject notes that tokenizers may split text into subword units and preserve details such as spaces and punctuation.

**Think:** *The bridge between text and token IDs.*

---

### Tensor

A numerical data structure used by the model. In this project, the tokenizer's `encode()` method returns a tensor containing token IDs.

For now, the important idea is simply:

```text
Tensor → container for numerical model data
```

**Think:** *The data structure carrying the numbers the model works with.*

---

### LLM

The language model processes the input token IDs and produces scores for the possible next tokens.

In this project the default model is **Qwen/Qwen3-0.6B**.

**Think:** *The model predicting what token should come next.*

---

### Logits

The LLM's scores for the possible next tokens.

For every generation step, the model produces a score for each token in the vocabulary. Higher scores indicate tokens the model considers more likely.

```text
token A → 2.1
token B → 7.8   ← more likely
token C → 4.2
```

The project exposes these through `get_logits_from_input_ids()`.

**Important:** logits are scores, not the actual generated tokens.

**Think:** *How much does the model want each possible next token?*

---

## Structure and constraints

### Grammar

A set of rules describing which sequences are **structurally valid**.

In this project, the relevant structure is JSON. The grammar determines things such as where `{`, `}`, `:`, `,`, strings, numbers, etc. can legally appear.

**Think:** *Is this sequence structurally valid JSON?*

---

### Schema

A description of the **required structure and types of the output**.

In this project, `functions_definition.json` defines things such as:

* function name
* parameter names
* parameter types
* return type

The generated function call must follow this schema exactly: required keys must be present, types must match, and extra keys are not allowed.

**Think:** *What exact JSON structure and types does this function call require?*

---

### State

The decoder's knowledge of **where it currently is in the expected output structure**.

For example, while generating a function call, the state might conceptually tell us:

```text
START
→ expecting {
→ expecting "name"
→ expecting :
→ expecting function name
→ expecting ,
→ expecting "parameters"
→ expecting :
→ expecting {
→ expecting parameter name
→ ...
```

The current state determines which tokens can legally come next.

**Think:** *Where am I in the structure, and what can come next?*

---

### Constraint Decoding

A generation technique that **restricts the model's choices at every token-generation step**.

Instead of asking the model to produce valid JSON and hoping it does so, we examine the possible next tokens and only allow those that maintain:

1. valid JSON structure
2. compliance with the expected schema

The subject explicitly requires constrained decoding to guarantee valid, schema-compliant JSON.

**Think:** *Don't just ask the model to follow the rules — enforce the rules while it generates.*

---

### Logit Masking

The operation used to enforce the constraints.

After the LLM produces logits, the decoder identifies invalid tokens and sets their logits to **negative infinity**:

```text
Before:

token A → 5.2
token B → 7.1
token C → 3.4
token D → 6.8


If B and D are invalid:

token A → 5.2
token B → -∞
token C → 3.4
token D → -∞
```

The invalid tokens can no longer be selected.

**Think:** *Remove illegal choices from the model's options.*

---

### Token Selection

Choosing the next token from the remaining valid tokens after the constraints have been applied.

Conceptually:

```text
LLM → logits
       ↓
constraint check
       ↓
mask invalid tokens
       ↓
select valid token
```

The subject describes this as the final stage of each generation step.

**Think:** *Which allowed token do we actually generate?*

---

### Generation

The complete process of producing the output **one token at a time**.

After selecting a token, it is added to the sequence, the model produces new logits, and the process repeats until the output is complete.

```text
Input IDs
   ↓
LLM
   ↓
Logits
   ↓
Constraint decoding
   ↓
Token selection
   ↓
Next token
   ↓
Update sequence
   ↓
Repeat
```

**Think:** *The loop that generates the complete response.*

---

## How the concepts fit together

The basic LLM generation pipeline is:

```text
Prompt
  ↓
Tokenization
  ↓
Input / Token IDs
  ↓
LLM
  ↓
Logits
  ↓
Token Selection
```

The project inserts **constrained decoding** before token selection:

```text
                         ┌─────────────────────┐
                         │  Grammar + Schema   │
                         │         ↓           │
Prompt → Tokenizer → IDs → LLM → Logits → State
                                      │
                                      ↓
                               Valid tokens
                                      │
                                      ↓
                               Logit masking
                                      │
                                      ↓
                              Token selection
                                      │
                                      ↓
                               Next token
                                      │
                                      └──→ repeat
```

### The three key questions

| Concept        | Main question                                 |
| -------------- | --------------------------------------------- |
| **Vocabulary** | What tokens exist?                            |
| **Grammar**    | What token sequences are structurally valid?  |
| **Schema**     | What output structure and types are required? |

### The key relationship

> **The LLM produces logits for all possible next tokens. The decoder uses the current state, grammar, and schema to determine which tokens are valid, masks the invalid ones, and then selects the next token.**

This happens **token-by-token** until the complete function call has been generated.

