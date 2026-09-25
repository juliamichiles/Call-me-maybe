_This project has been created as part of the 42 curriculum by juliatav_

![visualization](assets/visualization.png)

##  Optimizations:
uses Coalescing optimization to enhence performance
pre-indexing/precomputation.
vocabulary trie 

# Instructions:
### Run with defaults:
`make run`

### Run with custom INPUT and/or FUNCTIONS files:
`make run INPUT=<path-to-file> FUNCTIONS=<path-to-file>`

### Run with visualization enabled:
`make visualize`

# Trie algorithm:
VocabularyTrie efficiently retrieves token IDs given a prefix in O(L) time (L beingthe prefix length).

# Resources:

### FMS:
- https://www.youtube.com/watch?v=4rNYAvsSkwk

### Function calling in LLMs:
- https://medium.com/@jamestang/llm-function-calling-explained-a-deep-dive-into-the-request-and-response-payloads-894800fcad75

- https://www.youtube.com/watch?v=gosZ_vqXkMI

### LLMs:
- https://www.youtube.com/watch?v=wjZofJX0v4M&pp=ygUOMSBibHVlIDMgYnJvd24%3D
- https://www.youtube.com/watch?v=IHZwWFHWa-w&t=615s&pp=ygUOMSBibHVlIDMgYnJvd24%3D
- https://www.youtube.com/watch?v=xpvFinvqRCA

### JSON schema:
- https://www.youtube.com/watch?v=TAgUvtKLOOE

### Tokens:
- https://www.youtube.com/watch?v=6FIvLzTU_3s

### Constraint decoding:
- https://pub.towardsai.net/constrained-decoding-forcing-llms-to-respect-your-taxonomy-3aaaf13329f9


## System Architecture & Workflow

## Generation Pipeline

```mermaid
flowchart TD
    classDef step fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef terminal fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    A([Start: User Prompt]) --> B["_format_prompt()"]:::step
    B --> C["model.encode(prompt)"]:::step
    C --> D["Initialize JSONStateMachine"]:::step
    
    D --> E{"State Machine Complete?"}:::decision
    E -- Yes --> Z["json.loads(buffer)"]:::step
    E -- No --> F["state_machine.get_allowed_token_ids()"]:::step
    
    F --> G{"Any Pending Deterministic Tokens?"}:::decision
    G -- Yes --> H["Append Deterministic Tokens to Input IDs"]:::step
    H --> E
    
    G -- No --> I["model.get_logits_from_input_ids()"]:::step
    I --> J["select_next_token()<br/>Mask unallowed logits with -inf"]:::step
    J --> K["state_machine.update(next_token)"]:::step
    K --> L["Append next_token to Input IDs"]:::step
    L --> E
    
    Z --> Y([Return Parsed Function Call]):::terminal

flowchart TD
    classDef force fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef llm fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef term fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% Deterministic States
    S_START["EMIT_START<br/>Appends: {&quot;name&quot;: &quot;"]:::force
    S_HEADER["EMIT_PARAMS_HEADER<br/>Appends: &quot;, &quot;parameters&quot;: {"]:::force
    S_KEY["EMIT_PARAM_KEY<br/>Pops param from queue<br/>Appends: &quot;param_name&quot;: "]:::force
    S_SEP["EMIT_PARAM_SEP<br/>Appends: , "]:::force
    S_END_DET["EMIT_END<br/>Appends: }}"]:::force

    %% LLM-Driven States
    S_FN["SELECT_FUNCTION<br/>Constrained by available function names"]:::llm
    S_VAL["SELECT_PARAMETER_VALUE<br/>Constrained by parameter type<br/>(string, number, boolean)"]:::llm

    %% Terminal State
    S_END_TERM(("END")):::term

    %% Transitions
    S_START --> S_FN
    S_FN -- "Match complete" --> S_HEADER
    
    S_HEADER -- "Queue not empty" --> S_KEY
    S_HEADER -- "Queue empty" --> S_END_TERM
    
    S_KEY --> S_VAL
    
    S_VAL -- "Value finished & Queue not empty" --> S_SEP
    S_VAL -- "Value finished & Queue empty" --> S_END_DET
    
    S_SEP --> S_KEY
    S_END_DET --> S_END_TERM
