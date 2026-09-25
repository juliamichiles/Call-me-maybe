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


# JSON State Machine:
## System Architecture & Workflow

```mermaid
flowchart TD
    classDef force fill:#f2f0ff,stroke:#6554af,stroke-width:2px;
    classDef llm fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef term fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef pipe fill:#fff3e0,stroke:#e65100,stroke-dasharray: 5 5;

    subgraph Pipeline ["Pipeline Loop: generation.py"]
        A([Start: User Prompt]) --> B["_format_prompt()"]
        B --> C["model.encode(prompt)"]
        C --> D["Initialize JSONStateMachine"]
        
        D --> E{"State Machine Complete?"}
        E -- Yes --> Z["json.loads(buffer)"]
        E -- No --> F["get_allowed_token_ids()"]
        
        F --> G{"Any Pending Deterministic Tokens?"}
        G -- Yes --> H["Append Deterministic Tokens to Input IDs"]
        H --> E
        
        G -- No --> I["model.get_logits_from_input_ids()"]
        I --> J["select_next_token()\nMask unallowed logits with -inf"]
        J --> K["state_machine.update(next_token)"]
        K --> L["Append next_token to Input IDs"]
        L --> E
        
        Z --> Y([Return Parsed Function Call])
    end

    subgraph StateMachine ["State Machine Transitions: state_machine.py"]
        direction TB
        
        %% Deterministic States
        S_START["EMIT_START\nAppends: '{\"name\": \"'"]:::force
        S_HEADER["EMIT_PARAMS_HEADER\nAppends: '\", \"parameters\": {'"]:::force
        S_KEY["EMIT_PARAM_KEY\nPops param from queue\nAppends: '\"param_name\": '"]:::force
        S_SEP["EMIT_PARAM_SEP\nAppends: ', '"]:::force
        S_END_DET["EMIT_END\nAppends: '}}'"]:::force

        %% LLM-Driven States
        S_FN["SELECT_FUNCTION\nConstrained by available function names"]:::llm
        S_VAL["SELECT_PARAMETER_VALUE\nConstrained by parameter type\n(string, number, boolean)"]:::llm

        %% Terminal State
        S_END_TERM(("END")):::term

        %% Transitions
        S_START --> S_FN
        S_FN -- "Match complete" --> S_HEADER
        
        S_HEADER -- "queue not empty" --> S_KEY
        S_HEADER -- "queue empty" --> S_END_TERM
        
        S_KEY --> S_VAL
        
        S_VAL -- "Value finished & queue not empty" --> S_SEP
        S_VAL -- "Value finished & queue empty" --> S_END_DET
        
        S_SEP --> S_KEY
        S_END_DET --> S_END_TERM
    end

    %% Mapping between loop and states
    F -. Resolves transitions .-> StateMachine
    K -. State transition trigger .-> StateMachine

