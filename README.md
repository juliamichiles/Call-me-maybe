
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
┌─────────────────────────────────────────────────────────────┐
│                    JSONStateMachine                         │
└─────────────────────────────────────────────────────────────┘

 INIT
  │
  ├── store prompt / functions / vocabulary
  ├── state = START
  ├── buffer = ""
  ├── selected_function = None
  ├── param_queue = []
  └── precompute value tokens
  │
  ▼
┌──────────────────────┐
│ MODEL GENERATES TOKEN│◄──────────────────────────────┐
└──────────┬───────────┘                               │
           │                                           │
           ▼                                           │
      update(token)                                    │
           │                                           │
           ├── token ID → token string                 │
           ├── append to buffer                        │
           └── update_internal_state()                 │
                         │                             │
                         ▼                             │
               ┌───────────────────┐                   │
               │ Function selected?│                   │
               └─────────┬─────────┘                   │
                    NO   │   YES                       │
                         │                             │
                         ▼                             │
                  detect function                      │
                         │                             │
                         ▼                             │
                  set param_queue                      │
                         │                             │
                         ▼                             │
                  determine type                       │
                         │                             │
                 ┌───────┼────────┐                    │
                 ▼       ▼        ▼                    │
              NUMBER  STRING  BOOLEAN                  │
                 │       │        │                    │
                 └───────┴────────┘                    │
                         │                             │
                         ▼                             │
                 parameter complete?                   │
                    │           │                      │
                   NO          YES                     │
                    │           │                      │
                    │           ▼                      │
                    │       pop parameter              │
                    │           │                      │
                    └───────────┘                      │
                                                       │
                         │                             │
                         ▼                             │
                 get_allowed_token_ids()               │
                         │                             │
                         ▼                             │
              _get_candidate_tokens()                  │
                         │                             │
                         ├── Trie lookup               │
                         ├── value_tokens              │
                         └── structural targets        │
                         │                             │
                         ▼                             │
              _is_candidate_valid()                    │
                         │                             │
                         ▼                             │
                  allowed token IDs                    │
                         │                             │
                         └──────────────► MODEL ───────┘
                                           │
                                           │
                              all params complete?
                                           │
                                           ▼
                                      buffer ends
                                       with "}}"
                                           │
                                           ▼
                                          END
                                           │
                                           ▼
                                  get_result_dict()
                                           │
                                           ▼
                                      json.loads()
                                           │
                                           ▼
                                    Python dict
    ### add UML State Machine
