# LexiconConstraint Visual Diagrams

---

## 1. Class Hierarchy

```mermaid
classDiagram
    class LexiconConstraint {
        +blank_index : int
        +word_boundary_token : int
        +trie : Dict
        +all_valid_tokens : Set[int]
        +word_list : List[str]
        -_cache : Dict
        +from_file_paths(tokens_file, lexicon_file) LexiconConstraint
        -_load_tokens_file(tokens_file) tuple
        -_load_lexicon_file_with_words(lexicon_file, token_to_id) tuple
        -_build_trie(lexicon) Dict
        -_get_all_valid_tokens() Set[int]
        +get_valid_next_tokens(sequence) Set[int]
        +get_valid_next_tokens_with_word_info(sequence) Tuple
        +get_word_alternatives(phoneme_sequence) List[str]
        +decode_sequence_to_words(token_ids, ...) str
        +get_constraint_mask(sequences, last_labels, vocab_size) Tensor
        +clear_cache()
    }

    class VectorizedLexiconConstraint {
        +supports_state_tracking : bool
        +transition_table : Tensor [num_states × vocab]
        -_sink_state : int
        -_end_state_mask : Tensor [num_states]
        -_state_to_word_indices : Dict[int, List[int]]
        -_mask_buffer : Tensor
        +initialize_state(batch_size, beam_size) Tensor
        +get_constraint_mask_with_state(state, vocab_size, last_labels) Tensor
        +update_state(parent_state, emitted_labels, prev_last_labels) Tensor
        +get_words_at_state(state) List[int]
        -_build_dense_transition_table(device) tuple
        -_infer_vocab_ceiling() int
        -_ensure_table_device(device)
    }

    LexiconConstraint <|-- VectorizedLexiconConstraint : extends
```

---

## 2. Trie Structure (Example: "cat", "car", "dog")

Token IDs: `c=1, a=2, t=3, r=4, d=5, o=6, g=7, |=8`

```mermaid
graph TD
    ROOT(["ROOT (start / after word boundary)"])

    ROOT -->|"1 (c)"| C(["c"])
    ROOT -->|"5 (d)"| D(["d"])

    C -->|"2 (a)"| CA(["a"])

    CA -->|"3 (t)"| CAT(["t"])
    CA -->|"4 (r)"| CAR(["r"])

    CAT -->|"8 (BND)"| CAT_END["__end__: [idx_cat]"]
    CAR -->|"8 (BND)"| CAR_END["__end__: [idx_car]"]

    D -->|"6 (o)"| DO(["o"])
    DO -->|"7 (g)"| DOG(["g"])
    DOG -->|"8 (BND)"| DOG_END["__end__: [idx_dog]"]

    CAT_END -->|"reset"| ROOT
    CAR_END -->|"reset"| ROOT
    DOG_END -->|"reset"| ROOT

    style ROOT fill:#4a90d9,color:#fff
    style CAT_END fill:#27ae60,color:#fff
    style CAR_END fill:#27ae60,color:#fff
    style DOG_END fill:#27ae60,color:#fff
```

> **`__end__`** nodes store a list of word indices — multiple indices indicate homophones
> (e.g., "aunt" and "ant" sharing the same phoneme sequence).
> After a word boundary token, the traversal resets to ROOT to begin the next word.

---

## 3. VectorizedLexiconConstraint: Compiled DFA States

The trie is BFS-enumerated into integer states and compiled into a dense `transition_table[state, token] → next_state`.

```mermaid
stateDiagram-v2
    [*] --> ROOT

    ROOT --> C : token=c(1)
    ROOT --> D : token=d(5)

    C --> CA : token=a(2)

    CA --> CAT : token=t(3)
    CA --> CAR : token=r(4)

    CAT --> BND_CAT : token=BND(8)
    CAR --> BND_CAR : token=BND(8)

    BND_CAT --> ROOT : auto-reset (end_state_mask=True)
    BND_CAR --> ROOT : auto-reset (end_state_mask=True)

    D --> DO : token=o(6)
    DO --> DOG : token=g(7)
    DOG --> BND_DOG : token=BND(8)
    BND_DOG --> ROOT : auto-reset (end_state_mask=True)

    ROOT --> SINK : unknown token
    C --> SINK : unknown token
    CA --> SINK : unknown token
    SINK --> SINK : any token

    note right of SINK : Sink state = num_nodes.\nAll transitions from sink stay in sink.
    note right of ROOT : Extra BND(8) from ROOT\nallowed (consecutive silences).
```

---

## 4. Beam Search Runtime Data Flow (VectorizedLexiconConstraint)

```mermaid
flowchart TD
    A["initialize_state(B, K)\n→ state [B×K] = 0 (ROOT)"]
    A --> LOOP

    subgraph LOOP ["Per decoder timestep"]
        direction TB
        B["get_constraint_mask_with_state(\n  state, vocab_size, last_labels\n)"]
        B --> C["mask [B, K, V]\n• blank always True\n• valid transitions True\n• last emitted token True"]
        C --> D["CTC decoder selects token\nemitted_labels [B, K]"]
        D --> E["update_state(\n  parent_state, emitted_labels,\n  prev_last_labels\n)"]
        E --> F{"advance?"}
        F -->|"non-blank, non-repeat"| G["table[state, token] → next_state"]
        F -->|"blank or repeat"| H["keep current state"]
        G --> I{"end_state_mask?"}
        I -->|"True (word boundary)"| J["reset → ROOT\nword complete"]
        I -->|"False"| K["stay at next_state"]
        J --> NEW_STATE["updated state [B×K]"]
        K --> NEW_STATE
        H --> NEW_STATE
    end

    NEW_STATE -->|"next timestep"| LOOP
    J --> LM["get_words_at_state(state)\n→ word indices for LM rescoring"]
```

---

## 5. `get_valid_next_tokens_with_word_info` Logic

```mermaid
flowchart TD
    START(["sequence: [t₁, t₂, …, tₙ]"])
    START --> CACHE{"In cache?"}
    CACHE -->|"hit"| RETURN_CACHE["return cached result"]
    CACHE -->|"miss"| TRAVERSE

    subgraph TRAVERSE ["Traverse trie"]
        direction TB
        T1["node = trie root"]
        T1 --> LOOP2["for each token in sequence"]
        LOOP2 --> T2{"token in node?"}
        T2 -->|"No"| EMPTY["return (∅, False, [])"]
        T2 -->|"Yes"| T3["node = node[token]"]
        T3 --> T4{"token == BND?"}
        T4 -->|"Yes"| T5{"'__end__' in node?"}
        T5 -->|"No"| EMPTY
        T5 -->|"Yes"| T6["word_indices = node['__end__']\nat_word_boundary = True\nnode = ROOT"]
        T4 -->|"No"| T7["at_word_boundary = False\nword_indices = []"]
        T6 --> LOOP2
        T7 --> LOOP2
    end

    TRAVERSE --> COLLECT["valid_tokens = keys(node) − '__end__'"]
    COLLECT --> BND_EXTRA{"allow_boundary?"}
    BND_EXTRA -->|"Yes (just reset to ROOT)"| ADD_BND["also add BND to valid_tokens\n(allows consecutive silences)"]
    BND_EXTRA -->|"No"| RESULT
    ADD_BND --> RESULT
    RESULT["return (valid_tokens, at_word_boundary, word_indices)"]
    RESULT --> CACHE_STORE["store in cache"]
```
