Ma# Day-Specific Transformer — Validation WER & PER
_Generated: 2026-04-15 13:23_
_Partition: val | Weights: modelWeights_PER_24 | Chunk: 1s / 7.5s context_

## Day-Specific Linear + Softsign Transformer (WER from Lightbeam)

| Seed | WER |
|------|-----|
| 0 | 0.140284 |
| 1 | 0.147198 |
| 2 | 0.140830 |
| 3 | 0.135189 |
| 4 | 0.136827 |
| 5 | 0.138464 |
| 6 | 0.136463 |
| 7 | 0.137373 |
| 8 | 0.143923 |
| 9 | 0.133006 |
| **mean** | **0.138966** |

## Day-Specific Linear Transformer (WER from Lightbeam)

| Seed | WER |
|------|-----|
| 0 | 0.126274 |
| 1 | 0.134643 |
| 2 | 0.134461 |
| 3 | 0.134279 |
| 4 | 0.133188 |
| 5 | 0.125546 |
| 6 | 0.132096 |
| 7 | 0.135007 |
| 8 | 0.130640 |
| 9 | 0.122453 |
| **mean** | **0.130859** |

## Unidirectional Transformer (no day-specific, no softsign)
_Weights: modelWeights_PER_24 | Chunk: 1s / 5–20s context (variable) | WER from Lightbeam_

| Seed | WER |
|------|-----|
| 0 | 0.114629 |
| 1 | 0.116266 |
| 2 | 0.110444 |
| 3 | 0.118996 |
| 4 | 0.113719 |
| 5 | 0.119360 |
| 6 | 0.125546 |
| 7 | 0.118814 |
| 8 | 0.123726 |
| 9 | 0.116448 |
| **mean** | **0.117795** |

---

## Appendix — PER Results

## Day-Specific Linear + Softsign Transformer

| Seed | PER |
|------|-----|
| 0 | 0.175189 |
| 1 | 0.174363 |
| 2 | 0.172465 |
| 3 | 0.174570 |
| 4 | 0.173579 |
| 5 | 0.173703 |
| 6 | 0.170113 |
| 7 | 0.173332 |
| 8 | 0.175519 |
| 9 | 0.169865 |
| **mean** | **0.173270** |

## Day-Specific Linear Transformer

| Seed | PER |
|------|-----|
| 0 | 0.169659 |
| 1 | 0.168751 |
| 2 | 0.168379 |
| 3 | 0.168586 |
| 4 | 0.167884 |
| 5 | 0.164252 |
| 6 | 0.161859 |
| 7 | 0.166068 |
| 8 | 0.166770 |
| 9 | 0.162849 |
| **mean** | **0.166506** |

---

## GRU Shared Linear Layer
_Generated: 2026-04-21 | Partition: val | Llama 3.2-1B_

| Seed | WER |
|------|-----|
| 0 | 0.186317 |
| 1 | 0.185044 |
| 2 | 0.180131 |
| 3 | 0.179585 |
| 4 | 0.186681 |
| 5 | 0.184680 |
| 6 | 0.186317 |
| 7 | 0.187773 |
| 8 | 0.190684 |
| 9 | 0.187409 |
| **mean** | **0.185462** |

---

## Original GRU
_Generated: 2026-04-21 | Partition: val | Llama 3.2-1B_

| Seed | WER |
|------|-----|
| 0 | 0.140102 |
| 1 | 0.141921 |
| 2 | 0.140102 |
| 3 | 0.137009 |
| 4 | 0.139556 |
| 5 | 0.142831 |
| 6 | 0.148836 |
| 7 | 0.134098 |
| 8 | 0.143195 |
| 9 | 0.139010 |
| **mean** | **0.140666** |

---

