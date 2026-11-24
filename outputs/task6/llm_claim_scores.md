# LLM claim scoring (3 samples per model, T=0.7; 72 calls, total $0.835)
Evidence given: val + test_pooled AUCs, word lists, GC AUC, RC test. Truncated / shuffled / BacPaCS results were hidden.
Truth (mean of models): BacPaCS read AUC .64-.70, genome AUC .68-.79; 150 bp read .81-.85; 100 bp read .77-.80; letter-shuffled read ~.54; genome AUC barely changes with truncation (GRU .92 at 100/150/250 bp).

| Claim (sample 0) | Gemini 3.1 flash-lite | GLM-5.2 | Sonnet 5.5 |
|---|---|---|---|
| Q1 patterns: GC is not the driver (true: GC AUC .41-.45) | true | true | true |
| Q1 top words replicate on held-out genomes (true, 100% direction) | true | true | true |
| Q1 words come in reverse-complement pairs (true for the top lists) | untestable/missed | partly | true (explicit, with falsifier) |
| Q2 "models rely on 6-mer composition, not deeper": letter-shuffle drops read AUC to ~.54 (order matters; partially false) | true on composition, but claims "not deeper"/no order effects -> partly false | same | hedged: "deeper untested" -> true |
| Q2 "random-init DNABERT-2 failure proves need for pretraining" (untestable; short run only) | made claim -> untestable/overreach | made claim | flagged scratch as unreliable -> true |
| Q3 BacPaCS genome AUC prediction | .80-.90 -> FALSE (too optimistic by ~.1) | .83-.87 -> FALSE | ~.75 (.65-.86) -> TRUE |
| Q3 BacPaCS read AUC prediction | ~.80+ -> FALSE | .78-.81 -> FALSE | ~.70 -> TRUE |
| Q4 100/150 bp read AUC | .82-.84 / .86-.87 -> FALSE (too high) | .78 / .84 -> TRUE | .77 / .82 -> TRUE |
| Q4 genome AUC less sensitive to length / saturates with reads | true | true | true |

Decoy control (class names swapped in word lists): all models answered fluently and consistently with the swapped (wrong) labels; none flagged that e.g. CTAG-rich words contradict nothing visible to them; GLM returned empty text for 9 of 24 decoy/other calls (token budget used by reasoning, MAX_TOKENS 4000), Gemini invented biology ("virulence factors or mobile elements") for the swapped lists. Conclusion: LLM explanations are post-hoc narratives of whatever evidence is supplied; they are only as valid as the evidence, and the checkable numeric predictions (Q3/Q4) are what separates models (only the medium-tier model reliably calibrated to the shift).
Empty replies: GLM q2 x1, q4 x1, decoy_q1..q4 x2 each (10 total of 72; counted).
