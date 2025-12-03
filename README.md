## DHMMs: Full-loop HMM for IGHD segments (with ≥2 Ds allowed)

### What this model captures (high level)
- The sequence alternates between a background N state (uniform emissions) and D-gene states that emit a specific IGHD gene sequence.
- Entry into a D segment can start at any internal position (left trimming).
- Within a D segment you typically continue forward; you can exit back to N at any time (right trimming).
- Rarely, you may jump from the end of a D directly into another D without passing through N (small probability).

### Full loop model (allows 0, 1, 2, … Ds)

```
                 +-------------------+
                 |        N          |  (uniform emissions)
                 +-------------------+
                      ^        |
                      | p_stay |      enter any D at any internal position
                      |        |  (left trimming; total 1 - p_stay distributed over positions)
                      |        v
                +---------------------+   p_continue_d      +---------------------+
   ... <--------| D : gene g, pos 1   | ------------------> | D : gene g, pos 2   | --> ... --> | D : gene g, pos L |
                +---------------------+                     +---------------------+              +--------------------+
                      |     ^                                        |     ^                               |     ^
                      |     |                                        |     |                               |     |
                      |     +---- exit to N (1 - p_continue_d) ------+     +---- exit to N (1 - p_continue_d)      |
                      |                                                                                              |
                      +------------------------------ exit to N (1 - p_direct_d_end) -------------------------------+

          optional direct jump to any D position (rare): p_direct_d_end (spread over all D positions)

Legend:
- N: background insertion-like region (uniform emissions)
- D: emits the IGHD gene’s reference base at that position (match_prob)
- p_stay = p_stay_n
- p_continue_d: keep advancing within D
- 1 - p_continue_d: exit to N from inside D (right trimming)
- p_direct_d_end: from the last base of a D, jump directly into any D position (no N in between)
```

Typical default parameters (tunable):
- p_stay_n = 0.75
- p_continue_d = 0.95
- p_direct_d_end = 0.01
- match_prob = 0.85

Notes:
- Left trimming: N→D entry mass is spread uniformly across all positions of all D genes.
- Right trimming: you can exit to N from any D position.
- Direct D→D jumps are kept small; N→D→N→D is the common path for multiple Ds.

### Constrained alternative (≤1 D)

```
         +--------+            +---------------------+           +---------+
         |  N0    |  enter D   | D : gene g, pos 1…L |  exit ->  |   N1    |
         +--------+ ----------> +---------------------+ --------> +---------+
            ^   |                 p_continue_d / exit to N1          ^
            |   | p_stay_n                                         (loop)
            +---+

Where:
- N0 allows entry to any internal D position (left trimming)
- Exiting D goes to N1, which cannot re-enter D (forbids a second D)
```

### What we compare (Question 2: “Is there a 2nd D?”)
- We compute sequence log-likelihood under two models on the same read:
  - Full loop model (≥2 Ds allowed)
  - Constrained ≤1 D model (re-entry forbidden)
- Decision statistic: Δ = log P(read | full) − log P(read | ≤1 D)
  - Large positive Δ ⇒ evidence for at least two D segments
  - Thresholds can be calibrated (e.g., Δ > 3 strong evidence)

### Why this design
- A single, recurrent N↔D topology handles any number of D segments without baking in “two slots.”
- Left/right trimming and a small direct D→D jump capture realistic junction behavior.
- Comparing against the ≤1 D constrained variant isolates evidence for multiple D segments while keeping the generative assumptions consistent.
