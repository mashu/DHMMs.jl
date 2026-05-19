# DHMMs.jl

Segment-detection **profile** Hidden Markov Models, parameterised by a topology
and a list of integer patterns. Each pattern is realised as a profile-HMM
segment with match (M), insert (I), and (silent, marginalised-out) delete
states, so divergent and indel-containing instances of a pattern can still be
detected and scored. Built on
[HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl).

## Topologies

| Mode | Topology | Use |
|------|----------|-----|
| [`NullMode`](@ref)   | `N ⟲`                          | Background-only null for Bayes-factor comparisons |
| [`SingleMode`](@ref) | `N₀ ⟲ → profileᵢ → N₁ ⟲`      | At most one (possibly trimmed, indel-bearing) pattern instance |
| [`LoopMode`](@ref)   | `N ⟲ ⇄ profileᵢ`              | Zero, one, or many instances; separated by background, or chained directly under `p_direct > 0` |

## Performance

Transition matrices are stored as `SparseMatrixCSC` and Viterbi /
forward / forward-backward dispatch to HiddenMarkovModels.jl's sparse
`argmaxplus_transmul!`, iterating only over the `O(L)` non-zero out-edges
per state instead of all `O(n_states)` columns.

## Statistical model

Each profile block is a standard plan-7-style profile HMM:

- **Match states** `M_{i,j}` emit the pattern symbol at position `j` with
  probability `match_prob`, other symbols uniformly.
- **Insert states** `I_{i,j}` sit between `M_{i,j}` and `M_{i,j+1}`, emit a
  uniform background distribution, and self-loop with probability `p_ii`.
- **Delete states** are silent: a chain `D_{j+1}…D_{j+k-1}` is marginalised
  into a direct `M_j → M_{j+k}` transition with weight
  `p_md · p_dd^(k-2) · (1 − p_dd)`. Delete chains that would run past the last
  position become additional 3'-trim mass at `M_j → N`.
- **5'-trim** is a truncated-geometric prior over entry positions of each
  profile (`p_5trim`); **3'-trim** is per-match-position exit probability
  (`p_3trim`). Untruncated geometric mean trim length is `p / (1 − p)`.

The background `N` is a uniform-emission self-loop with mass `p_stay_n`.

## Quick start

```julia
using DHMMs

patterns = [[1, 2, 3], [3, 3, 1, 2]]
m_null   = SegmentHMM(NullMode())
m_loop   = SegmentHMM(LoopMode(), patterns;
                       p_5trim=0.7, p_3trim=0.7,
                       p_mi=0.025, p_md=0.025)

obs       = [4, 4, 1, 2, 3, 4, 4]
ll        = logdensityof(m_loop, obs)
log_odds  = logdensityof(m_loop, obs) - logdensityof(m_null, obs)
γ, _      = forward_backward(m_loop, obs)
segments  = decode(m_loop, obs)
```

See the [Example](example.md) for the full D-gene workflow including
trim-aware modelling, posterior decoding aggregated to per-pattern occupancy,
per-D best-explanation scoring, and dinucleotide-shuffle calibration.
