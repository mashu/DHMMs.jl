# DHMMs.jl

Segment-detection Hidden Markov Models, parameterised by a topology and a list
of integer patterns. Built on
[HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl).

## Topologies

| Mode | Topology | Use |
|------|----------|-----|
| [`NullMode`](@ref)   | `N ⟲`           | Background-only null for Bayes-factor comparisons |
| [`SingleMode`](@ref) | `N₀ ⟲ → Pᵢ → N₁ ⟲` | At most one (possibly trimmed) pattern instance |
| [`LoopMode`](@ref)   | `N ⟲ ⇄ Pᵢ`     | Zero, one, or many (possibly trimmed) instances |

## Statistical model

Each pattern position emits its symbol with probability `match_prob` and any
other symbol uniformly. Entry into a pattern from background follows a
truncated-geometric **5'-trim** distribution over positions
(`p_5trim`); exit from each internal position is a Bernoulli **3'-trim**
(`p_3trim`). Background `N` is a uniform-emission self-loop with mass
`p_stay_n`.

Mean trim length under the geometric prior is `p / (1 − p)` nucleotides:
`p = 0.75` ↔ mean 3 nt, `p = 0.8` ↔ mean 4 nt.

## Quick start

```julia
using DHMMs

patterns = [[1, 2, 3], [3, 3, 1, 2]]
m_null   = SegmentHMM(NullMode())
m_loop   = SegmentHMM(LoopMode(), patterns; p_5trim=0.7, p_3trim=0.7)

obs = [4, 4, 1, 2, 3, 4, 4]
ll          = logdensityof(m_loop, obs)
log_odds    = logdensityof(m_loop, obs) - logdensityof(m_null, obs)
γ, _        = forward_backward(m_loop, obs)
segments    = decode(m_loop, obs)
```

See the [Example](example.md) page for the full D-gene workflow including
trim-aware modelling, posterior decoding, per-pattern likelihoods, and
shuffle-based significance calibration.
