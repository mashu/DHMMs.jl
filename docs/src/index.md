# DHMMs.jl

Segment detection Hidden Markov Models.

## Models

| Mode | Topology | Description |
|------|----------|-------------|
| [`NullMode`](@ref) | `N ⟲` | Background only |
| [`SingleMode`](@ref) | `N₀ ⟲ → Pᵢ → N₁ ⟲` | At most one pattern |
| [`LoopMode`](@ref) | `N ⟲ ⇄ Pᵢ ⟲` | Multiple patterns |

## Quick Start

```julia
using DHMMs

patterns = [[1,2,3], [3,3,1,2]]
m = SegmentHMM(LoopMode(), patterns)

obs = [4, 4, 1, 2, 3, 4, 4]
ll = logdensityof(m, obs)
segments = decode(m, obs)
```

