# DHMMs.jl

[![Build Status](https://github.com/mashu/DHMMs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/DHMMs.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mashu/DHMMs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/DHMMs.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mashu.github.io/DHMMs.jl/dev)

Segment detection HMMs using [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mashu/DHMMs.jl")
```

## Models

| Mode | Topology | Description |
|------|----------|-------------|
| `NullMode` | `N ⟲` | Background only |
| `SingleMode` | `N₀ ⟲ → Pᵢ → N₁ ⟲` | At most one pattern segment |
| `LoopMode` | `N ⟲ ⇄ Pᵢ ⟲` | Multiple pattern segments allowed |

## Usage

```julia
using DHMMs

# Define patterns as integer sequences (e.g., DNA: A=1, C=2, G=3, T=4)
patterns = [[1,2,3], [3,3,1,2]]

# Build models
m_null = SegmentHMM(NullMode(), patterns)
m_single = SegmentHMM(SingleMode(), patterns)
m_loop = SegmentHMM(LoopMode(), patterns)

# Score and decode
obs = [4, 4, 1, 2, 3, 4, 4]
ll = logdensityof(m_loop, obs)
segments = decode(m_loop, obs)
```

## License

MIT
