# API

## Model topologies

```@docs
NullMode
SingleMode
LoopMode
```

## Construction

```@docs
SegmentHMM
```

## Inference

`logdensityof`, `viterbi`, and `forward_backward` are re-exported from
[HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl) and
dispatched on `SegmentHMM`.

```@docs
logdensityof(::SegmentHMM, ::Any)
viterbi(::SegmentHMM, ::Any)
forward_backward(::SegmentHMM, ::Any)
decode
```

## Output

```@docs
Segment
```

## Notes on statistics

- Each pattern is a profile HMM with match (`M`), insert (`I`), and silent
  (marginalised) delete states; `Segment` collapses `M` and `I` into a single
  `:P` segment per pattern instance.
- Log-odds `ll_single − ll_null` is a Bayes factor in nats, **not** a calibrated
  p-value. Calibrate against shuffled observations — see the example.
- For per-pattern uncertainty use `forward_backward` and aggregate `γ` rows by
  pattern (sum `γ[s, :]` over all states `s` with `m.states[s][2] == i`).
- For "best-explaining D" build a `SingleMode` model per candidate pattern
  rather than reading off the Viterbi label of a multi-pattern model: per-D
  log-odds are then on a common scale.
- Indel priors `p_mi`, `p_md`, `p_ii`, `p_dd` follow the standard profile-HMM
  parameterisation. Set `p_mi = p_md = 0` to recover a match-only model.
- Trimming is modelled by geometric `p_5trim` / `p_3trim` priors with mean
  `p / (1 − p)` nucleotides.
