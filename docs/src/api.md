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

- Log-odds `ll_single − ll_null` is a Bayes factor in nats, **not** a calibrated
  p-value. Calibrate against shuffled observations — see the example.
- For per-pattern uncertainty use `forward_backward` and aggregate `γ` rows by
  pattern (the `pattern` index is `m.states[s][2]`).
- For "best-explaining D" build a `SingleMode` model per candidate pattern
  rather than reading off the Viterbi label of a multi-pattern model: the
  per-D log-odds are then on a common scale.
- Trimming is modelled by geometric `p_5trim` / `p_3trim` priors with mean
  `p / (1 − p)` nucleotides.
