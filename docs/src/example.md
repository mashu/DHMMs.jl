# Example: Detecting D Segments in CDR3

This example shows how to detect IGHD gene segments in expressed CDR3 sequences
using DHMMs. Expressed Ds are trimmed on both sides and embedded in N-additions,
so the model must allow partial matches and multiple instances.

## Setup

```julia
using DHMMs
using BioSequences
using FASTX
using Random
```

## DNA encoding

DHMMs works on integer-encoded sequences. The package intentionally provides no
DNA helpers — pick any encoding consistent between patterns and observations:

```julia
function dna_to_int(nt::DNA)
    nt == DNA_A && return 1
    nt == DNA_C && return 2
    nt == DNA_G && return 3
    nt == DNA_T && return 4
    error("Unknown nucleotide: $nt")
end

seq_to_ints(seq::LongDNA{4}) = [dna_to_int(nt) for nt in seq]
```

## Loading D genes

```julia
function load_d_genes(fasta_path::String)
    genes = Vector{Int}[]
    names = String[]
    open(FASTAReader, fasta_path) do reader
        for record in reader
            push!(names, description(record))
            push!(genes, seq_to_ints(sequence(LongDNA{4}, record)))
        end
    end
    names, genes
end

# Or define manually for the example:
d_genes = [
    dna"GGTATAGTGGGAGCTAC",                  # IGHD1-1
    dna"AGGATATTGTAGTAGTAC",                 # IGHD2-2
    dna"GTATTACTATGGTTCGGGGAGTTATTATAAC",    # IGHD3-3
]
patterns = [seq_to_ints(g) for g in d_genes]
```

For repertoire work you also want the reverse complements, because expressed Ds
appear on either strand:

```julia
rc_patterns = [seq_to_ints(reverse_complement(g)) for g in d_genes]
all_patterns = vcat(patterns, rc_patterns)
```

## Building models with trim priors

Expressed Ds are trimmed by exonucleases on both ends. Trim length is
approximately geometric with mean 2–5 nt. Set `p_5trim` and `p_3trim` so the
mean trim under the model matches your prior: `E[trim] = p / (1 - p)`, so
`p = 0.75` ↔ mean 3 nt, `p = 0.8` ↔ mean 4 nt.

```julia
# Background-only null
m_null = SegmentHMM(NullMode())

# Single-D model: at most one D, both ends trimmable
m_single = SegmentHMM(SingleMode(), all_patterns;
    p_stay_n=0.6, p_skip=0.05,
    p_5trim=0.75, p_3trim=0.75,
    match_prob=0.9)

# Multi-D model: zero, one, or more Ds, separated by N-additions
m_loop = SegmentHMM(LoopMode(), all_patterns;
    p_stay_n=0.85,
    p_5trim=0.75, p_3trim=0.75,
    match_prob=0.9)
```

`match_prob` should reflect substitution rate, not trim rate — keep it high
(0.9–0.95) and let `p_5trim`/`p_3trim` absorb the trimming.

## Scoring a CDR3

```julia
cdr3 = dna"AATTATTGTGGTGGTGATTGCTATGCGAATGTATAGCAGTGGCTGATGC"
obs = seq_to_ints(cdr3)

ll_null   = logdensityof(m_null, obs)
ll_single = logdensityof(m_single, obs)
ll_loop   = logdensityof(m_loop, obs)

log_odds_has_d   = ll_single - ll_null
log_odds_multi_d = ll_loop   - ll_single
```

These are Bayes factors in nats. They are **not calibrated p-values** — see
"Calibrating significance" below before using a fixed threshold.

## Best-explaining D via per-pattern likelihoods

The cleanest way to ask *which* D best explains the read is to score each D in
isolation under `SingleMode` and compare. No helper needed:

```julia
per_d_ll = map(eachindex(all_patterns)) do i
    m_i = SegmentHMM(SingleMode(), [all_patterns[i]];
                     p_5trim=0.75, p_3trim=0.75, match_prob=0.9)
    logdensityof(m_i, obs) - ll_null
end
best_d = argmax(per_d_ll)
```

`per_d_ll` are log-odds-vs-null for each candidate D and are directly
comparable to one another (same null, same trim prior).

## Posterior decoding (per-position D probability)

Viterbi gives one path; the forward–backward posterior gives `P(state = s | obs)`
at every position. Aggregate state rows by pattern to get a per-D occupancy
track:

```julia
γ, _ = forward_backward(m_loop, obs)

n_pat = length(all_patterns)
p_per_d = zeros(n_pat, length(obs))
for (s, info) in enumerate(m_loop.states)
    info[1] === :P || continue
    p_per_d[info[2], :] .+= γ[s, :]
end
# p_per_d[i, t] = P(position t is inside D_i)
```

This is what you should use to call "D present" with a confidence rather than a
hard Viterbi label.

## Viterbi decoding into segments

```julia
segments = decode(m_loop, obs)
for seg in segments
    region = String(cdr3[seg.start:seg.stop])
    if seg.type === :P
        println("D$(seg.pattern): $region ($(seg.start)-$(seg.stop))")
    else
        println("N: $region ($(seg.start)-$(seg.stop))")
    end
end
```

Two adjacent occurrences of the same D are returned as two `Segment`s.

## Calibrating significance via shuffle

The log-odds above has no closed-form null. The standard fix is to compare to
the same statistic on shuffled versions of the read:

```julia
function null_quantile(obs, m_single, m_null; B=200, rng=Random.default_rng())
    obs_score = logdensityof(m_single, obs) - logdensityof(m_null, obs)
    null_scores = map(1:B) do _
        sh = shuffle(rng, obs)
        logdensityof(m_single, sh) - logdensityof(m_null, sh)
    end
    p = (1 + count(>=(obs_score), null_scores)) / (B + 1)
    obs_score, p
end

score, p = null_quantile(obs, m_single, m_null)
```

For more realistic null preservation of dinucleotide composition, shuffle
dinucleotides rather than single nucleotides; the package leaves this choice to
you.

## Negative control

```julia
random_cdr3 = dna"AAAAAACCCCCCGGGGGGTTTTTT"
obs_neg = seq_to_ints(random_cdr3)

@show logdensityof(m_single, obs_neg) - logdensityof(m_null, obs_neg)
@show filter(s -> s.type === :P, decode(m_loop, obs_neg))
```

A run-homopolymer like this should give a small or negative log-odds and few
or no pattern segments under the trim prior.
