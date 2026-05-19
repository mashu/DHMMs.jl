"""
    DHMMs

Segment detection Hidden Markov Models built on HiddenMarkovModels.jl.
"""
module DHMMs

using Distributions: Categorical
import HiddenMarkovModels
using HiddenMarkovModels: HMM, viterbi, logdensityof, forward_backward

export SegmentHMM, NullMode, SingleMode, LoopMode
export decode, Segment, logdensityof, viterbi, forward_backward

# ============================================================================
# Mode types
# ============================================================================

"""
    ModelMode

Abstract type for model topologies.
"""
abstract type ModelMode end

"""
    NullMode <: ModelMode

Background-only model. Topology: `N ⟲`
"""
struct NullMode <: ModelMode end

"""
    SingleMode <: ModelMode

Single pattern model. Topology: `N₀ ⟲ → Pᵢ → N₁ ⟲`

Allows at most one pattern segment with optional skip. 5'/3' trimming of the
pattern is modelled by geometric distributions over entry and exit positions
(see `p_5trim`, `p_3trim`).
"""
struct SingleMode <: ModelMode end

"""
    LoopMode <: ModelMode

Loop model allowing multiple patterns. Topology: `N ⟲ ⇄ Pᵢ`

Multiple pattern instances are separated by at least one background emission.
Trimming is modelled by `p_5trim` and `p_3trim`.
"""
struct LoopMode <: ModelMode end

const StateInfo = Tuple{Symbol, Int, Int}

# ============================================================================
# Main struct
# ============================================================================

"""
    SegmentHMM{M<:ModelMode, H<:HMM}

Segment detection HMM parameterised by mode `M`.

# Fields
- `hmm::H`: underlying HiddenMarkovModels.HMM
- `states::Vector{StateInfo}`: state metadata `(type, pattern_idx, position)`
- `pattern_lengths::Vector{Int}`: length of each pattern
"""
struct SegmentHMM{M<:ModelMode, H<:HMM}
    hmm::H
    states::Vector{StateInfo}
    pattern_lengths::Vector{Int}
end

# ============================================================================
# Emissions
# ============================================================================

function match_emission(symbol::Int, n_symbols::Int, match_prob::Float64)
    p = fill((1.0 - match_prob) / (n_symbols - 1), n_symbols)
    p[symbol] = match_prob
    Categorical(p)
end

uniform_emission(n_symbols::Int) = Categorical(fill(1.0 / n_symbols, n_symbols))

# ============================================================================
# Trim prior
# ============================================================================

"""
    trim_pmf(len, p_trim) -> Vector{Float64}

Truncated geometric PMF over positions `1:len` representing 5'-trim:
`P(enter at position j) ∝ p_trim^(j-1)`, normalised.

- `p_trim = 0` puts all mass on position 1 (no trim).
- `p_trim → 1` approaches uniform.
- Untruncated mean trim length is `p_trim / (1 - p_trim)`.

Used internally to weight `N → P_{i,j}` entry transitions; the same shape is
applied (reversed) to `P_{i,j} → N` exits via `p_3trim`.
"""
function trim_pmf(len::Int, p_trim::Float64)
    0.0 <= p_trim < 1.0 || throw(ArgumentError("p_trim must be in [0, 1)"))
    len >= 1 || throw(ArgumentError("len must be >= 1"))
    w = [p_trim^(j - 1) for j in 1:len]
    w ./ sum(w)
end

# ============================================================================
# Constructors
# ============================================================================

"""
    SegmentHMM(mode::ModelMode, patterns::Vector{Vector{Int}}; kwargs...)

Construct a segment HMM.

# Arguments
- `mode`: model topology ([`NullMode`](@ref), [`SingleMode`](@ref), [`LoopMode`](@ref))
- `patterns`: vector of integer sequences representing patterns

# Keyword Arguments
- `n_symbols=4`: alphabet size
- `match_prob=0.85`: per-position emission probability for the matching symbol
- `p_stay_n=0.6` (Single) / `0.75` (Loop): self-loop probability for the background state
- `p_skip=0.05` (Single only): probability of skipping patterns entirely
- `p_5trim=0.0`: geometric per-step probability of 5'-trimming the pattern.
   `0` means entry is always at position 1; values close to 1 approach uniform entry.
- `p_3trim=0.1` (Single) / `0.05` (Loop): geometric per-step probability of 3'-trim
   (replaces the old `p_continue`: `p_continue == 1 - p_3trim`).
"""
function SegmentHMM(::NullMode, patterns::Vector{Vector{Int}}=Vector{Int}[];
                    n_symbols::Int=4, kwargs...)
    dists = [uniform_emission(n_symbols)]
    hmm = HMM([1.0], fill(1.0, 1, 1), dists)
    SegmentHMM{NullMode, typeof(hmm)}(hmm, [(:N, 0, 0)], Int[])
end

function SegmentHMM(::SingleMode, patterns::Vector{Vector{Int}};
                    n_symbols::Int=4,
                    p_stay_n::Float64=0.6,
                    p_skip::Float64=0.05,
                    p_5trim::Float64=0.0,
                    p_3trim::Float64=0.1,
                    match_prob::Float64=0.85)
    isempty(patterns) && throw(ArgumentError("patterns must be non-empty for SingleMode"))
    pat_mass = 1.0 - p_stay_n - p_skip
    pat_mass > 0 || throw(ArgumentError("p_stay_n + p_skip must be < 1"))

    lens = length.(patterns)
    n_pat = length(patterns)
    total_p = sum(lens)
    n_states = 2 + total_p

    starts = cumsum([0; lens[1:end-1]]) .+ 2
    n_end = n_states

    trans = zeros(n_states, n_states)
    trans[1, 1] = p_stay_n
    trans[1, n_end] = p_skip

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + j - 1] = pat_mass * pmf[j] / n_pat
        end
    end

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        L = length(pat)
        for j in 1:L
            cur = s + j - 1
            if j < L
                trans[cur, cur + 1] = 1.0 - p_3trim
                trans[cur, n_end] = p_3trim
            else
                trans[cur, n_end] = 1.0
            end
        end
    end
    trans[n_end, n_end] = 1.0

    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    dists[1] = uniform_emission(n_symbols)
    dists[n_end] = uniform_emission(n_symbols)
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for (j, sym) in enumerate(pat)
            dists[s + j - 1] = match_emission(sym, n_symbols, match_prob)
        end
    end

    states = Vector{StateInfo}(undef, n_states)
    states[1] = (:N, 0, 0)
    states[n_end] = (:N, 0, 0)
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for j in 1:length(pat)
            states[s + j - 1] = (:P, i, j)
        end
    end

    init = zeros(n_states)
    init[1] = 1.0

    hmm = HMM(init, trans, dists)
    SegmentHMM{SingleMode, typeof(hmm)}(hmm, states, lens)
end

function SegmentHMM(::LoopMode, patterns::Vector{Vector{Int}};
                    n_symbols::Int=4,
                    p_stay_n::Float64=0.75,
                    p_5trim::Float64=0.0,
                    p_3trim::Float64=0.05,
                    match_prob::Float64=0.85)
    isempty(patterns) && throw(ArgumentError("patterns must be non-empty for LoopMode"))
    pat_mass = 1.0 - p_stay_n
    pat_mass > 0 || throw(ArgumentError("p_stay_n must be < 1"))

    lens = length.(patterns)
    n_pat = length(patterns)
    total_p = sum(lens)
    n_states = 1 + total_p

    starts = cumsum([0; lens[1:end-1]]) .+ 2

    trans = zeros(n_states, n_states)
    trans[1, 1] = p_stay_n

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + j - 1] = pat_mass * pmf[j] / n_pat
        end
    end

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        L = length(pat)
        for j in 1:L
            cur = s + j - 1
            if j < L
                trans[cur, cur + 1] = 1.0 - p_3trim
                trans[cur, 1] = p_3trim
            else
                trans[cur, 1] = 1.0
            end
        end
    end

    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    dists[1] = uniform_emission(n_symbols)
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for (j, sym) in enumerate(pat)
            dists[s + j - 1] = match_emission(sym, n_symbols, match_prob)
        end
    end

    states = Vector{StateInfo}(undef, n_states)
    states[1] = (:N, 0, 0)
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for j in 1:length(pat)
            states[s + j - 1] = (:P, i, j)
        end
    end

    init = zeros(n_states)
    init[1] = 1.0

    hmm = HMM(init, trans, dists)
    SegmentHMM{LoopMode, typeof(hmm)}(hmm, states, lens)
end

# ============================================================================
# Forward to HiddenMarkovModels
# ============================================================================

"""
    logdensityof(m::SegmentHMM, obs)

Marginal log-likelihood of `obs` under `m` (forward algorithm).
"""
HiddenMarkovModels.logdensityof(m::SegmentHMM, obs) = logdensityof(m.hmm, obs)

"""
    viterbi(m::SegmentHMM, obs) -> (path, loglik)

Most-likely state path and its log-likelihood.
"""
HiddenMarkovModels.viterbi(m::SegmentHMM, obs) = viterbi(m.hmm, obs)

"""
    forward_backward(m::SegmentHMM, obs) -> (γ, logL)

Posterior state marginals. `γ[s, t]` is `P(state_t = s | obs)`.
Aggregate rows by pattern (using `m.states`) for per-pattern occupancy at each
position.
"""
HiddenMarkovModels.forward_backward(m::SegmentHMM, obs) = forward_backward(m.hmm, obs)

# ============================================================================
# Segment
# ============================================================================

"""
    Segment

Decoded segment from Viterbi path.

# Fields
- `type::Symbol`: `:N` (background) or `:P` (pattern)
- `pattern::Int`: pattern index (`0` for background)
- `start::Int`: start position in observation (inclusive)
- `stop::Int`: end position in observation (inclusive)
"""
struct Segment
    type::Symbol
    pattern::Int
    start::Int
    stop::Int
end

"""
    decode(m::SegmentHMM, obs) -> Vector{Segment}

Decode `obs` into segments using Viterbi. Adjacent occurrences of the same
pattern are kept as separate segments by detecting a reset in pattern position.
"""
function decode(m::SegmentHMM, obs::AbstractVector{<:Integer})
    isempty(obs) && return Segment[]
    path, _ = viterbi(m.hmm, obs)

    segments = Segment[]
    typ, pat, pos = m.states[path[1]]
    seg_start = 1

    for i in 2:length(path)
        t, p, q = m.states[path[i]]
        # New segment if (type, pattern) changes, or if within a pattern the
        # position does not advance by 1 (i.e. re-entry into the same pattern).
        new_seg = (t, p) != (typ, pat) || (t === :P && q != pos + 1)
        if new_seg
            push!(segments, Segment(typ, pat, seg_start, i - 1))
            typ, pat = t, p
            seg_start = i
        end
        pos = q
    end
    push!(segments, Segment(typ, pat, seg_start, length(path)))

    segments
end

# ============================================================================
# Show
# ============================================================================

mode_name(::Type{NullMode}) = "Null"
mode_name(::Type{SingleMode}) = "Single"
mode_name(::Type{LoopMode}) = "Loop"

function Base.show(io::IO, ::MIME"text/plain", m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    ns = length(m.states)
    println(io, "SegmentHMM{$(mode_name(M))} with $np patterns, $ns states")

    if M === NullMode
        println(io, "  N ⟲")
    elseif M === SingleMode
        println(io, "  N₀ ⟲ → Pᵢ → N₁ ⟲")
    elseif M === LoopMode
        println(io, "  N ⟲ ⇄ Pᵢ")
    end

    np > 0 || return
    println(io, "  Patterns: ", join(["P$i($len)" for (i, len) in enumerate(m.pattern_lengths)], ", "))
end

function Base.show(io::IO, m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    print(io, "SegmentHMM{$(mode_name(M))}($np patterns)")
end

end # module DHMMs
