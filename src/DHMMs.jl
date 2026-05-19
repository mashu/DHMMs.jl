"""
    DHMMs

Segment-detection profile HMMs built on HiddenMarkovModels.jl.

Each pattern is a profile HMM segment with match (`M`), insert (`I`), and
(silent, marginalised) delete states. 5'- and 3'-truncation are modelled by
geometric trim priors. See the documentation for the topology and the
statistical model.
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

Single profile-HMM pattern segment. Topology: `N₀ ⟲ → profileᵢ → N₁ ⟲`

At most one (possibly trimmed) pattern instance per observation.
"""
struct SingleMode <: ModelMode end

"""
    LoopMode <: ModelMode

Multiple profile-HMM pattern segments. Topology: `N ⟲ ⇄ profileᵢ`

Multiple instances may be detected; consecutive instances are separated by at
least one background emission.
"""
struct LoopMode <: ModelMode end

# `(state_type, pattern_idx, position)`; `state_type ∈ (:N, :M, :I)`
const StateInfo = Tuple{Symbol, Int, Int}

# ============================================================================
# Main struct
# ============================================================================

"""
    SegmentHMM{M<:ModelMode, H<:HMM}

Segment-detection HMM parameterised by mode `M`.

# Fields
- `hmm::H`: underlying HiddenMarkovModels.HMM
- `states::Vector{StateInfo}`: metadata `(state_type, pattern_idx, position)` per
  state. `state_type` is `:N` for background, `:M` for a match position, `:I`
  for an insert position. `pattern_idx == 0` for background.
- `pattern_lengths::Vector{Int}`: length of each pattern's match track.
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

Truncated geometric PMF over positions `1:len`:
`P(enter at position j) ∝ p_trim^(j-1)`, normalised.

- `p_trim = 0` puts all mass on position 1 (no trim).
- `p_trim → 1` approaches uniform.
- Untruncated mean trim length is `p_trim / (1 - p_trim)`.
"""
function trim_pmf(len::Int, p_trim::Float64)
    0.0 <= p_trim < 1.0 || throw(ArgumentError("p_trim must be in [0, 1)"))
    len >= 1 || throw(ArgumentError("len must be >= 1"))
    w = [p_trim^(j - 1) for j in 1:len]
    w ./ sum(w)
end

# ============================================================================
# Profile-HMM block construction
# ============================================================================

# Fill in transitions, emissions, and state metadata for one profile block at
# state offset `s` (= index of `M_{i,1}`). Exits go to `exit_state`.
#
# Block layout (interleaved): for j = 1..L,
#   states[s + 2(j-1)]     = M_{i,j}
#   states[s + 2(j-1) + 1] = I_{i,j}
#
# Silent D states are marginalised: `M_j → M_{j+k}` for k ≥ 2 carries the
# analytic weight `p_md * p_dd^(k-2) * p_dm`. Leakage of the delete chain past
# the last position is folded into `M_j → exit_state` as effective 3'-trim.
function _add_profile_block!(trans::Matrix{Float64},
                             dists::Vector,
                             states::Vector{StateInfo},
                             s::Int,
                             pat::Vector{Int},
                             pattern_idx::Int,
                             exit_state::Int,
                             n_symbols::Int,
                             match_prob::Float64,
                             p_3trim::Float64,
                             p_mi::Float64,
                             p_md::Float64,
                             p_ii::Float64,
                             p_dd::Float64)
    L = length(pat)
    p_mm = 1.0 - p_mi - p_md
    p_im = 1.0 - p_ii
    p_dm = 1.0 - p_dd
    enter = 1.0 - p_3trim

    for j in 1:L
        m_idx = s + 2 * (j - 1)
        i_idx = m_idx + 1

        states[m_idx] = (:M, pattern_idx, j)
        states[i_idx] = (:I, pattern_idx, j)
        dists[m_idx]  = match_emission(pat[j], n_symbols, match_prob)
        dists[i_idx]  = uniform_emission(n_symbols)

        if j < L
            trans[m_idx, m_idx + 2] = enter * p_mm                  # → M_{j+1}
            trans[m_idx, i_idx]     = enter * p_mi                  # → I_j
            for k in 2:(L - j)                                      # → M_{j+k} via delete chain
                target = s + 2 * (j + k - 1)
                trans[m_idx, target] = enter * p_md * p_dd^(k - 2) * p_dm
            end
            leakage = p_md * p_dd^(L - j - 1)                        # delete chain falls off the end
            trans[m_idx, exit_state] = p_3trim + enter * leakage

            trans[i_idx, i_idx]     = p_ii                          # I self-loop (extend)
            trans[i_idx, m_idx + 2] = p_im                          # I → M_{j+1} (close)
        else
            trans[m_idx, i_idx]      = enter * p_mi                 # M_L → I_L
            trans[m_idx, exit_state] = 1.0 - enter * p_mi           # everything else → exit
            trans[i_idx, i_idx]      = p_ii                         # I_L self-loop
            trans[i_idx, exit_state] = p_im                         # I_L → exit
        end
    end
end

# ============================================================================
# Constructors
# ============================================================================

"""
    SegmentHMM(mode::ModelMode, patterns::Vector{Vector{Int}}; kwargs...)

Construct a segment-detection HMM. Each pattern is realised as a profile-HMM
block with match, insert, and (silent, marginalised) delete states.

# Arguments
- `mode`: model topology ([`NullMode`](@ref), [`SingleMode`](@ref), [`LoopMode`](@ref))
- `patterns`: vector of integer sequences (the match track of each profile)

# Keyword arguments
- `n_symbols = 4`: alphabet size.
- `match_prob = 0.9`: per-position emission probability of the matching symbol
  at match states; other symbols share the remainder uniformly.
- `p_stay_n = 0.6` (Single) / `0.75` (Loop): background self-loop probability.
- `p_skip = 0.05` (Single only): prior probability of no pattern at all.
- `p_5trim = 0.0`: per-step geometric 5'-trim probability for entry into the
  profile. `0` = always enter at position 1.
- `p_3trim = 0.1` (Single) / `0.05` (Loop): per-match-position exit probability
  to background (geometric 3'-trim).
- `p_mi = 0.025`: open-insertion probability (M → I).
- `p_md = 0.025`: open-deletion probability (M → D).
- `p_ii = 0.3`: insertion-extension probability (I → I).
- `p_dd = 0.3`: deletion-extension probability (D → D).

Set `p_mi = p_md = 0` to recover a pure match-only model.
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
                    match_prob::Float64=0.9,
                    p_mi::Float64=0.025,
                    p_md::Float64=0.025,
                    p_ii::Float64=0.3,
                    p_dd::Float64=0.3)
    isempty(patterns) && throw(ArgumentError("patterns must be non-empty for SingleMode"))
    pat_mass = 1.0 - p_stay_n - p_skip
    pat_mass > 0 || throw(ArgumentError("p_stay_n + p_skip must be < 1"))
    p_mi + p_md <= 1.0 || throw(ArgumentError("p_mi + p_md must be <= 1"))
    0 <= p_ii < 1 || throw(ArgumentError("p_ii must be in [0, 1)"))
    0 <= p_dd < 1 || throw(ArgumentError("p_dd must be in [0, 1)"))

    lens = length.(patterns)
    n_pat = length(patterns)
    n_states = 2 + 2 * sum(lens)
    n_end = n_states

    starts = Vector{Int}(undef, n_pat)
    cum = 1
    for i in 1:n_pat
        starts[i] = cum + 1
        cum += 2 * lens[i]
    end

    trans = zeros(n_states, n_states)
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    states = Vector{StateInfo}(undef, n_states)

    states[1] = (:N, 0, 0)
    states[n_end] = (:N, 0, 0)
    dists[1] = uniform_emission(n_symbols)
    dists[n_end] = uniform_emission(n_symbols)

    trans[1, 1]      = p_stay_n
    trans[1, n_end]  = p_skip
    trans[n_end, n_end] = 1.0

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + 2 * (j - 1)] = pat_mass * pmf[j] / n_pat
        end
        _add_profile_block!(trans, dists, states, s, pat, i, n_end,
                            n_symbols, match_prob, p_3trim,
                            p_mi, p_md, p_ii, p_dd)
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
                    match_prob::Float64=0.9,
                    p_mi::Float64=0.025,
                    p_md::Float64=0.025,
                    p_ii::Float64=0.3,
                    p_dd::Float64=0.3)
    isempty(patterns) && throw(ArgumentError("patterns must be non-empty for LoopMode"))
    pat_mass = 1.0 - p_stay_n
    pat_mass > 0 || throw(ArgumentError("p_stay_n must be < 1"))
    p_mi + p_md <= 1.0 || throw(ArgumentError("p_mi + p_md must be <= 1"))
    0 <= p_ii < 1 || throw(ArgumentError("p_ii must be in [0, 1)"))
    0 <= p_dd < 1 || throw(ArgumentError("p_dd must be in [0, 1)"))

    lens = length.(patterns)
    n_pat = length(patterns)
    n_states = 1 + 2 * sum(lens)

    starts = Vector{Int}(undef, n_pat)
    cum = 1
    for i in 1:n_pat
        starts[i] = cum + 1
        cum += 2 * lens[i]
    end

    trans = zeros(n_states, n_states)
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    states = Vector{StateInfo}(undef, n_states)

    states[1] = (:N, 0, 0)
    dists[1]  = uniform_emission(n_symbols)
    trans[1, 1] = p_stay_n

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + 2 * (j - 1)] = pat_mass * pmf[j] / n_pat
        end
        _add_profile_block!(trans, dists, states, s, pat, i, 1,
                            n_symbols, match_prob, p_3trim,
                            p_mi, p_md, p_ii, p_dd)
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

Posterior state marginals. `γ[s, t]` is `P(state_t = s | obs)`. Aggregate by
pattern using `m.states[s][2]` to obtain per-pattern occupancy per position.
"""
HiddenMarkovModels.forward_backward(m::SegmentHMM, obs) = forward_backward(m.hmm, obs)

# ============================================================================
# Segment
# ============================================================================

"""
    Segment

Decoded segment from Viterbi path.

# Fields
- `type::Symbol`: `:N` (background) or `:P` (pattern; collapses `:M` and `:I` states)
- `pattern::Int`: pattern index (`0` for background)
- `start::Int`: start position in observation (1-based, inclusive)
- `stop::Int`: end position in observation (inclusive)
- `profile_start::Int`: first profile position visited within the pattern
  (1-based, inclusive); `0` for background segments. Implies 5'-trim of
  `profile_start - 1`.
- `profile_stop::Int`: last profile position visited (inclusive); `0` for
  background segments. With `pattern_lengths[pattern]`, gives 3'-trim of
  `pattern_lengths[pattern] - profile_stop`.
"""
struct Segment
    type::Symbol
    pattern::Int
    start::Int
    stop::Int
    profile_start::Int
    profile_stop::Int
end

function Base.show(io::IO, s::Segment)
    if s.type === :N
        print(io, "Segment(N, obs ", s.start, "-", s.stop, ")")
    else
        print(io, "Segment(P", s.pattern,
              " profile ", s.profile_start, "-", s.profile_stop,
              ", obs ", s.start, "-", s.stop, ")")
    end
end

@inline _segment_key(info::StateInfo) = (info[1] === :N ? :N : :P, info[2])
@inline _profile_pos(info::StateInfo) = info[1] === :N ? 0 : info[3]

"""
    decode(m::SegmentHMM, obs) -> Vector{Segment}

Decode `obs` into segments via Viterbi. Match and insert states of a profile
collapse into a single `:P` segment for the corresponding pattern; the
`profile_start` / `profile_stop` fields report which positions of the profile
were visited (so 5'- and 3'-trim are recoverable from a single segment).
Adjacent occurrences of the same pattern (separated by background) are
returned as distinct segments.
"""
function decode(m::SegmentHMM, obs::AbstractVector{<:Integer})
    isempty(obs) && return Segment[]
    path, _ = viterbi(m.hmm, obs)

    segments = Segment[]
    info = m.states[path[1]]
    cur = _segment_key(info)
    seg_start = 1
    seg_pstart = _profile_pos(info)
    seg_pstop  = seg_pstart
    for i in 2:length(path)
        info = m.states[path[i]]
        k = _segment_key(info)
        if k != cur
            push!(segments, Segment(cur[1], cur[2], seg_start, i - 1, seg_pstart, seg_pstop))
            cur = k
            seg_start = i
            seg_pstart = _profile_pos(info)
            seg_pstop  = seg_pstart
        elseif info[1] !== :N
            seg_pstop = info[3]
        end
    end
    push!(segments, Segment(cur[1], cur[2], seg_start, length(path), seg_pstart, seg_pstop))
    segments
end

# ============================================================================
# Show
# ============================================================================

mode_name(::Type{NullMode})   = "Null"
mode_name(::Type{SingleMode}) = "Single"
mode_name(::Type{LoopMode})   = "Loop"

function Base.show(io::IO, ::MIME"text/plain", m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    ns = length(m.states)
    println(io, "SegmentHMM{$(mode_name(M))} with $np patterns, $ns states (profile HMM)")

    if M === NullMode
        println(io, "  N ⟲")
    elseif M === SingleMode
        println(io, "  N₀ ⟲ → profileᵢ → N₁ ⟲")
    elseif M === LoopMode
        println(io, "  N ⟲ ⇄ profileᵢ")
    end

    np > 0 || return
    println(io, "  Patterns: ", join(["P$i($len)" for (i, len) in enumerate(m.pattern_lengths)], ", "))
end

function Base.show(io::IO, m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    print(io, "SegmentHMM{$(mode_name(M))}($np patterns)")
end

end # module DHMMs
