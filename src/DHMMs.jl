"""
    DHMMs

Segment-detection profile HMMs built on HiddenMarkovModels.jl.

Each pattern is a profile HMM segment with match (`M`), insert (`I`), and
(silent, marginalised) delete states. 5'- and 3'-truncation are modelled by
geometric trim priors. Transition matrices are stored sparsely so Viterbi
iterates only over non-zero entries.
"""
module DHMMs

using Distributions: Categorical
using SparseArrays: sparse
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

At most one (possibly trimmed, indel-bearing) pattern instance per observation.
"""
struct SingleMode <: ModelMode end

"""
    LoopMode <: ModelMode

Multiple profile-HMM pattern segments. Topology: `N ⟲ ⇄ profileᵢ`

Multiple instances may be detected. Consecutive instances are separated by at
least one background emission unless `p_direct > 0`, in which case adjacent
pattern instances (with any combination of 3'-/5'-trim) are allowed.
"""
struct LoopMode <: ModelMode end

const StateInfo = Tuple{Symbol, Int, Int}

# ============================================================================
# Main struct
# ============================================================================

"""
    SegmentHMM{M<:ModelMode, H<:HMM}

Segment-detection HMM parameterised by mode `M`.

# Fields
- `hmm::H`: underlying HiddenMarkovModels.HMM (with sparse transition matrix)
- `states::Vector{StateInfo}`: metadata `(state_type, pattern_idx, position)` per
  state. `state_type ∈ (:N, :M, :I)`; `pattern_idx == 0` for background.
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

`p_trim = 0` concentrates mass on position 1, `p_trim → 1` approaches uniform,
untruncated mean trim length is `p_trim / (1 - p_trim)`.
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
# state offset `s` (= index of `M_{i,1}`). Exits go either to `exit_state`
# (background) with weight `1 - p_direct`, or directly into another pattern
# (any `(target, frac)` in `direct_targets`) with weight `p_direct * frac`.
# `direct_targets` must sum to 1 in `frac`.
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
                             p_dd::Float64,
                             p_direct::Float64,
                             direct_targets::Vector{Pair{Int, Float64}})
    L = length(pat)
    p_mm = 1.0 - p_mi - p_md
    p_im = 1.0 - p_ii
    p_dm = 1.0 - p_dd
    enter = 1.0 - p_3trim
    keep_n = 1.0 - p_direct

    @inline function set_exit!(row::Int, mass::Float64)
        trans[row, exit_state] += mass * keep_n
        if p_direct > 0
            for (tgt, frac) in direct_targets
                trans[row, tgt] += mass * p_direct * frac
            end
        end
    end

    for j in 1:L
        m_idx = s + 2 * (j - 1)
        i_idx = m_idx + 1

        states[m_idx] = (:M, pattern_idx, j)
        states[i_idx] = (:I, pattern_idx, j)
        dists[m_idx]  = match_emission(pat[j], n_symbols, match_prob)
        dists[i_idx]  = uniform_emission(n_symbols)

        if j < L
            trans[m_idx, m_idx + 2] = enter * p_mm
            trans[m_idx, i_idx]     = enter * p_mi
            for k in 2:(L - j)
                target = s + 2 * (j + k - 1)
                trans[m_idx, target] = enter * p_md * p_dd^(k - 2) * p_dm
            end
            leakage = p_md * p_dd^(L - j - 1)
            set_exit!(m_idx, p_3trim + enter * leakage)

            trans[i_idx, i_idx]     = p_ii
            trans[i_idx, m_idx + 2] = p_im
        else
            trans[m_idx, i_idx] = enter * p_mi
            set_exit!(m_idx, 1.0 - enter * p_mi)
            trans[i_idx, i_idx] = p_ii
            set_exit!(i_idx, 1.0 - p_ii)
        end
    end
end

# Build [(M_{k,j} target_state, fraction)] over all patterns using each
# pattern's 5'-trim PMF, normalised across patterns by 1 / n_pat.
function _entry_targets(starts::Vector{Int}, patterns::Vector{Vector{Int}}, p_5trim::Float64)
    n_pat = length(patterns)
    targets = Pair{Int, Float64}[]
    for (k, pat) in enumerate(patterns)
        s = starts[k]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            push!(targets, (s + 2 * (j - 1)) => pmf[j] / n_pat)
        end
    end
    targets
end

# ============================================================================
# Constructors
# ============================================================================

"""
    SegmentHMM(mode::ModelMode, patterns::Vector{Vector{Int}}; kwargs...)

Construct a segment-detection HMM. Each pattern is realised as a profile-HMM
block with match, insert, and silent (marginalised) delete states. The
transition matrix is stored sparsely.

# Arguments
- `mode`: model topology ([`NullMode`](@ref), [`SingleMode`](@ref), [`LoopMode`](@ref))
- `patterns`: vector of integer sequences (the match track of each profile)

# Keyword arguments
- `n_symbols = 4`: alphabet size.
- `match_prob = 0.9`: per-position match emission probability.
- `p_stay_n = 0.6` (Single) / `0.75` (Loop): background self-loop.
- `p_skip = 0.05` (Single only): prior probability of no pattern at all.
- `p_5trim = 0.0`: geometric 5'-trim probability (entry distribution).
- `p_3trim = 0.1` (Single) / `0.05` (Loop): per-match-position 3'-trim
  (per-step exit probability).
- `p_mi = 0.025`: open-insertion probability (M → I).
- `p_md = 0.025`: open-deletion probability (M → D).
- `p_ii = 0.3`: insertion extension (I → I).
- `p_dd = 0.3`: deletion extension (D → D).
- `p_direct = 0.0` (Loop only): probability that an exit from one profile goes
  directly into another profile (any pattern, including self) instead of
  through background. Distributed across target patterns and entry positions
  by the same trim prior as the N → P entries. `0` reproduces the previous
  "always separate Ds by ≥ 1 background emission" behaviour.

Set `p_mi = p_md = 0` to recover a pure match-only model.
"""
function SegmentHMM(::NullMode, patterns::Vector{Vector{Int}}=Vector{Int}[];
                    n_symbols::Int=4, kwargs...)
    dists = [uniform_emission(n_symbols)]
    hmm = HMM([1.0], sparse(fill(1.0, 1, 1)), dists)
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

    states[1]     = (:N, 0, 0)
    states[n_end] = (:N, 0, 0)
    dists[1]      = uniform_emission(n_symbols)
    dists[n_end]  = uniform_emission(n_symbols)

    trans[1, 1]         = p_stay_n
    trans[1, n_end]     = p_skip
    trans[n_end, n_end] = 1.0

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + 2 * (j - 1)] = pat_mass * pmf[j] / n_pat
        end
        _add_profile_block!(trans, dists, states, s, pat, i, n_end,
                            n_symbols, match_prob, p_3trim,
                            p_mi, p_md, p_ii, p_dd,
                            0.0, Pair{Int, Float64}[])
    end

    init = zeros(n_states)
    init[1] = 1.0
    hmm = HMM(init, sparse(trans), dists)
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
                    p_dd::Float64=0.3,
                    p_direct::Float64=0.0)
    isempty(patterns) && throw(ArgumentError("patterns must be non-empty for LoopMode"))
    pat_mass = 1.0 - p_stay_n
    pat_mass > 0 || throw(ArgumentError("p_stay_n must be < 1"))
    p_mi + p_md <= 1.0 || throw(ArgumentError("p_mi + p_md must be <= 1"))
    0 <= p_ii < 1 || throw(ArgumentError("p_ii must be in [0, 1)"))
    0 <= p_dd < 1 || throw(ArgumentError("p_dd must be in [0, 1)"))
    0 <= p_direct <= 1 || throw(ArgumentError("p_direct must be in [0, 1]"))

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

    direct_targets = _entry_targets(starts, patterns, p_5trim)

    for (i, pat) in enumerate(patterns)
        s = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + 2 * (j - 1)] = pat_mass * pmf[j] / n_pat
        end
        _add_profile_block!(trans, dists, states, s, pat, i, 1,
                            n_symbols, match_prob, p_3trim,
                            p_mi, p_md, p_ii, p_dd,
                            p_direct, direct_targets)
    end

    init = zeros(n_states)
    init[1] = 1.0
    hmm = HMM(init, sparse(trans), dists)
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

Most-likely state path and its log-likelihood. Uses the sparse
`argmaxplus_transmul!` specialisation in HiddenMarkovModels.jl.
"""
HiddenMarkovModels.viterbi(m::SegmentHMM, obs) = viterbi(m.hmm, obs)

"""
    forward_backward(m::SegmentHMM, obs) -> (γ, logL)

Posterior state marginals. Aggregate by pattern via `m.states[s][2]` for
per-pattern occupancy at each position.
"""
HiddenMarkovModels.forward_backward(m::SegmentHMM, obs) = forward_backward(m.hmm, obs)

# ============================================================================
# Segment
# ============================================================================

"""
    Segment

Decoded segment from Viterbi path.

# Fields
- `type::Symbol`: `:N` (background) or `:P` (pattern; collapses `:M` / `:I` states)
- `pattern::Int`: pattern index (`0` for background)
- `start::Int`: observation start (1-based, inclusive)
- `stop::Int`: observation stop (inclusive)
- `profile_start::Int`: first profile position visited (1-based); `0` for `:N`.
  5'-trim is `profile_start - 1`.
- `profile_stop::Int`: last profile position visited; `0` for `:N`. 3'-trim is
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

@inline _profile_pos(info::StateInfo) = info[1] === :N ? 0 : info[3]

"""
    decode(m::SegmentHMM, obs) -> Vector{Segment}

Decode `obs` into segments via Viterbi. Match and insert states of one profile
collapse into a single `:P` segment with `profile_start` / `profile_stop`
covering the M-position range visited. Adjacent occurrences of the same
pattern (whether separated by background or joined via `p_direct`) are
returned as distinct segments; the latter is detected by a non-increasing
M-position within a candidate `:P` run.
"""
function decode(m::SegmentHMM, obs::AbstractVector{<:Integer})
    isempty(obs) && return Segment[]
    path, _ = viterbi(m.hmm, obs)

    info       = m.states[path[1]]
    cur_type   = info[1] === :N ? :N : :P
    cur_pat    = cur_type === :N ? 0  : info[2]
    seg_start  = 1
    seg_pstart = _profile_pos(info)
    seg_pstop  = seg_pstart
    last_m_pos = info[1] === :M ? info[3] : 0
    segments   = Segment[]

    for i in 2:length(path)
        info     = m.states[path[i]]
        new_type = info[1] === :N ? :N : :P
        new_pat  = new_type === :N ? 0 : info[2]

        is_new = new_type !== cur_type || new_pat != cur_pat
        if !is_new && info[1] === :M && last_m_pos > 0 && info[3] <= last_m_pos
            is_new = true
        end

        if is_new
            push!(segments, Segment(cur_type, cur_pat, seg_start, i - 1, seg_pstart, seg_pstop))
            cur_type   = new_type
            cur_pat    = new_pat
            seg_start  = i
            seg_pstart = _profile_pos(info)
            seg_pstop  = seg_pstart
            last_m_pos = info[1] === :M ? info[3] : 0
        elseif info[1] !== :N
            seg_pstop = info[3]
            info[1] === :M && (last_m_pos = info[3])
        end
    end
    push!(segments, Segment(cur_type, cur_pat, seg_start, length(path), seg_pstart, seg_pstop))
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
    println(io, "SegmentHMM{$(mode_name(M))} with $np patterns, $ns states (profile HMM, sparse)")

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
