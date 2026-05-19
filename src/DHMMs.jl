"""
    DHMMs

Segment-detection profile HMMs built on HiddenMarkovModels.jl.

Each pattern is a profile HMM segment with match (`Match`), insert (`Insert`),
and silent (marginalised) delete states. 5'/3'-truncation is modelled by
geometric trim priors. Transition matrices are stored sparsely so Viterbi
iterates only over non-zero entries.
"""
module DHMMs

using Distributions: Categorical
using SparseArrays: sparse
using PrecompileTools: @setup_workload, @compile_workload
import HiddenMarkovModels
using HiddenMarkovModels: HMM, viterbi, logdensityof, forward_backward

export SegmentHMM, NullMode, SingleMode, LoopMode
export decode, Segment, logdensityof, viterbi, forward_backward

# ============================================================================
# Topology modes
# ============================================================================

"""
    ModelMode

Abstract type for model topologies.
"""
abstract type ModelMode end

"""
    NullMode <: ModelMode

Background-only model. Topology: `N ⟲`.
"""
struct NullMode <: ModelMode end

"""
    SingleMode <: ModelMode

At most one (possibly trimmed, indel-bearing) profile instance.
Topology: `N₀ ⟲ → profileᵢ → N₁ ⟲`.
"""
struct SingleMode <: ModelMode end

"""
    LoopMode <: ModelMode

Zero, one, or many profile instances. Topology: `N ⟲ ⇄ profileᵢ`. Instances
are separated by ≥ 1 background emission unless `p_direct > 0`, in which case
direct profile-to-profile transitions are also allowed.
"""
struct LoopMode <: ModelMode end

mode_name(::NullMode)   = "Null"
mode_name(::SingleMode) = "Single"
mode_name(::LoopMode)   = "Loop"

# ============================================================================
# State info: small abstract type with three concrete leaves
# ============================================================================

"""
    StateInfo

Per-state metadata. One of [`Background`](@ref), [`Match`](@ref), or
[`Insert`](@ref). Access through dispatched methods
[`segment_kind`](@ref), [`pattern_idx`](@ref), [`profile_pos`](@ref).
"""
abstract type StateInfo end

"Background (`N`) emission state. Singleton."
struct Background <: StateInfo end

"Match state at `position` within `pattern`."
struct Match <: StateInfo
    pattern::Int
    position::Int
end

"Insert state at `position` within `pattern`."
struct Insert <: StateInfo
    pattern::Int
    position::Int
end

"""
    segment_kind(s::StateInfo) -> Symbol

`:N` for background, `:P` for any profile state. This is the `type` carried in
each [`Segment`](@ref) so consumers can pattern-match on user-facing labels.
"""
segment_kind(::Background) = :N
segment_kind(::Match)      = :P
segment_kind(::Insert)     = :P

"""
    pattern_idx(s::StateInfo) -> Int

Index into the supplied pattern list; `0` for background.
"""
pattern_idx(::Background) = 0
pattern_idx(s::Match)     = s.pattern
pattern_idx(s::Insert)    = s.pattern

"""
    profile_pos(s::StateInfo) -> Int

Profile position the state corresponds to; `0` for background.
"""
profile_pos(::Background) = 0
profile_pos(s::Match)     = s.position
profile_pos(s::Insert)    = s.position

# Used by decode for tracking the last visited Match position in a segment.
# Insert states do not update this counter (they sit between matches).
match_position(::Background) = 0
match_position(s::Match)     = s.position
match_position(::Insert)     = 0

# Direct re-entry of the same pattern: only an arriving Match state whose
# position has not strictly advanced beyond the segment's prior match track
# constitutes a new segment.
reentry(::StateInfo, ::Integer) = false
reentry(s::Match, prev_max::Integer) = prev_max > 0 && s.position ≤ prev_max

# ============================================================================
# SegmentHMM
# ============================================================================

"""
    SegmentHMM{M<:ModelMode, H<:HMM}

Segment-detection HMM parameterised by topology mode `M`.

# Fields
- `hmm::H`: underlying HiddenMarkovModels.HMM (with sparse transition matrix)
- `states::Vector{StateInfo}`: per-state metadata
- `pattern_lengths::Vector{Int}`: length of each pattern's match track
"""
struct SegmentHMM{M<:ModelMode, H<:HMM}
    hmm::H
    states::Vector{StateInfo}
    pattern_lengths::Vector{Int}
end

# ============================================================================
# Emissions
# ============================================================================

function match_emission(symbol::Integer, n_symbols::Integer, match_prob::Real)
    p = fill((1.0 - match_prob) / (n_symbols - 1), n_symbols)
    p[Int(symbol)] = match_prob
    Categorical(p)
end

uniform_emission(n_symbols::Integer) = Categorical(fill(1.0 / n_symbols, n_symbols))

# ============================================================================
# Trim prior
# ============================================================================

"""
    trim_pmf(len, p_trim) -> Vector{Float64}

Truncated geometric PMF over positions `1:len`:
`P(enter at position j) ∝ p_trim^(j-1)`, normalised.

`p_trim = 0` concentrates mass at position 1; `p_trim → 1` approaches uniform;
untruncated geometric mean trim length is `p_trim / (1 - p_trim)`.
"""
function trim_pmf(len::Integer, p_trim::Real)
    0.0 ≤ p_trim < 1.0 || throw(ArgumentError("p_trim must be in [0, 1)"))
    len ≥ 1            || throw(ArgumentError("len must be ≥ 1"))
    p = Float64(p_trim)
    w = [p^(j - 1) for j in 1:len]
    w ./ sum(w)
end

# ============================================================================
# Profile-HMM block construction
# ============================================================================

# (target_state => fraction); fractions sum to 1 across all entry positions of
# all patterns, weighted by 1 / n_pat and the per-pattern 5'-trim PMF.
const DirectTargets = Vector{Pair{Int, Float64}}

function entry_targets(starts::AbstractVector{<:Integer},
                       patterns::AbstractVector{<:AbstractVector{<:Integer}},
                       p_5trim::Real)
    n_pat = length(patterns)
    targets = DirectTargets()
    for (k, pat) in enumerate(patterns)
        s = starts[k]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            push!(targets, (s + 2 * (j - 1)) => pmf[j] / n_pat)
        end
    end
    targets
end

function distribute_exit!(trans::AbstractMatrix{Float64},
                          row::Integer,
                          mass::Real,
                          exit_state::Integer,
                          keep_n::Real,
                          p_direct::Real,
                          direct_targets::DirectTargets)
    trans[row, exit_state] += mass * keep_n
    p_direct > 0 || return
    for (tgt, frac) in direct_targets
        trans[row, tgt] += mass * p_direct * frac
    end
    return
end

function add_profile_block!(trans::AbstractMatrix{Float64},
                            dists::AbstractVector,
                            states::AbstractVector{StateInfo},
                            s::Integer,
                            pat::AbstractVector{<:Integer},
                            pat_idx::Integer,
                            exit_state::Integer,
                            n_symbols::Integer,
                            match_prob::Real,
                            p_3trim::Real,
                            p_mi::Real,
                            p_md::Real,
                            p_ii::Real,
                            p_dd::Real,
                            p_direct::Real,
                            direct_targets::DirectTargets)
    L = length(pat)
    p_mm   = 1.0 - p_mi - p_md
    p_im   = 1.0 - p_ii
    p_dm   = 1.0 - p_dd
    enter  = 1.0 - p_3trim
    keep_n = 1.0 - p_direct

    for j in 1:L
        m_idx = s + 2 * (j - 1)
        i_idx = m_idx + 1

        states[m_idx] = Match(pat_idx, j)
        states[i_idx] = Insert(pat_idx, j)
        dists[m_idx]  = match_emission(pat[j], n_symbols, match_prob)
        dists[i_idx]  = uniform_emission(n_symbols)

        if j < L
            trans[m_idx, m_idx + 2] = enter * p_mm
            trans[m_idx, i_idx]     = enter * p_mi
            for k in 2:(L - j)
                trans[m_idx, s + 2 * (j + k - 1)] = enter * p_md * p_dd^(k - 2) * p_dm
            end
            leakage = p_md * p_dd^(L - j - 1)
            distribute_exit!(trans, m_idx, p_3trim + enter * leakage,
                             exit_state, keep_n, p_direct, direct_targets)

            trans[i_idx, i_idx]     = p_ii
            trans[i_idx, m_idx + 2] = p_im
        else
            trans[m_idx, i_idx] = enter * p_mi
            distribute_exit!(trans, m_idx, 1.0 - enter * p_mi,
                             exit_state, keep_n, p_direct, direct_targets)
            trans[i_idx, i_idx] = p_ii
            distribute_exit!(trans, i_idx, 1.0 - p_ii,
                             exit_state, keep_n, p_direct, direct_targets)
        end
    end
    return
end

# ============================================================================
# Constructors (dispatch on the mode singleton)
# ============================================================================

"""
    SegmentHMM(mode::ModelMode, patterns; kwargs...)

Construct a segment-detection HMM. See [`NullMode`](@ref), [`SingleMode`](@ref),
[`LoopMode`](@ref).

# Keyword arguments
- `n_symbols = 4`: alphabet size.
- `match_prob = 0.9`: per-position match emission probability.
- `p_stay_n = 0.6` (Single) / `0.75` (Loop): background self-loop.
- `p_skip = 0.05` (Single only): prior probability of no pattern at all.
- `p_5trim = 0.0`: geometric 5'-trim probability (entry distribution).
- `p_3trim = 0.1` (Single) / `0.05` (Loop): per-match-position 3'-trim.
- `p_mi = 0.025`: open-insertion probability (M → I).
- `p_md = 0.025`: open-deletion probability (M → D).
- `p_ii = 0.3`: insertion extension (I → I).
- `p_dd = 0.3`: deletion extension (D → D).
- `p_direct = 0.0` (Loop only): probability that a profile exit feeds directly
  into another profile entry (distributed by the 5'-trim PMF) instead of
  through background. `0` recovers the "always separate by ≥ 1 N emission".

Set `p_mi = p_md = 0` for a pure match-only profile.
"""
function SegmentHMM(::NullMode,
                    patterns::AbstractVector{<:AbstractVector{<:Integer}}=Vector{Int}[];
                    n_symbols::Integer=4, kwargs...)
    dists = [uniform_emission(n_symbols)]
    hmm = HMM([1.0], sparse(fill(1.0, 1, 1)), dists)
    SegmentHMM{NullMode, typeof(hmm)}(hmm, StateInfo[Background()], Int[])
end

function SegmentHMM(::SingleMode,
                    patterns::AbstractVector{<:AbstractVector{<:Integer}};
                    n_symbols::Integer=4,
                    p_stay_n::Real=0.6,
                    p_skip::Real=0.05,
                    p_5trim::Real=0.0,
                    p_3trim::Real=0.1,
                    match_prob::Real=0.9,
                    p_mi::Real=0.025,
                    p_md::Real=0.025,
                    p_ii::Real=0.3,
                    p_dd::Real=0.3)
    isempty(patterns)                  && throw(ArgumentError("patterns must be non-empty"))
    pat_mass = 1.0 - p_stay_n - p_skip
    pat_mass > 0                       || throw(ArgumentError("p_stay_n + p_skip must be < 1"))
    p_mi + p_md ≤ 1.0                  || throw(ArgumentError("p_mi + p_md must be ≤ 1"))
    0 ≤ p_ii < 1                       || throw(ArgumentError("p_ii must be in [0, 1)"))
    0 ≤ p_dd < 1                       || throw(ArgumentError("p_dd must be in [0, 1)"))

    lens     = length.(patterns)
    n_pat    = length(patterns)
    n_states = 2 + 2 * sum(lens)
    n_end    = n_states

    starts = Vector{Int}(undef, n_pat)
    cum = 1
    for i in 1:n_pat
        starts[i] = cum + 1
        cum += 2 * lens[i]
    end

    trans  = zeros(n_states, n_states)
    dists  = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    states = Vector{StateInfo}(undef, n_states)

    states[1]     = Background()
    states[n_end] = Background()
    dists[1]      = uniform_emission(n_symbols)
    dists[n_end]  = uniform_emission(n_symbols)

    trans[1, 1]         = p_stay_n
    trans[1, n_end]     = p_skip
    trans[n_end, n_end] = 1.0

    for (i, pat) in enumerate(patterns)
        s   = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + 2 * (j - 1)] = pat_mass * pmf[j] / n_pat
        end
        add_profile_block!(trans, dists, states, s, pat, i, n_end,
                           n_symbols, match_prob, p_3trim,
                           p_mi, p_md, p_ii, p_dd,
                           0.0, DirectTargets())
    end

    init = zeros(n_states)
    init[1] = 1.0
    hmm = HMM(init, sparse(trans), dists)
    SegmentHMM{SingleMode, typeof(hmm)}(hmm, states, lens)
end

function SegmentHMM(::LoopMode,
                    patterns::AbstractVector{<:AbstractVector{<:Integer}};
                    n_symbols::Integer=4,
                    p_stay_n::Real=0.75,
                    p_5trim::Real=0.0,
                    p_3trim::Real=0.05,
                    match_prob::Real=0.9,
                    p_mi::Real=0.025,
                    p_md::Real=0.025,
                    p_ii::Real=0.3,
                    p_dd::Real=0.3,
                    p_direct::Real=0.0)
    isempty(patterns)                  && throw(ArgumentError("patterns must be non-empty"))
    pat_mass = 1.0 - p_stay_n
    pat_mass > 0                       || throw(ArgumentError("p_stay_n must be < 1"))
    p_mi + p_md ≤ 1.0                  || throw(ArgumentError("p_mi + p_md must be ≤ 1"))
    0 ≤ p_ii < 1                       || throw(ArgumentError("p_ii must be in [0, 1)"))
    0 ≤ p_dd < 1                       || throw(ArgumentError("p_dd must be in [0, 1)"))
    0 ≤ p_direct ≤ 1                   || throw(ArgumentError("p_direct must be in [0, 1]"))

    lens     = length.(patterns)
    n_pat    = length(patterns)
    n_states = 1 + 2 * sum(lens)

    starts = Vector{Int}(undef, n_pat)
    cum = 1
    for i in 1:n_pat
        starts[i] = cum + 1
        cum += 2 * lens[i]
    end

    trans  = zeros(n_states, n_states)
    dists  = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    states = Vector{StateInfo}(undef, n_states)

    states[1]   = Background()
    dists[1]    = uniform_emission(n_symbols)
    trans[1, 1] = p_stay_n

    direct_targets = entry_targets(starts, patterns, p_5trim)

    for (i, pat) in enumerate(patterns)
        s   = starts[i]
        pmf = trim_pmf(length(pat), p_5trim)
        for j in 1:length(pat)
            trans[1, s + 2 * (j - 1)] = pat_mass * pmf[j] / n_pat
        end
        add_profile_block!(trans, dists, states, s, pat, i, 1,
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

Marginal log-likelihood of `obs` (forward algorithm).
"""
HiddenMarkovModels.logdensityof(m::SegmentHMM, obs) = logdensityof(m.hmm, obs)

"""
    viterbi(m::SegmentHMM, obs) -> (path, loglik)

Most-likely state path. Uses HiddenMarkovModels.jl's sparse
`argmaxplus_transmul!` specialisation since `m.hmm.trans isa SparseMatrixCSC`.
"""
HiddenMarkovModels.viterbi(m::SegmentHMM, obs) = viterbi(m.hmm, obs)

"""
    forward_backward(m::SegmentHMM, obs) -> (γ, logL)

Posterior state marginals. Aggregate over a pattern by summing rows whose
[`pattern_idx`](@ref) matches.
"""
HiddenMarkovModels.forward_backward(m::SegmentHMM, obs) = forward_backward(m.hmm, obs)

# ============================================================================
# Segment
# ============================================================================

"""
    Segment

Decoded segment from Viterbi path.

# Fields
- `type::Symbol`: `:N` or `:P`
- `pattern::Int`: pattern index (`0` for `:N`)
- `start::Int`, `stop::Int`: observation positions (inclusive)
- `profile_start::Int`, `profile_stop::Int`: profile positions visited; `0` for
  `:N`. 5'-trim = `profile_start - 1`, 3'-trim = `pattern_lengths[pattern] - profile_stop`.
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

"""
    decode(m::SegmentHMM, obs) -> Vector{Segment}

Decode `obs` into segments. Match and insert states of one profile collapse
into a single `:P` segment; same-pattern direct re-entries (e.g. under
`p_direct > 0`) are split via the non-increasing M-position rule.
"""
function decode(m::SegmentHMM, obs::AbstractVector{<:Integer})
    isempty(obs) && return Segment[]
    path, _ = viterbi(m.hmm, obs)

    info       = m.states[path[1]]
    cur_kind   = segment_kind(info)
    cur_pat    = pattern_idx(info)
    seg_start  = 1
    seg_pstart = profile_pos(info)
    seg_pstop  = seg_pstart
    prev_m_pos = match_position(info)
    segments   = Segment[]

    for t in 2:length(path)
        info     = m.states[path[t]]
        new_kind = segment_kind(info)
        new_pat  = pattern_idx(info)
        if new_kind !== cur_kind || new_pat != cur_pat || reentry(info, prev_m_pos)
            push!(segments, Segment(cur_kind, cur_pat, seg_start, t - 1,
                                    seg_pstart, seg_pstop))
            cur_kind   = new_kind
            cur_pat    = new_pat
            seg_start  = t
            seg_pstart = profile_pos(info)
            seg_pstop  = seg_pstart
            prev_m_pos = match_position(info)
        else
            p = profile_pos(info)
            p > 0 && (seg_pstop = p)
            mp = match_position(info)
            mp > 0 && (prev_m_pos = mp)
        end
    end
    push!(segments, Segment(cur_kind, cur_pat, seg_start, length(path),
                            seg_pstart, seg_pstop))
    segments
end

# ============================================================================
# Show methods (dispatched on the mode singleton)
# ============================================================================

show_topology(io::IO, ::NullMode)   = println(io, "  N ⟲")
show_topology(io::IO, ::SingleMode) = println(io, "  N₀ ⟲ → profileᵢ → N₁ ⟲")
show_topology(io::IO, ::LoopMode)   = println(io, "  N ⟲ ⇄ profileᵢ")

function Base.show(io::IO, ::MIME"text/plain", m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    ns = length(m.states)
    println(io, "SegmentHMM{$(mode_name(M()))} with $np patterns, $ns states (profile HMM, sparse)")
    show_topology(io, M())
    np > 0 || return
    println(io, "  Patterns: ",
            join(["P$i($len)" for (i, len) in enumerate(m.pattern_lengths)], ", "))
end

function Base.show(io::IO, m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    print(io, "SegmentHMM{$(mode_name(M()))}($np patterns)")
end

# ============================================================================
# Precompile workload — pay inference and codegen at build time
# ============================================================================

@setup_workload begin
    patterns = [[1, 2, 3], [3, 3, 1, 2]]
    obs      = [4, 1, 2, 3, 4, 1, 2, 3]
    @compile_workload begin
        m_null   = SegmentHMM(NullMode())
        m_single = SegmentHMM(SingleMode(), patterns; p_5trim=0.5)
        m_loop   = SegmentHMM(LoopMode(), patterns; p_5trim=0.5, p_direct=0.1)
        logdensityof(m_null,   obs)
        logdensityof(m_single, obs)
        logdensityof(m_loop,   obs)
        viterbi(m_loop, obs)
        forward_backward(m_loop, obs)
        decode(m_loop, obs)
    end
end

end # module DHMMs
