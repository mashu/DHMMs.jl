"""
    DHMMs

Segment detection Hidden Markov Models built on HiddenMarkovModels.jl.
"""
module DHMMs

using Distributions: Categorical
import HiddenMarkovModels
using HiddenMarkovModels: HMM, viterbi, logdensityof

export SegmentHMM, NullMode, SingleMode, LoopMode
export decode, Segment, logdensityof, viterbi

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

Allows at most one pattern segment with optional skip.
"""
struct SingleMode <: ModelMode end

"""
    LoopMode <: ModelMode

Loop model allowing multiple patterns. Topology: `N ⟲ ⇄ Pᵢ ⟲`

Patterns can chain directly via `p_direct` parameter.
"""
struct LoopMode <: ModelMode end

const StateInfo = Tuple{Symbol, Int, Int}

# ============================================================================
# Main struct
# ============================================================================

"""
    SegmentHMM{M<:ModelMode, H<:HMM}

Segment detection HMM parameterized by mode `M`.

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

@inline function match_emission(symbol::Int, n_symbols::Int, match_prob::Float64)
    p = fill((1.0 - match_prob) / (n_symbols - 1), n_symbols)
    p[symbol] = match_prob
    Categorical(p)
end

@inline uniform_emission(n_symbols::Int) = Categorical(fill(1.0 / n_symbols, n_symbols))

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
- `match_prob=0.85`: emission probability for matching symbol
- `p_stay_n=0.6`: self-loop probability for N state
- `p_continue=0.9` (Single) / `0.95` (Loop): continue within pattern
- `p_skip=0.05` (Single only): skip patterns entirely
- `p_direct=0.01` (Loop only): direct pattern-to-pattern transition
"""
function SegmentHMM(::NullMode, ::Vector{Vector{Int}}; n_symbols::Int=4, kwargs...)
    dists = [uniform_emission(n_symbols)]
    hmm = HMM([1.0], fill(1.0, 1, 1), dists)
    SegmentHMM{NullMode, typeof(hmm)}(hmm, [(:N, 0, 0)], Int[])
end

function SegmentHMM(::SingleMode, patterns::Vector{Vector{Int}};
                    n_symbols::Int=4, p_stay_n::Float64=0.6, p_skip::Float64=0.05,
                    p_continue::Float64=0.9, match_prob::Float64=0.85)
    lens = length.(patterns)
    total_p = sum(lens)
    n_states = 2 + total_p
    
    starts = cumsum([0; lens[1:end-1]]) .+ 2
    n_end = n_states
    
    trans = zeros(n_states, n_states)
    trans[1, 1] = p_stay_n
    trans[1, n_end] = p_skip
    p_per = (1.0 - p_stay_n - p_skip) / total_p
    
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for j in 1:length(pat)
            trans[1, s + j - 1] = p_per
        end
    end
    
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for j in 1:length(pat)
            cur = s + j - 1
            if j < length(pat)
                trans[cur, cur + 1] = p_continue
                trans[cur, n_end] = 1.0 - p_continue
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
                    n_symbols::Int=4, p_stay_n::Float64=0.6, p_continue::Float64=0.95,
                    p_direct::Float64=0.01, match_prob::Float64=0.85)
    lens = length.(patterns)
    total_p = sum(lens)
    n_states = 1 + total_p
    
    starts = cumsum([0; lens[1:end-1]]) .+ 2
    
    trans = zeros(n_states, n_states)
    trans[1, 1] = p_stay_n
    p_per = (1.0 - p_stay_n) / total_p
    
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for j in 1:length(pat)
            trans[1, s + j - 1] = p_per
        end
    end
    
    for (i, pat) in enumerate(patterns)
        s = starts[i]
        for j in 1:length(pat)
            cur = s + j - 1
            if j < length(pat)
                trans[cur, cur + 1] = p_continue
                trans[cur, 1] = 1.0 - p_continue
            else
                trans[cur, 1] = 1.0 - p_direct
                if p_direct > 0
                    p_per2 = p_direct / total_p
                    for (k, pat2) in enumerate(patterns)
                        s2 = starts[k]
                        for m in 1:length(pat2)
                            trans[cur, s2 + m - 1] += p_per2
                        end
                    end
                end
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

Compute log-likelihood of observation sequence under the model.
"""
HiddenMarkovModels.logdensityof(m::SegmentHMM, obs) = logdensityof(m.hmm, obs)

"""
    viterbi(m::SegmentHMM, obs)

Run Viterbi algorithm. Returns `(path, loglik)`.
"""
HiddenMarkovModels.viterbi(m::SegmentHMM, obs) = viterbi(m.hmm, obs)

# ============================================================================
# Segment
# ============================================================================

"""
    Segment

Decoded segment from Viterbi path.

# Fields
- `type::Symbol`: `:N` (background) or `:P` (pattern)
- `pattern::Int`: pattern index (0 for background)
- `start::Int`: start position in observation
- `stop::Int`: end position in observation
"""
struct Segment
    type::Symbol
    pattern::Int
    start::Int
    stop::Int
end

"""
    decode(m::SegmentHMM, obs::AbstractVector{Int}) -> Vector{Segment}

Decode observation sequence into segments using Viterbi.
"""
function decode(m::SegmentHMM, obs::AbstractVector{Int})
    isempty(obs) && return Segment[]
    path, _ = viterbi(m.hmm, obs)
    
    segments = Segment[]
    typ, pat, _ = m.states[path[1]]
    seg_start = 1
    
    for i in 2:length(path)
        t, p, _ = m.states[path[i]]
        if (t, p) != (typ, pat)
            push!(segments, Segment(typ, pat, seg_start, i - 1))
            typ, pat = t, p
            seg_start = i
        end
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
        println(io, "  N ⟲ ⇄ Pᵢ ⟲")
    end
    
    np > 0 || return
    println(io, "  Patterns: ", join(["P$i($len)" for (i, len) in enumerate(m.pattern_lengths)], ", "))
end

function Base.show(io::IO, m::SegmentHMM{M}) where M
    np = length(m.pattern_lengths)
    print(io, "SegmentHMM{$(mode_name(M))}($np patterns)")
end

end # module DHMMs
