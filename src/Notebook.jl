using Distributions
using HiddenMarkovModels
using BioSequences
using FASTX

function load_sequences(file::String)
    records = Vector{Tuple{String, LongDNA{4}}}()
    for record in FASTX.FASTAReader(open(file))
        push!(records, (description(record), sequence(LongDNA{4}, record)))
    end
    return records
end

# ============================================================================
# Helper functions
# ============================================================================

function dna_to_int(nt::DNA)
    nt == DNA_A && return 1
    nt == DNA_C && return 2
    nt == DNA_G && return 3
    nt == DNA_T && return 4
    error("Unknown nucleotide: $nt")
end

seq_to_ints(seq::LongSequence{DNAAlphabet{4}}) = [dna_to_int(nt) for nt in seq]

function match_emission(expected_nt::DNA; match_prob=0.85)
    probs = fill((1 - match_prob) / 3, 4)
    probs[dna_to_int(expected_nt)] = match_prob
    return Categorical(probs)
end

uniform_emission() = Categorical([0.25, 0.25, 0.25, 0.25])

# ============================================================================
# Genomic positions for IGHD genes and helpers
# ============================================================================

const IGHD_GENOMIC_POS = Dict(
    "IGHD7-27" => 100137087,
    "IGHD1-26" => 100152158,
    "IGHD6-25" => 100152663,
    "IGHD5-24" => 100155040,
    "IGHD4-23" => 100155994,
    "IGHD3-22" => 100157155,
    "IGHD2-21" => 100159627,
    "IGHD1-20" => 100162234,
    "IGHD6-19" => 100162742,
    "IGHD5-18" => 100164566,
    "IGHD4-17" => 100165557,
    "IGHD3-16" => 100166683,
    "IGHD2-15" => 100169006,
    "IGHD1-14" => 100171687,
    "IGHD6-13" => 100172191,
    "IGHD5-12" => 100173698,
    "IGHD4-11" => 100174645,
    "IGHD3-10" => 100175546,
    "IGHD3-9"  => 100175730,
    "IGHD2-8"  => 100178260,
    "IGHD1-7"  => 100180956,
    "IGHD6-6"  => 100181459,
    "IGHD5-5"  => 100183282,
    "IGHD4-4"  => 100184249,
    "IGHD3-3"  => 100185407,
    "IGHD2-2"  => 100187874,
    "IGHD1-1"  => 100190550
)

gene_base(name::String) = replace(name, r"\*.*$" => "")

# ============================================================================
# Model: N-only (null model - everything is spurious)
# ============================================================================

function build_null_model()
    init = [1.0]
    trans = fill(1.0, 1, 1)
    dists = [uniform_emission()]
    return HMM(init, trans, dists)
end

# ============================================================================
# Model: Single D (N → D → N)
# All D genes compete, Viterbi picks the best one
# ============================================================================

function build_single_d_model(d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}};
                               p_stay_n=0.6,
                               p_skip_d=0.05,
                               p_continue_d=0.9,
                               match_prob=0.85)
    
    d_lengths = [length(seq) for (_, seq) in d_genes]
    n_d_genes = length(d_genes)
    total_d_positions = sum(d_lengths)
    
    n_states = 2 + total_d_positions
    d_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ 2
    n_end_state = n_states
    
    trans = zeros(n_states, n_states)
    
    trans[1, 1] = p_stay_n
    trans[1, n_end_state] = p_skip_d
    # Allow entry at any position of any D gene (left trimming)
    p_per_pos = (1.0 - p_stay_n - p_skip_d) / total_d_positions
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            trans[1, start_state + pos - 1] = p_per_pos
        end
    end
    
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        d_len = length(d_seq)
        start_state = d_gene_starts[d_idx]
        for pos in 1:d_len
            current_state = start_state + pos - 1
            if pos < d_len
                trans[current_state, current_state + 1] = p_continue_d
                trans[current_state, n_end_state] = 1.0 - p_continue_d
            else
                trans[current_state, n_end_state] = 1.0
            end
        end
    end
    trans[n_end_state, n_end_state] = 1.0
    
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    dists[1] = uniform_emission()
    dists[n_end_state] = uniform_emission()
    
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    init = zeros(n_states)
    init[1] = 1.0
    
    # State info for decoding
    state_info = Vector{Tuple{Symbol, String, Int}}(undef, n_states)
    state_info[1] = (:N_start, "", 0)
    state_info[n_end_state] = (:N_end, "", 0)
    for (d_idx, (d_name, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            state_info[start_state + pos - 1] = (:D, d_name, pos)
        end
    end
    
    return HMM(init, trans, dists), state_info
end

# ============================================================================
# Model: Recurrent N↔D loop (allows 0,1,2, … D segments)
# ============================================================================

function build_full_loop_model(d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}};
                               p_stay_n::Float64=0.6,
                               p_continue_d::Float64=0.95,
                               p_direct_d_end::Float64=0.01,
                               match_prob::Float64=0.85)
    
    d_lengths = [length(seq) for (_, seq) in d_genes]
    n_d_genes = length(d_genes)
    total_d_positions = sum(d_lengths)
    
    # States: N (single) + all D positions
    n_states = 1 + total_d_positions
    n_state_N = 1
    d_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ 2
    
    trans = zeros(n_states, n_states)
    
    # N → N (stay) and N → any D position (left trimming)
    trans[n_state_N, n_state_N] = p_stay_n
    p_to_d_total = 1.0 - p_stay_n
    p_per_pos = p_to_d_total / total_d_positions
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            trans[n_state_N, start_state + pos - 1] = p_per_pos
        end
    end
    
    # D within-gene progress and exits
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        d_len = length(d_seq)
        start_state = d_gene_starts[d_idx]
        for pos in 1:d_len
            current_state = start_state + pos - 1
            if pos < d_len
                trans[current_state, current_state + 1] = p_continue_d
                trans[current_state, n_state_N] = 1.0 - p_continue_d
            else
                # At D end: exit to N with 1 - p_direct_d_end, or jump to another D (no N) with small mass
                trans[current_state, n_state_N] = 1.0 - p_direct_d_end
                if p_direct_d_end > 0
                    p_per_pos2 = p_direct_d_end / total_d_positions
                    for (d2_idx, (_, d2_seq)) in enumerate(d_genes)
                        start_state2 = d_gene_starts[d2_idx]
                        for pos2 in 1:length(d2_seq)
                            trans[current_state, start_state2 + pos2 - 1] += p_per_pos2
                        end
                    end
                end
            end
        end
    end
    
    # Emissions
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    dists[n_state_N] = uniform_emission()
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    init = zeros(n_states)
    init[n_state_N] = 1.0
    
    # State info
    state_info = Vector{Tuple{Symbol, String, Int}}(undef, n_states)
    state_info[n_state_N] = (:N, "", 0)
    for (d_idx, (d_name, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            state_info[start_state + pos - 1] = (:D, d_name, pos)
        end
    end
    
    return HMM(init, trans, dists), state_info
end

# ============================================================================
# Model: One‑D constrained loop (forbids re‑entry into D after first exit)
# ============================================================================

function build_oneD_loop_model(d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}};
                               p_stay_n::Float64=0.6,
                               p_continue_d::Float64=0.95,
                               match_prob::Float64=0.85)
    d_lengths = [length(seq) for (_, seq) in d_genes]
    total_d_positions = sum(d_lengths)
    
    # States: N0 (pre‑D), D positions, N1 (post‑D, absorbing w.r.t D entries)
    n_states = 2 + total_d_positions
    n_state_N0 = 1
    d_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ 2
    n_state_N1 = n_states
    
    trans = zeros(n_states, n_states)
    
    # N0 → N0 or enter any D position
    trans[n_state_N0, n_state_N0] = p_stay_n
    p_to_d_total = 1.0 - p_stay_n
    p_per_pos = p_to_d_total / total_d_positions
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            trans[n_state_N0, start_state + pos - 1] = p_per_pos
        end
    end
    
    # D → continue within gene or exit to N1 (no direct D→D)
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        d_len = length(d_seq)
        start_state = d_gene_starts[d_idx]
        for pos in 1:d_len
            current_state = start_state + pos - 1
            if pos < d_len
                trans[current_state, current_state + 1] = p_continue_d
                trans[current_state, n_state_N1] = 1.0 - p_continue_d
            else
                trans[current_state, n_state_N1] = 1.0
            end
        end
    end
    
    # N1 loops
    trans[n_state_N1, n_state_N1] = 1.0
    
    # Emissions
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    dists[n_state_N0] = uniform_emission()
    dists[n_state_N1] = uniform_emission()
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    init = zeros(n_states)
    init[n_state_N0] = 1.0
    
    state_info = Vector{Tuple{Symbol, String, Int}}(undef, n_states)
    state_info[n_state_N0] = (:N, "", 0)
    state_info[n_state_N1] = (:N, "", 0)
    for (d_idx, (d_name, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            state_info[start_state + pos - 1] = (:D, d_name, pos)
        end
    end
    
    return HMM(init, trans, dists), state_info
end
# ============================================================================
# Decode Viterbi path into segments
# ============================================================================

function decode_viterbi_path(best_states, state_info, cdr3_seq)
    segments = []
    current_region = nothing
    region_start = 1
    
    for (pos, state) in enumerate(best_states)
        state_type, d_name, d_pos = state_info[state]
        region_key = (state_type, d_name)
        
        if region_key != current_region
            if current_region !== nothing
                prev_type, prev_name = current_region
                push!(segments, (
                    type = prev_type,
                    name = prev_name,
                    start = region_start,
                    stop = pos - 1,
                    seq = cdr3_seq[region_start:pos-1]
                ))
            end
            current_region = region_key
            region_start = pos
        end
    end
    
    if current_region !== nothing
        prev_type, prev_name = current_region
        push!(segments, (
            type = prev_type,
            name = prev_name,
            start = region_start,
            stop = length(best_states),
            seq = cdr3_seq[region_start:length(best_states)]
        ))
    end
    
    return segments
end

#

# ============================================================================
# Main analysis
# ============================================================================

function analyze_cdr3(cdr3_seq::LongSequence{DNAAlphabet{4}},
                      d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}})
    
    obs_seq = seq_to_ints(cdr3_seq)
    
    # ========================================================================
    # QUESTION 1: Is there any D gene? (N-only vs Single-D)
    # ========================================================================
    
    hmm_null = build_null_model()
    hmm_single, state_info_single = build_single_d_model(d_genes)
    
    loglik_null = logdensityof(hmm_null, obs_seq)
    loglik_single = logdensityof(hmm_single, obs_seq)
    
    log_odds_has_d = loglik_single - loglik_null
    
    println("=" ^ 70)
    println("QUESTION 1: Is there a D gene?")
    println("=" ^ 70)
    println("Model comparison: N-only vs N→D→N")
    println()
    println("  Log P(seq | N-only):   ", round(loglik_null, digits=2))
    println("  Log P(seq | single D): ", round(loglik_single, digits=2))
    println("  Log-odds (D vs null):  ", round(log_odds_has_d, digits=2))
    println()
    
    if log_odds_has_d > 3
        println("→ STRONG evidence for D gene (log-odds > 3)")
    elseif log_odds_has_d > 1
        println("→ MODERATE evidence for D gene (log-odds > 1)")
    elseif log_odds_has_d > 0
        println("→ WEAK evidence for D gene (log-odds > 0)")
    else
        println("→ NO evidence for D gene")
    end
    
    # Show best D from single model
    best_states_single, _ = viterbi(hmm_single, obs_seq)
    segments_single = decode_viterbi_path(best_states_single, state_info_single, cdr3_seq)
    d_segments_single = filter(s -> s.type == :D, segments_single)
    
    if !isempty(d_segments_single)
        seg = d_segments_single[1]
        println()
        println("Best single D: $(seg.name) at $(seg.start)-$(seg.stop)")
        println("  Sequence: $(seg.seq)")
    end
    
    # ========================================================================
    # QUESTION 2: Is there a second D? (Model comparison)
    # ========================================================================
    
    println()
    println("=" ^ 70)
    println("QUESTION 2: Is there a second D gene?")
    println("=" ^ 70)
    println("Model comparison for recurrent N↔D loop:")
    println()
    
    # Build loop models: full (≥2 allowed) vs constrained (≤1)
    hmm_full, info_full = build_full_loop_model(d_genes; p_stay_n=0.75, p_continue_d=0.95, p_direct_d_end=0.01, match_prob=0.85)
    hmm_oneD, info_oneD = build_oneD_loop_model(d_genes; p_stay_n=0.75, p_continue_d=0.95, match_prob=0.85)
    
    loglik_full = logdensityof(hmm_full, obs_seq)
    loglik_oneD = logdensityof(hmm_oneD, obs_seq)
    
    println("  Log P(seq | full loop, ≥2 D allowed): ", round(loglik_full, digits=2))
    println("  Log P(seq | ≤1 D constrained):        ", round(loglik_oneD, digits=2))
    println()
    
    # Decision: evidence for ≥2 D segments?
    log_odds_double = loglik_full - loglik_oneD
    println("Log-odds (≥2 D vs ≤1 D): ", round(log_odds_double, digits=2))
    println()
    
    if log_odds_double > 3
        println("→ STRONG evidence for double D (log-odds > 3)")
    elseif log_odds_double > 1
        println("→ MODERATE evidence for double D (log-odds > 1)")
    elseif log_odds_double > 0
        println("→ WEAK evidence for double D (log-odds > 0)")
    else
        println("→ ≤1 real D is more likely than double D")
    end
    
    # ========================================================================
    # Show Viterbi decoding for full loop model
    # ========================================================================
    
    println()
    println("=" ^ 70)
    println("VITERBI DECODING (full loop model)")
    println("=" ^ 70)
    
    best_states, vloglik = viterbi(hmm_full, obs_seq)
    segments = decode_viterbi_path(best_states, info_full, cdr3_seq)
    
    d_positions = Int[]
    for seg in segments
        if seg.type in (:D, )
            base = gene_base(seg.name)
            pos = get(IGHD_GENOMIC_POS, base, missing)
            if pos !== missing
                push!(d_positions, pos)
            end
            pos_str = pos === missing ? "NA" : string(pos)
            println("D: $(seg.name) at $(seg.start)-$(seg.stop) (len $(length(seg.seq)))  [gpos $(pos_str)]")
            println("    $(seg.seq)")
        else
            println("$(seg.type) at $(seg.start)-$(seg.stop)")
        end
    end
    violates_order = false
    if length(d_positions) >= 2
        in_order = all(d_positions[i] < d_positions[i+1] for i in 1:length(d_positions)-1)
        violates_order = !in_order
    end
    println()
    println("violates_order: ", violates_order)
    
    return (
        log_odds_has_d = log_odds_has_d,
        log_odds_double = log_odds_double,
        best_model = log_odds_double > 0 ? "≥2 D" : "≤1 D",
        logliks = (null=loglik_null, single=loglik_single,
                   full=loglik_full, oneD=loglik_oneD)
    )
end

# ============================================================================
# Build models
# ============================================================================

d_genes = load_sequences("data/KI+1KGP-IGHD-SHORT.fasta")

cdr3_seq = dna"AATTATTGTGGTGGTGATTGCTATGCGAATGTATAGCAGTGGCTGATGC"
results = analyze_cdr3(cdr3_seq, d_genes)
