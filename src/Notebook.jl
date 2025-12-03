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
# MODEL A: Single D gene model (N → D → N)
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
    
    # N_start: stay, skip to N_end, or enter any D
    p_to_d_total = 1.0 - p_stay_n - p_skip_d
    p_per_d = p_to_d_total / n_d_genes
    
    trans[1, 1] = p_stay_n
    trans[1, n_end_state] = p_skip_d
    for start_state in d_gene_starts
        trans[1, start_state] = p_per_d
    end
    
    # D gene transitions
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
    
    # Emissions
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
    
    # State info
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
# MODEL B: Double D gene model (N → D₁ → N → D₂ → N) - FIXED
# ============================================================================

function build_double_d_model(d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}};
                               p_stay_n=0.6,
                               p_continue_d=0.9,
                               match_prob=0.85)
    
    d_lengths = [length(seq) for (_, seq) in d_genes]
    n_d_genes = length(d_genes)
    total_d_positions = sum(d_lengths)
    
    n_states = 3 + 2 * total_d_positions
    
    n_start = 1
    d1_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ 2
    n_middle = 2 + total_d_positions
    d2_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ n_middle .+ 1
    n_end = n_states
    
    trans = zeros(n_states, n_states)
    
    # N_start → D1 (no skip allowed - must use at least one D in this model)
    p_to_d1_total = 1.0 - p_stay_n
    p_per_d1 = p_to_d1_total / n_d_genes
    
    trans[n_start, n_start] = p_stay_n
    for start_state in d1_gene_starts
        trans[n_start, start_state] = p_per_d1
    end
    
    # D1 → N_middle
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        d_len = length(d_seq)
        start_state = d1_gene_starts[d_idx]
        
        for pos in 1:d_len
            current_state = start_state + pos - 1
            if pos < d_len
                trans[current_state, current_state + 1] = p_continue_d
                trans[current_state, n_middle] = 1.0 - p_continue_d
            else
                trans[current_state, n_middle] = 1.0
            end
        end
    end
    
    # N_middle → D2 or N_end (THIS WAS THE BUG!)
    # Fixed: allocate probability properly
    p_skip_d2 = 0.1   # Small prob to skip D2
    p_to_d2_total = 1.0 - p_stay_n - p_skip_d2  # = 0.3 (not 0!)
    p_per_d2 = p_to_d2_total / n_d_genes
    
    trans[n_middle, n_middle] = p_stay_n
    trans[n_middle, n_end] = p_skip_d2
    for start_state in d2_gene_starts
        trans[n_middle, start_state] = p_per_d2
    end
    
    # D2 → N_end
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        d_len = length(d_seq)
        start_state = d2_gene_starts[d_idx]
        
        for pos in 1:d_len
            current_state = start_state + pos - 1
            if pos < d_len
                trans[current_state, current_state + 1] = p_continue_d
                trans[current_state, n_end] = 1.0 - p_continue_d
            else
                trans[current_state, n_end] = 1.0
            end
        end
    end
    
    trans[n_end, n_end] = 1.0
    
    # Emissions
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    dists[n_start] = uniform_emission()
    dists[n_middle] = uniform_emission()
    dists[n_end] = uniform_emission()
    
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d1_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d2_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    init = zeros(n_states)
    init[n_start] = 1.0
    
    # State info
    state_info = Vector{Tuple{Symbol, String, Int}}(undef, n_states)
    state_info[n_start] = (:N_start, "", 0)
    state_info[n_middle] = (:N_middle, "", 0)
    state_info[n_end] = (:N_end, "", 0)
    
    for (d_idx, (d_name, d_seq)) in enumerate(d_genes)
        start_state = d1_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            state_info[start_state + pos - 1] = (:D1, d_name, pos)
        end
    end
    
    for (d_idx, (d_name, d_seq)) in enumerate(d_genes)
        start_state = d2_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            state_info[start_state + pos - 1] = (:D2, d_name, pos)
        end
    end
    
    d2_state_range = (n_middle + 1):(n_end - 1)
    
    return HMM(init, trans, dists), state_info, d2_state_range
end

# ============================================================================
# Build models
# ============================================================================

d_genes = load_sequences("data/KI+1KGP-IGHD-SHORT.fasta")

hmm_single, state_info_single = build_single_d_model(d_genes)
hmm_double, state_info_double, d2_states = build_double_d_model(d_genes)

cdr3_seq = dna"GCGAGAGATAATCGCTATTACGATTTTTGGAGTGGTTATTCTTCGGGTTACTACTACTACTACTACTACATGGACGTC"
cdr3_seq = dna"AATTATTGTGGTGGTGATTGCTATGCGAATGTATAGCAGTGGCTGATGC"
obs_seq = seq_to_ints(cdr3_seq)


# ============================================================================
# QUESTION 1: Does CDR3 contain more than 1 D? (Model comparison / Forward-Backward)
# ============================================================================

# Method A: Log-odds ratio between models
loglik_single = logdensityof(hmm_single, obs_seq)
loglik_double = logdensityof(hmm_double, obs_seq)
log_odds_double_vs_single = loglik_double - loglik_single

println("=" ^ 70)
println("QUESTION 1: Double D detection")
println("=" ^ 70)
println("Log-likelihood (single D model): ", loglik_single)
println("Log-likelihood (double D model): ", loglik_double)
println("Log-odds (double / single):      ", log_odds_double_vs_single)
println()

# Method B: Using forward-backward on double model to get P(visited D2 states)
γ, _ = forward_backward(hmm_double, obs_seq)

# Sum posterior probability of being in D2 states at any time point
# If this is high, sequence likely used a second D
prob_in_d2_by_time = vec(sum(γ[d2_states, :], dims=1))
max_prob_d2 = maximum(prob_in_d2_by_time)
total_d2_occupancy = sum(prob_in_d2_by_time)

println("Forward-Backward analysis (double D model):")
println("Maximum P(in D2 state) at any position: ", max_prob_d2)
println("Total D2 state occupancy (expected #positions in D2): ", total_d2_occupancy)
println()

# Decision
if log_odds_double_vs_single > 0
    println("→ CONCLUSION: Double D is MORE LIKELY (log-odds = $log_odds_double_vs_single)")
else
    println("→ CONCLUSION: Single D is MORE LIKELY (log-odds = $log_odds_double_vs_single)")
end
println()

# ============================================================================
# QUESTION 2: What are the Ds in the CDR3? (Viterbi decoding)
# ============================================================================

println("=" ^ 70)
println("QUESTION 2: Identify D genes (Viterbi)")
println("=" ^ 70)

# Use double-D model for Viterbi (can find 0, 1, or 2 D genes)
best_states, loglik = viterbi(hmm_double, obs_seq)

println("Viterbi path log-likelihood: ", only(loglik))
println()

# Decode the path
println("Position-by-position decoding:")
println("-" ^ 50)

current_region = nothing
region_start = 1
d_genes_found = []

for (pos, state) in enumerate(best_states)
    state_type, d_name, d_pos = state_info_double[state]
    
    region_key = (state_type, d_name)
    
    if region_key != current_region
        if current_region !== nothing
            # Print previous region
            prev_type, prev_name = current_region
            region_end = pos - 1
            if prev_type == :D1 || prev_type == :D2
                push!(d_genes_found, (prev_type, prev_name, region_start, region_end))
                println("CDR3[$region_start:$region_end] → $prev_type: $prev_name")
            else
                println("CDR3[$region_start:$region_end] → $prev_type (N-region)")
            end
        end
        current_region = region_key
        region_start = pos
    end
end

# Print last region
if current_region !== nothing
    prev_type, prev_name = current_region
    region_end = length(best_states)
    if prev_type == :D1 || prev_type == :D2
        push!(d_genes_found, (prev_type, prev_name, region_start, region_end))
        println("CDR3[$region_start:$region_end] → $prev_type: $prev_name")
    else
        println("CDR3[$region_start:$region_end] → $prev_type (N-region)")
    end
end

println()
println("=" ^ 50)
println("SUMMARY: D genes identified")
println("=" ^ 50)
if isempty(d_genes_found)
    println("No D genes found in the sequence")
else
    for (dtype, dname, start, stop) in d_genes_found
        segment = cdr3_seq[start:stop]
        println("$dtype: $dname at positions $start-$stop")
        println("   Sequence: $segment")
    end
end
println()
println("Number of D segments found: ", length(d_genes_found))

