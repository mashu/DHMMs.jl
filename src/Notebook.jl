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

function dna_to_int(nt::DNA)
    if nt == DNA_A
        return 1
    elseif nt == DNA_C
        return 2
    elseif nt == DNA_G
        return 3
    elseif nt == DNA_T
        return 4
    else
        error("Unknown nucleotide: $nt")
    end
end
seq_to_ints(seq::LongSequence{DNAAlphabet{4}}) = dna_to_int.(seq)

# ============================================================================
# Emission distribution for a position in D gene
# Match nucleotide gets high probability, others share the rest
# ============================================================================

function match_emission(expected_nt::DNA; match_prob=0.85)
    # Probability distribution over {A=1, C=2, G=3, T=4}
    probs = fill((1 - match_prob) / 3, 4)
    probs[dna_to_int(expected_nt)] = match_prob
    return Categorical(probs)
end

# Uniform emission for N-region
uniform_emission() = Categorical([0.25, 0.25, 0.25, 0.25])

# ============================================================================
# MODEL A: Single D gene model (N → D → N)
# 
# States:
#   1: N_start (N-region before D)
#   2 to 1+sum(lengths): D gene positions (flattened across all D genes)
#   last: N_end (N-region after D)
#
# Structure allows: N_start → any_D_start → through_D → N_end
#                   or N_start → N_end (skip D entirely)
# ============================================================================

function build_single_d_model(d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}};
                               p_stay_n=0.7,      # Probability to stay in N region
                               p_enter_d=0.25,    # Base prob to enter a D gene
                               p_continue_d=0.9,  # Prob to continue along D gene
                               match_prob=0.85)   # Emission match probability
    
    d_lengths = [length(seq) for (_, seq) in d_genes]
    n_d_genes = length(d_genes)
    total_d_positions = sum(d_lengths)
    
    # State indexing:
    # 1: N_start
    # 2 to 1+total_d_positions: D positions
    # 2+total_d_positions: N_end
    n_states = 2 + total_d_positions
    
    # Map D gene index and position to state number
    d_state_offset = 1  # N_start is state 1
    d_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ 2  # Starting state for each D gene
    
    n_end_state = n_states
    
    # Build transition matrix
    trans = zeros(n_states, n_states)
    
    # N_start transitions
    p_skip = 0.05  # Small probability to skip D entirely
    p_to_d_total = 1.0 - p_stay_n - p_skip
    p_per_d = p_to_d_total / n_d_genes
    
    trans[1, 1] = p_stay_n
    trans[1, n_end_state] = p_skip
    for (i, start_state) in enumerate(d_gene_starts)
        trans[1, start_state] = p_per_d
    end
    
    # D gene transitions (within each D gene)
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        d_len = length(d_seq)
        start_state = d_gene_starts[d_idx]
        
        for pos in 1:d_len
            current_state = start_state + pos - 1
            if pos < d_len
                # Can continue to next position or exit to N_end
                trans[current_state, current_state + 1] = p_continue_d
                trans[current_state, n_end_state] = 1.0 - p_continue_d
            else
                # Last position: must go to N_end
                trans[current_state, n_end_state] = 1.0
            end
        end
    end
    
    # N_end transitions (absorbing)
    trans[n_end_state, n_end_state] = 1.0
    
    # Build emission distributions
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    
    dists[1] = uniform_emission()  # N_start
    dists[n_end_state] = uniform_emission()  # N_end
    
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    # Initial distribution: start in N_start
    init = zeros(n_states)
    init[1] = 1.0
    
    # Build state info for decoding
    state_info = Vector{Tuple{Symbol, String, Int}}(undef, n_states)
    state_info[1] = (:N_start, "", 0)
    state_info[n_end_state] = (:N_end, "", 0)
    
    for (d_idx, (d_name, d_seq)) in enumerate(d_genes)
        start_state = d_gene_starts[d_idx]
        for pos in 1:length(d_seq)
            state_info[start_state + pos - 1] = (:D1, d_name, pos)
        end
    end
    
    return HMM(init, trans, dists), state_info
end

# ============================================================================
# MODEL B: Double D gene model (N → D₁ → N → D₂ → N)
#
# States:
#   1: N_start
#   2 to 1+total_d_pos: D1 positions (first D usage)
#   2+total_d_pos: N_middle
#   3+total_d_pos to 2+2*total_d_pos: D2 positions (second D usage)
#   3+2*total_d_pos: N_end
# ============================================================================

function build_double_d_model(d_genes::Vector{Tuple{String, LongSequence{DNAAlphabet{4}}}};
                               p_stay_n=0.7,
                               p_enter_d=0.25,
                               p_continue_d=0.9,
                               match_prob=0.85)
    
    d_lengths = [length(seq) for (_, seq) in d_genes]
    n_d_genes = length(d_genes)
    total_d_positions = sum(d_lengths)
    
    # State indexing:
    # 1: N_start
    # 2 to 1+total_d_positions: D1 positions
    # 2+total_d_positions: N_middle
    # 3+total_d_positions to 2+2*total_d_positions: D2 positions
    # 3+2*total_d_positions: N_end
    
    n_states = 3 + 2 * total_d_positions
    
    n_start = 1
    d1_start_offset = 1
    d1_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ 2
    
    n_middle = 2 + total_d_positions
    
    d2_start_offset = n_middle
    d2_gene_starts = cumsum([0; d_lengths[1:end-1]]) .+ n_middle .+ 1
    
    n_end = n_states
    
    # Build transition matrix
    trans = zeros(n_states, n_states)
    
    # N_start transitions → D1 or skip to N_middle
    p_skip_d1 = 0.1
    p_to_d1_total = 1.0 - p_stay_n - p_skip_d1
    p_per_d1 = p_to_d1_total / n_d_genes
    
    trans[n_start, n_start] = p_stay_n
    trans[n_start, n_middle] = p_skip_d1
    for start_state in d1_gene_starts
        trans[n_start, start_state] = p_per_d1
    end
    
    # D1 transitions
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
    
    # N_middle transitions → D2 or skip to N_end
    p_skip_d2 = 0.3  # Higher prob to skip second D (makes single D more likely by default)
    p_to_d2_total = 1.0 - p_stay_n - p_skip_d2
    p_per_d2 = p_to_d2_total / n_d_genes
    
    trans[n_middle, n_middle] = p_stay_n
    trans[n_middle, n_end] = p_skip_d2
    for start_state in d2_gene_starts
        trans[n_middle, start_state] = p_per_d2
    end
    
    # D2 transitions
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
    
    # N_end (absorbing)
    trans[n_end, n_end] = 1.0
    
    # Build emission distributions
    dists = Vector{Categorical{Float64, Vector{Float64}}}(undef, n_states)
    
    dists[n_start] = uniform_emission()
    dists[n_middle] = uniform_emission()
    dists[n_end] = uniform_emission()
    
    # D1 emissions
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d1_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    # D2 emissions (same as D1)
    for (d_idx, (_, d_seq)) in enumerate(d_genes)
        start_state = d2_gene_starts[d_idx]
        for (pos, nt) in enumerate(d_seq)
            dists[start_state + pos - 1] = match_emission(nt; match_prob)
        end
    end
    
    # Initial distribution
    init = zeros(n_states)
    init[n_start] = 1.0
    
    # State info for decoding
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
    
    # Store d2_gene_starts for probability calculation
    d2_state_range = (n_middle + 1):(n_end - 1)
    
    return HMM(init, trans, dists), state_info, d2_state_range
end

# ============================================================================
# Build models
# ============================================================================

d_genes = load_sequences("data/KI+1KGP-IGHD-SHORT.fasta")

hmm_single, state_info_single = build_single_d_model(d_genes)
hmm_double, state_info_double, d2_states = build_double_d_model(d_genes)

# cdr3_seq = dna"GCGAGAGATAATCGCTATTACGATTTTTGGAGTGGTTATTCTTCGGGTTACTACTACTACTACTACTACATGGACGTC"
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

#              GTATTACGATTTTTGGAGTGGTTATTATACC # Full D3-3*01
#              CTATTACGATTTTTGGAGTGGTTATT      # Partial D3-3*01 from viterbi
#GCGAGAGATAATCGCTATTACGATTTTTGGAGTGGTTATTCTTCGGGTTACTACTACTACTACTACTACATGGACGTC
#             GTATTACTATGATAGT AGTGGTTATT ACTAC # IGHD3-22*01
#
# YYDFWSGY,SSGYYY
