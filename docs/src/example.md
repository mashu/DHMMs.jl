# Example: Detecting D Segments in CDR3

This example shows how to detect IGHD gene segments in CDR3 sequences using DHMMs.

## Setup

```julia
using DHMMs
using BioSequences
using FASTX
```

## Helper Functions

Convert DNA to integer encoding (A=1, C=2, G=3, T=4):

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

## Load D Genes

```julia
function load_d_genes(fasta_path::String)
    genes = Vector{Int}[]
    names = String[]
    for record in FASTAReader(open(fasta_path))
        push!(names, description(record))
        push!(genes, seq_to_ints(sequence(LongDNA{4}, record)))
    end
    names, genes
end

# Example: load from file
# names, patterns = load_d_genes("IGHD.fasta")

# Or define manually:
d_genes = [
    dna"GGTATAGTGGGAGCTAC",      # IGHD1-1
    dna"AGGATATTGTAGTAGTAC",     # IGHD2-2  
    dna"GTATTACTATGGTTCGGGGAGTTATTATAAC",  # IGHD3-3
]
patterns = [seq_to_ints(g) for g in d_genes]
```

## Build Models

```julia
# Null model: no D segment (background only)
m_null = SegmentHMM(NullMode(), patterns)

# Single model: at most one D segment
m_single = SegmentHMM(SingleMode(), patterns; 
    p_stay_n=0.6, p_continue=0.9, match_prob=0.85)

# Loop model: multiple D segments allowed
m_loop = SegmentHMM(LoopMode(), patterns;
    p_stay_n=0.75, p_continue=0.95, p_direct=0.01, match_prob=0.85)
```

## Analyze a CDR3 Sequence

```julia
# Example CDR3 with potential double-D
cdr3 = dna"AATTATTGTGGTGGTGATTGCTATGCGAATGTATAGCAGTGGCTGATGC"
obs = seq_to_ints(cdr3)

# Compare model likelihoods
ll_null = logdensityof(m_null, obs)
ll_single = logdensityof(m_single, obs)
ll_loop = logdensityof(m_loop, obs)

# Log-odds for D presence
log_odds_has_d = ll_single - ll_null
println("Evidence for D: ", log_odds_has_d)

# Log-odds for multiple Ds
log_odds_multi = ll_loop - ll_single  
println("Evidence for multiple D: ", log_odds_multi)
```

## Decode Segments

```julia
segments = decode(m_loop, obs)

for seg in segments
    region = String(cdr3[seg.start:seg.stop])
    if seg.type == :P
        println("D$(seg.pattern): $region ($(seg.start)-$(seg.stop))")
    else
        println("N: $region ($(seg.start)-$(seg.stop))")
    end
end
```

## Decision Thresholds

```julia
function analyze_cdr3(cdr3::LongDNA{4}, patterns; threshold=1.0)
    obs = seq_to_ints(cdr3)
    
    m_null = SegmentHMM(NullMode(), patterns)
    m_single = SegmentHMM(SingleMode(), patterns)
    m_loop = SegmentHMM(LoopMode(), patterns)
    
    ll_null = logdensityof(m_null, obs)
    ll_single = logdensityof(m_single, obs)
    ll_loop = logdensityof(m_loop, obs)
    
    has_d = (ll_single - ll_null) > threshold
    has_multi_d = (ll_loop - ll_single) > threshold
    
    segments = decode(m_loop, obs)
    d_count = count(s -> s.type == :P, segments)
    
    (has_d=has_d, has_multi_d=has_multi_d, d_count=d_count, segments=segments)
end

result = analyze_cdr3(cdr3, patterns)
println("Has D: ", result.has_d)
println("Has multiple D: ", result.has_multi_d)
println("D segments found: ", result.d_count)
```

## Negative Example: Random Sequence

Test with a random sequence that should NOT match any D gene:

```julia
# Random sequence with no D gene content
random_cdr3 = dna"AAAAAACCCCCCGGGGGGTTTTTT"
obs_random = seq_to_ints(random_cdr3)

ll_null = logdensityof(m_null, obs_random)
ll_single = logdensityof(m_single, obs_random)
ll_loop = logdensityof(m_loop, obs_random)

log_odds_d = ll_single - ll_null
println("Random seq - Evidence for D: ", log_odds_d)  # Should be negative or near zero

segments = decode(m_loop, obs_random)
d_segments = filter(s -> s.type == :P, segments)
println("D segments found: ", length(d_segments))  # Should be 0 or spurious short matches
```

Expected output: negative log-odds indicating no D gene evidence.

## Positive vs Negative Comparison

```julia
# Positive: CDR3 containing actual D gene sequence
cdr3_with_d = dna"TTTTGGTATAGTGGGAGCTACAAAA"  # Contains IGHD1-1
obs_pos = seq_to_ints(cdr3_with_d)

# Negative: same length, no D gene
cdr3_without_d = dna"TTTTAAACCCGGGTTTAAACCCCCC"
obs_neg = seq_to_ints(cdr3_without_d)

# Compare
for (name, obs) in [("With D", obs_pos), ("Without D", obs_neg)]
    ll_null = logdensityof(m_null, obs)
    ll_single = logdensityof(m_single, obs)
    log_odds = ll_single - ll_null
    println("$name: log-odds = $(round(log_odds, digits=2))")
end
# Expected: "With D" has positive log-odds, "Without D" has negative/zero
```

## Batch Processing

```julia
function process_cdr3s(cdr3_seqs::Vector{LongDNA{4}}, patterns)
    # Build models once
    m_null = SegmentHMM(NullMode(), patterns)
    m_single = SegmentHMM(SingleMode(), patterns)
    m_loop = SegmentHMM(LoopMode(), patterns)
    
    results = map(cdr3_seqs) do cdr3
        obs = seq_to_ints(cdr3)
        
        ll_null = logdensityof(m_null, obs)
        ll_single = logdensityof(m_single, obs)
        ll_loop = logdensityof(m_loop, obs)
        
        (
            cdr3 = cdr3,
            log_odds_d = ll_single - ll_null,
            log_odds_multi = ll_loop - ll_single,
            segments = decode(m_loop, obs)
        )
    end
    results
end
```

