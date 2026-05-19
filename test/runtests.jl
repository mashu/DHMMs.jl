using Test
using DHMMs
using DHMMs: trim_pmf, Background, Match, Insert,
             segment_kind, pattern_idx, profile_pos
using HiddenMarkovModels: HMM, logdensityof, viterbi, forward_backward
using SparseArrays: SparseMatrixCSC, nnz

@testset "DHMMs" begin
    patterns = [[1, 2, 3], [3, 3, 1, 2]]  # ACG and GGAC

    @testset "Model construction" begin
        @test SegmentHMM(NullMode(), patterns) isa SegmentHMM{NullMode}
        @test SegmentHMM(NullMode()) isa SegmentHMM{NullMode}
        @test SegmentHMM(SingleMode(), patterns) isa SegmentHMM{SingleMode}
        @test SegmentHMM(LoopMode(), patterns) isa SegmentHMM{LoopMode}

        @test_throws ArgumentError SegmentHMM(SingleMode(), Vector{Int}[])
        @test_throws ArgumentError SegmentHMM(LoopMode(), Vector{Int}[])
        @test_throws ArgumentError SegmentHMM(SingleMode(), patterns; p_stay_n=0.99, p_skip=0.99)
        @test_throws ArgumentError SegmentHMM(LoopMode(), patterns; p_mi=0.6, p_md=0.6)
        @test_throws ArgumentError SegmentHMM(LoopMode(), patterns; p_ii=1.0)
    end

    @testset "State counts (profile = 2L per pattern)" begin
        m_null   = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        m_loop   = SegmentHMM(LoopMode(), patterns)

        @test length(m_null.states)   == 1
        @test length(m_single.states) == 2 + 2 * (3 + 4)
        @test length(m_loop.states)   == 1 + 2 * (3 + 4)
        @test count(s -> s isa Match,  m_single.states) == 3 + 4
        @test count(s -> s isa Insert, m_single.states) == 3 + 4
        @test count(s -> s isa Background, m_single.states) == 2
    end

    @testset "trim_pmf" begin
        @test trim_pmf(5, 0.0) == [1.0, 0.0, 0.0, 0.0, 0.0]
        @test sum(trim_pmf(5, 0.5)) ≈ 1.0
        @test issorted(trim_pmf(5, 0.5); rev=true)
        @test all(>(0), trim_pmf(10, 0.99))
        @test_throws ArgumentError trim_pmf(5, 1.0)
        @test_throws ArgumentError trim_pmf(5, -0.1)
        @test_throws ArgumentError trim_pmf(0, 0.5)
    end

    @testset "Profile state layout" begin
        # LoopMode, patterns = [[1,2,3], [3,3,1,2]]:
        # state 1 = Background; pattern 1 (L=3) at states 2..7 (M_1, I_1, ...,
        # M_3, I_3); pattern 2 (L=4) at states 8..15.
        m = SegmentHMM(LoopMode(), patterns)
        @test m.states[1]  == Background()
        @test m.states[2]  == Match(1, 1)
        @test m.states[3]  == Insert(1, 1)
        @test m.states[6]  == Match(1, 3)
        @test m.states[7]  == Insert(1, 3)
        @test m.states[8]  == Match(2, 1)
        @test m.states[14] == Match(2, 4)
        @test m.states[15] == Insert(2, 4)

        # Dispatched accessors
        @test segment_kind(m.states[1]) === :N
        @test segment_kind(m.states[2]) === :P
        @test segment_kind(m.states[3]) === :P
        @test pattern_idx(m.states[1])  == 0
        @test pattern_idx(m.states[8])  == 2
        @test profile_pos(m.states[1])  == 0
        @test profile_pos(m.states[14]) == 4
    end

    @testset "p_5trim = 0 concentrates entry at first match state only" begin
        m = SegmentHMM(SingleMode(), patterns; p_5trim=0.0)
        trans = m.hmm.trans
        # Pattern 1 match states are at 2, 4, 6; insert states at 3, 5, 7.
        @test trans[1, 2] > 0
        for k in 3:7
            @test trans[1, k] == 0
        end
        @test trans[1, 8] > 0  # M_{2,1}
        for k in 9:15
            @test trans[1, k] == 0
        end
    end

    @testset "p_5trim > 0 spreads entry across match states only" begin
        m = SegmentHMM(SingleMode(), patterns; p_5trim=0.5)
        trans = m.hmm.trans
        @test trans[1, 2] > trans[1, 4] > trans[1, 6] > 0    # M_{1,1..3}
        @test trans[1, 3] == 0 && trans[1, 5] == 0 && trans[1, 7] == 0  # never to I directly
    end

    @testset "Transition rows sum to 1" begin
        for m in (SegmentHMM(NullMode(), patterns),
                  SegmentHMM(SingleMode(), patterns; p_5trim=0.5),
                  SegmentHMM(LoopMode(), patterns; p_5trim=0.7),
                  SegmentHMM(LoopMode(), patterns; p_mi=0.1, p_md=0.1, p_ii=0.4, p_dd=0.4),
                  SegmentHMM(LoopMode(), patterns; p_direct=0.1, p_5trim=0.0),
                  SegmentHMM(LoopMode(), patterns; p_direct=0.2, p_5trim=0.5))
            row_sums = sum(m.hmm.trans; dims=2)
            @test all(s -> isapprox(s, 1.0; atol=1e-10), row_sums)
        end
    end

    @testset "Transition matrix is stored sparsely" begin
        for m in (SegmentHMM(NullMode(), patterns),
                  SegmentHMM(SingleMode(), patterns),
                  SegmentHMM(LoopMode(), patterns),
                  SegmentHMM(LoopMode(), patterns; p_direct=0.1))
            @test m.hmm.trans isa SparseMatrixCSC
            # At default settings, the profile topology has O(L) non-zeros per
            # state, far below the dense N² that would be n_states^2.
            @test nnz(m.hmm.trans) < length(m.hmm.trans)
        end
    end

    @testset "Sparse storage matches dense semantics" begin
        # Round-trip through full() should reproduce the same probabilities
        # and the same Viterbi result for the same observation.
        m   = SegmentHMM(LoopMode(), patterns; p_mi=0.05, p_md=0.05)
        obs = [4, 4, 1, 2, 3, 4, 4]
        ll1 = logdensityof(m, obs)
        path1, vl1 = viterbi(m, obs)
        dense_hmm = HMM(m.hmm.init, Matrix(m.hmm.trans), m.hmm.dists)
        @test isapprox(ll1, logdensityof(dense_hmm, obs); atol=1e-10)
        path2, vl2 = viterbi(dense_hmm, obs)
        @test path1 == path2
        @test isapprox(vl1, vl2; atol=1e-10)
    end

    @testset "Internal-delete transitions exist when p_md > 0" begin
        # Pattern of length 4 → M_1 should have a direct transition to M_3 and M_4
        # via the marginalised delete chain.
        pats = [[1, 2, 3, 4]]
        m = SegmentHMM(LoopMode(), pats; p_md=0.1, p_dd=0.5)
        # M_{1,1} at state 2; M_{1,2}=4, M_{1,3}=6, M_{1,4}=8
        trans = m.hmm.trans
        @test trans[2, 4] > 0   # → M_2 (match-match)
        @test trans[2, 6] > 0   # → M_3 (delete one)
        @test trans[2, 8] > 0   # → M_4 (delete two)
    end

    @testset "Delete transitions vanish when p_md = 0" begin
        pats = [[1, 2, 3, 4]]
        m = SegmentHMM(LoopMode(), pats; p_md=0.0)
        trans = m.hmm.trans
        @test trans[2, 4] > 0   # M_1 → M_2 still allowed
        @test trans[2, 6] == 0  # no delete
        @test trans[2, 8] == 0
    end

    @testset "Insert self-loop active when p_ii > 0" begin
        m = SegmentHMM(LoopMode(), patterns; p_mi=0.1, p_ii=0.5)
        trans = m.hmm.trans
        # I_{1,1} at state 3
        @test trans[3, 3] ≈ 0.5
        @test trans[3, 4] ≈ 0.5  # I → M_{1,2} (close)
        @test trans[2, 3] > 0     # M_{1,1} → I_{1,1} (open)
    end

    @testset "Log-density computation" begin
        obs = [1, 2, 3, 4, 4, 3, 2, 1]
        m_null   = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        ll_null   = logdensityof(m_null, obs)
        ll_single = logdensityof(m_single, obs)
        @test isfinite(ll_null) && ll_null < 0
        @test isfinite(ll_single) && ll_single < 0
    end

    @testset "Pattern detection (signal > noise)" begin
        obs_with_pattern = [4, 4, 1, 2, 3, 4, 4]
        m_null   = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        @test logdensityof(m_single, obs_with_pattern) > logdensityof(m_null, obs_with_pattern)
    end

    @testset "Profile detects insertion in pattern" begin
        # Pattern ACG = [1,2,3]; observe A-T-C-G (T inserted between A and C).
        pats = [[1, 2, 3]]
        m_match_only = SegmentHMM(SingleMode(), pats;
                                  p_mi=0.0, p_md=0.0, match_prob=0.99)
        m_profile = SegmentHMM(SingleMode(), pats;
                               p_mi=0.2, p_ii=0.1, p_md=0.0, match_prob=0.99)
        obs_ins = [1, 4, 2, 3]
        # The profile model with insertions should fit the inserted-T sequence
        # strictly better than the match-only model.
        @test logdensityof(m_profile, obs_ins) > logdensityof(m_match_only, obs_ins)
    end

    @testset "Profile detects deletion in pattern" begin
        # Pattern ACGT = [1,2,3,4]; observe ACT (G deleted, position 3 missing).
        pats = [[1, 2, 3, 4]]
        m_match_only = SegmentHMM(SingleMode(), pats;
                                  p_mi=0.0, p_md=0.0, match_prob=0.99)
        m_profile = SegmentHMM(SingleMode(), pats;
                               p_mi=0.0, p_md=0.2, p_dd=0.1, match_prob=0.99)
        obs_del = [1, 2, 4]
        @test logdensityof(m_profile, obs_del) > logdensityof(m_match_only, obs_del)
    end

    @testset "Decode returns segments covering observation" begin
        obs = [1, 2, 3, 4, 4, 3, 2, 1]
        m = SegmentHMM(LoopMode(), patterns)
        segs = decode(m, obs)
        @test segs isa Vector{Segment}
        @test !isempty(segs)
        @test all(s -> s.start >= 1 && s.stop <= length(obs), segs)
        @test all(s -> s.start <= s.stop, segs)
        @test segs[1].start == 1
        @test segs[end].stop == length(obs)
        @test all(s -> s.type === :N || s.type === :P, segs)
    end

    @testset "Decode separates adjacent same-pattern hits" begin
        # init forces N at position 1, so the first hit starts at position 2.
        pats = [[1, 2, 3]]
        obs = [4, 1, 2, 3, 4, 1, 2, 3]  # T ACG T ACG
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.4, p_3trim=0.01, match_prob=0.99,
                       p_mi=0.0, p_md=0.0)
        segs = decode(m, obs)
        p_segs = filter(s -> s.type === :P && s.pattern == 1, segs)
        @test length(p_segs) == 2
        @test (p_segs[1].start, p_segs[1].stop) == (2, 4)
        @test (p_segs[2].start, p_segs[2].stop) == (6, 8)
        @test all(s -> s.profile_start == 1 && s.profile_stop == 3, p_segs)
    end

    @testset "CDR3-style detection: D in the middle of a flanked read" begin
        # V-flank + D + J-flank, with D fully present (no trim).
        pats = [[1, 2, 3, 4]]                         # D = ACGT
        obs  = [4, 4, 4, 4, 1, 2, 3, 4, 1, 1, 1, 1]    # TTTT ACGT AAAA
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.85, p_3trim=0.05, match_prob=0.99,
                       p_mi=0.0, p_md=0.0)
        segs = decode(m, obs)
        @test segs[1].type === :N && segs[1].start == 1 && segs[1].stop == 4
        p_segs = filter(s -> s.type === :P, segs)
        @test length(p_segs) == 1
        @test p_segs[1].start == 5 && p_segs[1].stop == 8
        @test p_segs[1].profile_start == 1
        @test p_segs[1].profile_stop == 4
        @test segs[end].type === :N && segs[end].stop == length(obs)
    end

    @testset "CDR3-style detection: 5'-trimmed D in the middle" begin
        # True D = GACGT (length 5); expressed as ACGT (5'-G trimmed).
        pats = [[3, 1, 2, 3, 4]]                       # GACGT
        obs  = [4, 4, 4, 4, 1, 2, 3, 4, 1, 1, 1, 1]    # TTTT ACGT AAAA
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.85, p_5trim=0.75, p_3trim=0.05,
                       match_prob=0.99, p_mi=0.0, p_md=0.0)
        segs = decode(m, obs)
        p_segs = filter(s -> s.type === :P, segs)
        @test length(p_segs) == 1
        @test p_segs[1].start == 5 && p_segs[1].stop == 8
        # 5'-trim of 1 → entry at profile position 2; full traversal to position 5.
        @test p_segs[1].profile_start == 2
        @test p_segs[1].profile_stop == 5
    end

    @testset "p_direct allows direct profile→profile transitions" begin
        # With p_direct = 0, every exit from a profile goes to N. With
        # p_direct > 0, exits also feed back into pattern entries.
        m_no_direct   = SegmentHMM(LoopMode(), patterns; p_direct=0.0)
        m_with_direct = SegmentHMM(LoopMode(), patterns; p_direct=0.2)
        # M_{1,3} (state 6) terminal: exits all go to N when p_direct = 0.
        @test m_no_direct.hmm.trans[6, 8]  == 0   # no direct to M_{2,1}
        @test m_with_direct.hmm.trans[6, 8] > 0   # has direct mass into M_{2,1}
        @test m_with_direct.hmm.trans[6, 2] > 0   # direct self-restart at M_{1,1}
    end

    @testset "decode splits back-to-back patterns under p_direct" begin
        # Two ACG copies with NO background between them. Without p_direct the
        # Viterbi path must insert an N step, fragmenting the second copy.
        # With p_direct, the path can chain M_{1,3} → M_{1,1} directly.
        pats = [[1, 2, 3]]
        obs  = [4, 1, 2, 3, 1, 2, 3]  # T then ACG-ACG, no spacer
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.3, p_3trim=0.001,
                       match_prob=0.999, p_mi=0.0, p_md=0.0,
                       p_direct=0.95)
        segs = decode(m, obs)
        p_segs = filter(s -> s.type === :P, segs)
        @test length(p_segs) == 2
        @test (p_segs[1].start, p_segs[1].stop) == (2, 4)
        @test (p_segs[2].start, p_segs[2].stop) == (5, 7)
        @test all(s -> (s.profile_start, s.profile_stop) == (1, 3), p_segs)
    end

    @testset "CDR3-style detection: two D segments in one read" begin
        # V-flank + D1 + N-additions + D2 + J-flank.
        pats = [[1, 2, 3], [3, 3, 1, 2]]               # ACG, GGAC
        obs  = [4, 4,                                  # V-flank
                1, 2, 3,                               # D1 = ACG
                4, 4,                                  # N-additions
                3, 3, 1, 2,                            # D2 = GGAC
                1, 1, 1]                               # J-flank
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.6, p_3trim=0.05, match_prob=0.99,
                       p_mi=0.0, p_md=0.0)
        segs = decode(m, obs)
        p_segs = filter(s -> s.type === :P, segs)
        @test length(p_segs) == 2
        @test (p_segs[1].pattern, p_segs[1].start, p_segs[1].stop,
               p_segs[1].profile_start, p_segs[1].profile_stop) == (1, 3, 5, 1, 3)
        @test (p_segs[2].pattern, p_segs[2].start, p_segs[2].stop,
               p_segs[2].profile_start, p_segs[2].profile_stop) == (2, 8, 11, 1, 4)
    end

    @testset "forward_backward delegation" begin
        obs = [1, 2, 3, 4, 4, 3, 2, 1]
        m = SegmentHMM(LoopMode(), patterns)
        γ, _ = forward_backward(m, obs)
        @test size(γ, 1) == length(m.states)
        @test size(γ, 2) == length(obs)
        @test all(s -> isapprox(s, 1.0; atol=1e-10), sum(γ; dims=1))
    end

    @testset "Per-pattern posterior aggregation" begin
        obs = [4, 4, 1, 2, 3, 4, 4]
        m = SegmentHMM(LoopMode(), [[1, 2, 3]]; match_prob=0.99)
        γ, _ = forward_backward(m, obs)
        p_pat1 = zeros(length(obs))
        for (s, info) in enumerate(m.states)
            pattern_idx(info) == 1 || continue
            p_pat1 .+= γ[s, :]
        end
        @test p_pat1[4] > p_pat1[1]
    end

    @testset "Segment struct" begin
        s = Segment(:P, 1, 3, 5)
        @test s.type == :P
        @test s.pattern == 1
        @test s.start == 3
        @test s.stop == 5
    end

    @testset "Viterbi" begin
        obs = [1, 2, 3, 4, 4]
        m = SegmentHMM(LoopMode(), patterns)
        result = viterbi(m, obs)
        @test length(result[1]) == length(obs)
        @test result isa Tuple
    end

    @testset "Show methods" begin
        m_null   = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        m_loop   = SegmentHMM(LoopMode(), patterns)

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_null)
        @test occursin("N ⟲", String(take!(buf)))

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_single)
        @test occursin("N₀", String(take!(buf)))

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_loop)
        out = String(take!(buf))
        @test occursin("⇄", out)
        @test occursin("Patterns", out)

        for (mode_str, mm) in (("Null", m_null), ("Single", m_single), ("Loop", m_loop))
            buf = IOBuffer()
            show(buf, mm)
            @test occursin(mode_str, String(take!(buf)))
        end
    end

    @testset "Empty observations" begin
        m = SegmentHMM(LoopMode(), patterns)
        @test isempty(decode(m, Int[]))
    end

    @testset "Integer-typed observations" begin
        m = SegmentHMM(LoopMode(), patterns)
        @test decode(m, Int32[1, 2, 3, 4, 4]) isa Vector{Segment}
    end

    @testset "State elements are StateInfo subtypes" begin
        m = SegmentHMM(LoopMode(), Vector{Int}[[1, 2], [3, 4]])
        @test m.pattern_lengths == [2, 2]
        @test all(s -> s isa Union{Background, Match, Insert}, m.states)
    end
end
