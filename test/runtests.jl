using Test
using DHMMs
using DHMMs: trim_pmf
using HiddenMarkovModels: logdensityof, viterbi, forward_backward

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
        @test length(m_single.states) == 2 + 2 * (3 + 4)   # 2 N + 2L per pattern
        @test length(m_loop.states)   == 1 + 2 * (3 + 4)
        @test count(s -> s[1] === :M, m_single.states) == 3 + 4
        @test count(s -> s[1] === :I, m_single.states) == 3 + 4
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
        # LoopMode, patterns = [[1,2,3], [3,3,1,2]]
        # Layout: state 1 = N, then for pattern 1 (L=3):
        #   2 M_{1,1}, 3 I_{1,1}, 4 M_{1,2}, 5 I_{1,2}, 6 M_{1,3}, 7 I_{1,3}
        # Then for pattern 2 (L=4):
        #   8 M_{2,1}, 9 I_{2,1}, ..., 14 M_{2,4}, 15 I_{2,4}
        m = SegmentHMM(LoopMode(), patterns)
        @test m.states[1] == (:N, 0, 0)
        @test m.states[2] == (:M, 1, 1)
        @test m.states[3] == (:I, 1, 1)
        @test m.states[6] == (:M, 1, 3)
        @test m.states[7] == (:I, 1, 3)
        @test m.states[8] == (:M, 2, 1)
        @test m.states[14] == (:M, 2, 4)
        @test m.states[15] == (:I, 2, 4)
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
                  SegmentHMM(LoopMode(), patterns; p_mi=0.1, p_md=0.1, p_ii=0.4, p_dd=0.4))
            row_sums = sum(m.hmm.trans; dims=2)
            @test all(s -> isapprox(s, 1.0; atol=1e-10), row_sums)
        end
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
        pats = [[1, 2, 3]]
        obs = [1, 2, 3, 4, 1, 2, 3]  # ACG T ACG
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.4, p_3trim=0.01, match_prob=0.99,
                       p_mi=0.0, p_md=0.0)
        segs = decode(m, obs)
        p_segs = filter(s -> s.type === :P && s.pattern == 1, segs)
        @test length(p_segs) == 2
        @test p_segs[1].start == 1 && p_segs[1].stop == 3
        @test p_segs[2].start == 5 && p_segs[2].stop == 7
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
        obs = [4, 4, 1, 2, 3, 4, 4]  # background then ACG then background
        m = SegmentHMM(LoopMode(), [[1, 2, 3]]; match_prob=0.99)
        γ, _ = forward_backward(m, obs)
        # Build per-pattern occupancy by summing M and I rows for pattern 1.
        p_pat1 = zeros(length(obs))
        for (s, info) in enumerate(m.states)
            (info[1] === :M || info[1] === :I) && info[2] == 1 || continue
            p_pat1 .+= γ[s, :]
        end
        # Middle of the ACG hit (position 4) should have higher pattern-1
        # occupancy than position 1 (pure background).
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

    @testset "Type stability of state metadata" begin
        m = SegmentHMM(LoopMode(), Vector{Int}[[1, 2], [3, 4]])
        @test m.pattern_lengths == [2, 2]
        @test all(s -> s isa Tuple{Symbol, Int, Int}, m.states)
    end
end
