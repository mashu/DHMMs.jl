using Test
using DHMMs
using DHMMs: trim_pmf
using HiddenMarkovModels: logdensityof, viterbi, forward_backward

@testset "DHMMs" begin
    # Simple patterns (DNA-like: 1=A, 2=C, 3=G, 4=T)
    patterns = [[1, 2, 3], [3, 3, 1, 2]]  # ACG and GGAC

    @testset "Model construction" begin
        @test SegmentHMM(NullMode(), patterns) isa SegmentHMM{NullMode}
        @test SegmentHMM(NullMode()) isa SegmentHMM{NullMode}  # patterns optional
        @test SegmentHMM(SingleMode(), patterns) isa SegmentHMM{SingleMode}
        @test SegmentHMM(LoopMode(), patterns) isa SegmentHMM{LoopMode}

        @test_throws ArgumentError SegmentHMM(SingleMode(), Vector{Int}[])
        @test_throws ArgumentError SegmentHMM(LoopMode(), Vector{Int}[])
        @test_throws ArgumentError SegmentHMM(SingleMode(), patterns; p_stay_n=0.99, p_skip=0.99)
    end

    @testset "State counts" begin
        m_null = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        m_loop = SegmentHMM(LoopMode(), patterns)

        @test length(m_null.states) == 1
        @test length(m_single.states) == 2 + 3 + 4
        @test length(m_loop.states) == 1 + 3 + 4
    end

    @testset "trim_pmf" begin
        @test trim_pmf(5, 0.0) == [1.0, 0.0, 0.0, 0.0, 0.0]
        @test sum(trim_pmf(5, 0.5)) ≈ 1.0
        @test issorted(trim_pmf(5, 0.5); rev=true)  # decaying with position
        pmf_high = trim_pmf(10, 0.99)
        @test all(>(0), pmf_high)  # all positions reachable when p_trim ~ 1
        @test_throws ArgumentError trim_pmf(5, 1.0)
        @test_throws ArgumentError trim_pmf(5, -0.1)
        @test_throws ArgumentError trim_pmf(0, 0.5)
    end

    @testset "p_5trim = 0 concentrates entry at position 1" begin
        m = SegmentHMM(SingleMode(), patterns; p_5trim=0.0)
        # State 1 is N; states 2.. are pattern positions in order.
        # Entry from N to pattern positions must be zero except for the first
        # state of each pattern.
        trans = m.hmm.trans
        @test trans[1, 2] > 0      # start of pattern 1
        @test trans[1, 3] == 0     # mid pattern 1
        @test trans[1, 4] == 0     # mid pattern 1
        @test trans[1, 5] > 0      # start of pattern 2
        @test trans[1, 6] == 0
        @test trans[1, 7] == 0
        @test trans[1, 8] == 0
    end

    @testset "p_5trim > 0 spreads entry mass" begin
        m = SegmentHMM(SingleMode(), patterns; p_5trim=0.5)
        trans = m.hmm.trans
        @test trans[1, 2] > trans[1, 3] > trans[1, 4] > 0
    end

    @testset "Transition rows sum to 1" begin
        for m in (SegmentHMM(NullMode(), patterns),
                  SegmentHMM(SingleMode(), patterns; p_5trim=0.5),
                  SegmentHMM(LoopMode(), patterns; p_5trim=0.7))
            row_sums = sum(m.hmm.trans; dims=2)
            @test all(s -> isapprox(s, 1.0; atol=1e-12), row_sums)
        end
    end

    @testset "Log-density computation" begin
        obs = [1, 2, 3, 4, 4, 3, 2, 1]

        m_null = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)

        ll_null = logdensityof(m_null, obs)
        ll_single = logdensityof(m_single, obs)

        @test isfinite(ll_null)
        @test isfinite(ll_single)
        @test ll_null < 0
        @test ll_single < 0
    end

    @testset "Decode returns segments" begin
        obs = [1, 2, 3, 4, 4, 3, 2, 1]
        m = SegmentHMM(LoopMode(), patterns)

        segs = decode(m, obs)

        @test segs isa Vector{Segment}
        @test !isempty(segs)
        @test all(s -> s.start >= 1 && s.stop <= length(obs), segs)
        @test all(s -> s.start <= s.stop, segs)
        @test segs[1].start == 1
        @test segs[end].stop == length(obs)
    end

    @testset "Decode separates adjacent same-pattern hits" begin
        # Two clean copies of pattern 1 (ACG = [1,2,3]) separated by a single
        # background nt: the model in LoopMode should recover two distinct
        # pattern-1 segments, not merge them.
        pats = [[1, 2, 3]]
        obs = [1, 2, 3, 4, 1, 2, 3]  # ACG T ACG
        m = SegmentHMM(LoopMode(), pats;
                       p_stay_n=0.4, p_3trim=0.01, match_prob=0.99)
        segs = decode(m, obs)
        p_segs = filter(s -> s.type === :P && s.pattern == 1, segs)
        @test length(p_segs) == 2
        @test p_segs[1].start == 1 && p_segs[1].stop == 3
        @test p_segs[2].start == 5 && p_segs[2].stop == 7
    end

    @testset "Pattern detection (signal > noise)" begin
        obs_with_pattern = [4, 4, 1, 2, 3, 4, 4]

        m_null = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)

        ll_null = logdensityof(m_null, obs_with_pattern)
        ll_single = logdensityof(m_single, obs_with_pattern)

        @test ll_single > ll_null
    end

    @testset "forward_backward delegation" begin
        obs = [1, 2, 3, 4, 4, 3, 2, 1]
        m = SegmentHMM(LoopMode(), patterns)
        γ, logL = forward_backward(m, obs)
        @test size(γ, 1) == length(m.states)
        @test size(γ, 2) == length(obs)
        col_sums = sum(γ; dims=1)
        @test all(s -> isapprox(s, 1.0; atol=1e-10), col_sums)
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
        m_null = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        m_loop = SegmentHMM(LoopMode(), patterns)

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_null)
        @test occursin("N ⟲", String(take!(buf)))

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_single)
        @test occursin("N₀", String(take!(buf)))

        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m_loop)
        output = String(take!(buf))
        @test occursin("⇄", output)
        @test occursin("Patterns", output)

        buf = IOBuffer()
        show(buf, m_null)
        @test occursin("Null", String(take!(buf)))

        buf = IOBuffer()
        show(buf, m_single)
        @test occursin("Single", String(take!(buf)))

        buf = IOBuffer()
        show(buf, m_loop)
        @test occursin("Loop", String(take!(buf)))
    end

    @testset "Empty observations" begin
        m = SegmentHMM(LoopMode(), patterns)
        segs = decode(m, Int[])
        @test isempty(segs)
    end

    @testset "Integer-typed observations" begin
        m = SegmentHMM(LoopMode(), patterns)
        obs32 = Int32[1, 2, 3, 4, 4]
        @test decode(m, obs32) isa Vector{Segment}
    end

    @testset "Type stability of state metadata" begin
        patterns_typed = Vector{Int}[[1, 2], [3, 4]]
        m = SegmentHMM(LoopMode(), patterns_typed)
        @test m.pattern_lengths == [2, 2]
        @test all(s -> s isa Tuple{Symbol, Int, Int}, m.states)
    end
end
