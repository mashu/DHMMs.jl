using Test
using DHMMs
using HiddenMarkovModels: logdensityof

@testset "DHMMs" begin
    # Simple patterns (DNA-like: 1=A, 2=C, 3=G, 4=T)
    patterns = [[1, 2, 3], [3, 3, 1, 2]]  # ACG and GGAC

    @testset "Model construction" begin
        @test SegmentHMM(NullMode(), patterns) isa SegmentHMM{NullMode}
        @test SegmentHMM(SingleMode(), patterns) isa SegmentHMM{SingleMode}
        @test SegmentHMM(LoopMode(), patterns) isa SegmentHMM{LoopMode}
    end

    @testset "State counts" begin
        m_null = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        m_loop = SegmentHMM(LoopMode(), patterns)

        @test length(m_null.states) == 1
        @test length(m_single.states) == 2 + 3 + 4  # N_start + patterns + N_end
        @test length(m_loop.states) == 1 + 3 + 4    # N + patterns
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
        # Segments should cover entire sequence
        @test segs[1].start == 1
        @test segs[end].stop == length(obs)
    end

    @testset "Pattern detection" begin
        # Sequence containing pattern 1 (ACG = [1,2,3])
        obs_with_pattern = [4, 4, 1, 2, 3, 4, 4]  # TT ACG TT
        
        m_null = SegmentHMM(NullMode(), patterns)
        m_single = SegmentHMM(SingleMode(), patterns)
        
        ll_null = logdensityof(m_null, obs_with_pattern)
        ll_single = logdensityof(m_single, obs_with_pattern)
        
        # Model with pattern capability should fit better
        @test ll_single > ll_null
    end

    @testset "Segment struct" begin
        s = Segment(:P, 1, 3, 5)
        @test s.type == :P
        @test s.pattern == 1
        @test s.start == 3
        @test s.stop == 5
    end

    @testset "Show methods" begin
        m = SegmentHMM(LoopMode(), patterns)
        
        # Compact show
        buf = IOBuffer()
        show(buf, m)
        @test occursin("SegmentHMM", String(take!(buf)))
        
        # Full show
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        output = String(take!(buf))
        @test occursin("patterns", output)
        @test occursin("states", output)
    end

    @testset "Empty observations" begin
        m = SegmentHMM(LoopMode(), patterns)
        segs = decode(m, Int[])
        @test isempty(segs)
    end

    @testset "Type stability" begin
        patterns_typed = Vector{Int}[[1, 2], [3, 4]]
        m = SegmentHMM(LoopMode(), patterns_typed)
        @test m.pattern_lengths == [2, 2]
        @test all(s -> s isa Tuple{Symbol, Int, Int}, m.states)
    end
end

