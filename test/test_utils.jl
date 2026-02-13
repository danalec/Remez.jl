using Remez
using Test

@testset "Utility Functions" begin
    @testset "Gaussian elimination" begin
        m = BigFloat[1 1 2; 3 5 8; 13 34 21]
        v = BigFloat[1, -1, 2]
        ret = m \ v

        @test ret[1] ≈ big(109)/big(26)
        @test ret[2] ≈ big(-105)/big(130)
        @test ret[3] ≈ big(-31)/big(26)
    end

    @testset "Golden section search" begin
        x, y = Remez.goldensection(sin, big"0.0", big"0.1", big"4.0")
        @test isapprox(x, asin(BigFloat(1)))
        @test isapprox(y, 1)
    end

    @testset "Extrema-winnowing algorithm" begin
        extrema = [(x, sin(20*x)*sin(197*x))  for x in big"0.0":big"0.001":big"1.0"]
        winnowed = Remez.winnow_extrema(extrema, 12)
        prevx, prevy = -1.0, 0.0
        for (x,y) in winnowed
            @test x > prevx
            @test y != 0
            @test prevy * y <= 0 # tolerates initial prevx having no sign
            @test abs(y) > 0.9
            prevx, prevy = x, y
        end
    end
    
    @testset "Poly Eval Type Stability" begin
        coeffs = [1.0, 2.0, 3.0]
        @test typeof(Remez.poly_eval(coeffs, 0.5)) == Float64
        
        coeffs_big = [big(1.0), big(2.0), big(3.0)]
        @test typeof(Remez.poly_eval(coeffs_big, big(0.5))) == BigFloat
    end
end
