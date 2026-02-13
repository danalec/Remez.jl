using Remez
using Test

@testset "Minimax and Least Squares" begin
    @testset "Leastsquares test" begin
        n = 1000
        a = [i/BigFloat(n) for i in 0:n]
        (nc, dc) = ratfn_leastsquares(exp, a, 2, 2)

        for x in a
            @test isapprox(exp(x), Remez.ratfn_eval(nc, dc, x); rtol=0, atol=1e-4)
        end
    end

    @testset "Minimax polynomial" begin
        (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 4, 0)
        @test 0 < e < 1e-3
        for x = big"0.0":big"0.01":big"1.0"
            @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001
        end
    end

    @testset "Minimax rational" begin
        (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 2)
        @test 0 < e < 1e-3
        for x = big"0.0":big"0.01":big"1.0"
            @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001
        end
    end

    @testset "Minimax weighted" begin
        @testset "Polynomial weighted" begin
            (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 4, 0, (x,y)->1/y)
            @test 0 < e < 1e-3
            for x = big"0.0":big"0.01":big"1.0"
                @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
            end
        end

        @testset "Rational weighted" begin
            (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 2, (x,y)->1/y)
            @test 0 < e < 1e-3
            for x = big"0.0":big"0.01":big"1.0"
                @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
            end
        end
        
        @testset "Rational weighted odd degree" begin
            (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 1, (x,y)->1/y)
            @test 0 < e < 1e-3
            for x = big"0.0":big"0.01":big"1.0"
                @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
            end
        end
    end
end
