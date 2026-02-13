# Remez.jl - Optimized & Generic Minimax Approximation

This package provides a high-performance, numerically stable implementation of the [Remez algorithm](https://en.wikipedia.org/wiki/Remez_algorithm) for computing minimax polynomial and rational approximations to functions.

## Key Features

- **Numerical Stability**: Uses an internal **Chebyshev basis** for Remez iterations, drastically improving matrix conditioning for high-degree approximations.
- **Type Genericity**: Supports any `AbstractFloat` type (`Float64`, `BigFloat`, `Double64`, etc.). Precision is automatically determined by the input interval's type.
- **Performance**:
    - `RemezWorkspace`: Pre-allocates buffers for linear solvers and basis transformations, drastically reducing GC pressure in iterative loops.
    - Optimized evaluation using `evalpoly` and in-place basis conversions (`cheb_to_power_inplace!`).
    - Efficient extrema finding with heuristic grid sizing and dynamic programming winnowing.
- **Robustness**: 
    - Safe iteration limits to prevent infinite loops.
    - Robust extrema finding with high-density grid evaluation and golden-section search.
    - Graceful fallback: Returns the best found approximation if perfect equalization isn't reached.
- **Public API**:
    - `ratfn_minimax`: standard minimax (returns power basis).
    - `ratfn_minimax_cheb`: direct Chebyshev basis results.
    - `ratfn_leastsquares`: rational least-squares initialization.

## Installation

```julia
using Pkg
Pkg.add("Remez")
```

## Basic Usage

### Polynomial Approximation

Approximate `exp(x)` on `[0, 1]` with a degree-3 polynomial:

```julia
using Remez
# (numerator_coeffs, denominator_coeffs, max_error, alternation_points)
(nc, dc, e, x) = ratfn_minimax(exp, (0.0, 1.0), 3, 0)
```

### Rational Approximation

Approximate `exp(x)` with a (2, 2) rational function:

```julia
(nc, dc, e, x) = ratfn_minimax(exp, (0.0, 1.0), 2, 2)
```

### High-Precision with BigFloat

Simply use `big` values for the interval:

```julia
(nc, dc, e, x) = ratfn_minimax(exp, (big(0.0), big(1.0)), 10, 5)
```

## Advanced Usage

### Direct Chebyshev Coefficients

To avoid power-basis instability in high-degree approximations, use the direct Chebyshev API:

```julia
(nc_cheb, dc_cheb, e, x) = ratfn_minimax_cheb(exp, (0.0, 1.0), 5, 5)
```

### Using Workspaces

For repeated calls or performance-critical loops, you can use `RemezWorkspace`:

```julia
ws = RemezWorkspace{Float64}((0.0, 1.0), 3, 3)
(nc, dc, e, x) = Remez.ratfn_minimax!(ws, exp)
```

## Testing

The test suite is modularized into several focused files:
- `test_utils.jl`: Internal utility functions and math helpers.
- `test_minimax.jl`: Core minimax and least-squares verification.
- `test_genericity.jl`: Verification of type stability and precision across different numeric types.

To run all tests:
```bash
julia --project=. test/runtests.jl
```

## Benchmarking

The project include benchmark scripts to measure performance and profile bottlenecks:

- **[benchmark2.jl](benchmark2.jl)**: Standard benchmarking of core Remez functions with `BenchmarkTools`.
- **[benchmark.jl](benchmark.jl)**: Comprehensive benchmarking and terminal-optimized profiling using `Profile`.
- **[profile_remez.jl](profile_remez.jl)**: Specific script for profiling the minimax loop.

To run the standard benchmark:
```bash
julia --project=. benchmark.jl
```

## Performance Comparison

Compared to the original implementation, this version offers:
- **Major Allocation Reductions**: `RemezWorkspace` eliminates almost all per-iteration heap allocations.
- **Improved Algorithmic Stability**: The new DP-based winnowing logic ensures stable convergence even for delicate rational fits.
- **Optimized Evaluation**: Faster coefficient conversion and evaluation routines.

## License

This package is licensed under the Apache License 2.0.
