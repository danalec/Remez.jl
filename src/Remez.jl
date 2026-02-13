module Remez

using LinearAlgebra

export ratfn_minimax, ratfn_minimax_cheb, ratfn_leastsquares, RemezWorkspace

"""
    RemezWorkspace{T}(interval, n, d)

Workspace for Remez algorithm to minimize allocations during iterative steps.
"""
mutable struct RemezWorkspace{T<:AbstractFloat}
    lo::T
    hi::T
    n::Int
    d::Int
    nnodes::Int
    grid::Vector{T}
    yvals::Vector{T}
    matrix::Matrix{T}
    vector::Vector{T}
    all_coeffs::Vector{T}
    c_vals::Vector{T}
    nc_cheb::Vector{T}
    dc_cheb::Vector{T}
    
    # New buffers
    sums::Matrix{T}
    poly_matrix::Matrix{T}
    poly_vector::Vector{T}
    pbasis::Vector{T}
    ti_prev2::Vector{T}
    ti_prev1::Vector{T}
    ti_curr::Vector{T}

    # Extrema buffers
    extrema_x::Vector{T}
    extrema_y::Vector{T}
    num_extrema::Base.RefValue{Int}
    
    # Winnow buffers
    winnow_best_score::Matrix{T}
    winnow_best_idx::Matrix{Int}
    winnow_back_links::Matrix{Int}

    function RemezWorkspace{T}(interval, n, d) where {T}
        lo = T(minimum(interval))
        hi = T(maximum(interval))
        nnodes = 256 * (n + d + 1)
        grid = Vector{T}(undef, nnodes + 1)
        yvals = Vector{T}(undef, nnodes + 1)
        
        # Grid initialization (Chebyshev nodes)
        mid = (hi + lo) / 2
        halfwid = (hi - lo) / 2
        for i in 0:nnodes
            grid[i+1] = mid - halfwid * cospi(T(i) / T(nnodes))
        end

        matrix = Matrix{T}(undef, n + d + 1, n + d + 1)
        vector = Vector{T}(undef, n + d + 1)
        all_coeffs = Vector{T}(undef, n + d + 1)
        c_vals = Vector{T}(undef, max(n, d) + 1)
        nc_cheb = Vector{T}(undef, n + 1)
        dc_cheb = Vector{T}(undef, d + 1)
        
        sums = Matrix{T}(undef, max(n, d) * 2 + 1, 3)
        poly_matrix = Matrix{T}(undef, n + 2, n + 2)
        poly_vector = Vector{T}(undef, n + 2)
        
        pbasis = Vector{T}(undef, max(n, d) + 1)
        ti_prev2 = Vector{T}(undef, max(n, d) + 1)
        ti_prev1 = Vector{T}(undef, max(n, d) + 1)
        ti_curr = Vector{T}(undef, max(n, d) + 1)

        # Extrema buffers
        extrema_x = Vector{T}(undef, nnodes + 2) # +2 safety
        extrema_y = Vector{T}(undef, nnodes + 2)
        num_extrema = Ref(0)
        
        # Winnow buffers
        n_winnow = n + d + 2
        winnow_best_score = Matrix{T}(undef, n_winnow, 2)
        winnow_best_idx = Matrix{Int}(undef, n_winnow, 2)
        winnow_back_links = Matrix{Int}(undef, nnodes + 2, n_winnow)

        new{T}(lo, hi, n, d, nnodes, grid, yvals, matrix, vector, all_coeffs, c_vals, nc_cheb, dc_cheb,
               sums, poly_matrix, poly_vector, pbasis, ti_prev2, ti_prev1, ti_curr,
               extrema_x, extrema_y, num_extrema, winnow_best_score, winnow_best_idx, winnow_back_links)
    end
end

"""
Evaluate a polynomial.
"""
@inline function poly_eval(coeffs::AbstractVector{T}, x::T) where {T<:AbstractFloat}
    evalpoly(x, coeffs)
end

"""
    poly_cheb_eval(coeffs, x, lo, hi)

Evaluate a polynomial defined by Chebyshev coefficients using Clenshaw's algorithm.
"""
@inline function poly_cheb_eval(coeffs::AbstractVector{T}, x::T, lo::T, hi::T) where {T<:AbstractFloat}
    n = length(coeffs)
    n == 0 && return zero(T)
    n == 1 && return coeffs[1]
    
    # Map x to [-1, 1]
    x_mapped = (2 * x - (hi + lo)) / (hi - lo)
    two_x = 2 * x_mapped
    
    b2 = zero(T)
    b1 = zero(T)
    
    @inbounds for i in n:-1:2
        b0 = coeffs[i] + two_x * b1 - b2
        b2 = b1
        b1 = b0
    end
    
    return coeffs[1] + x_mapped * b1 - b2
end

"""
Evaluate a rational function.
"""
@inline function ratfn_eval(ncoeffs::AbstractVector{T}, dcoeffs::AbstractVector{T}, x::Number) where {T<:AbstractFloat}
    tx = T(x)
    return poly_eval(ncoeffs, tx) / poly_eval(dcoeffs, tx)
end

"""
    ratfn_cheb_eval(ncoeffs, dcoeffs, x, lo, hi)

Evaluate a rational function defined by Chebyshev coefficients.
"""
@inline function ratfn_cheb_eval(ncoeffs::AbstractVector{T}, dcoeffs::AbstractVector{T}, x::Number, lo::T, hi::T) where {T<:AbstractFloat}
    tx = T(x)
    return poly_cheb_eval(ncoeffs, tx, lo, hi) / poly_cheb_eval(dcoeffs, tx, lo, hi)
end

"""
    pcoeffs = cheb_to_power(ccoeffs, lo, hi)

Convert Chebyshev coefficients `ccoeffs` defined on `[lo, hi]` to power basis coefficients.
"""
function cheb_to_power(ccoeffs::AbstractVector{T}, lo::T, hi::T) where {T<:AbstractFloat}
    n = length(ccoeffs) - 1
    n < 0 && return T[]
    
    res = Vector{T}(undef, n + 1)
    ti_prev2 = zeros(T, n + 1)
    ti_prev1 = zeros(T, n + 1)
    ti_curr = zeros(T, n + 1)
    
    cheb_to_power_inplace!(res, ccoeffs, lo, hi, ti_prev2, ti_prev1, ti_curr)
    return res
end

function cheb_to_power_inplace!(res::AbstractVector{T}, ccoeffs::AbstractVector{T}, lo::T, hi::T,
                                ti_prev2::AbstractVector{T}, ti_prev1::AbstractVector{T}, ti_curr::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(ccoeffs) - 1
    fill!(res, zero(T))
    fill!(ti_prev2, zero(T))
    fill!(ti_prev1, zero(T))
    
    a = 2 / (hi - lo)
    b = -(hi + lo) / (hi - lo)
    
    ti_prev2[1] = one(T)
    @inbounds for j in 1:length(res)
        res[j] += ccoeffs[1] * ti_prev2[j]
    end
    
    if n >= 1
        ti_prev1[1] = b
        ti_prev1[2] = a
        @inbounds for j in 1:length(res)
            res[j] += ccoeffs[2] * ti_prev1[j]
        end
    end
    
    for i in 2:n
        fill!(ti_curr, zero(T))
        @inbounds for j in 1:i+1
            ti_curr[j] = 2 * b * ti_prev1[j] - ti_prev2[j]
        end
        @inbounds for j in 1:i
            ti_curr[j+1] += 2 * a * ti_prev1[j]
        end
        @inbounds for j in 1:length(res)
            res[j] += ccoeffs[i+1] * ti_curr[j]
        end
        copyto!(ti_prev2, ti_prev1)
        copyto!(ti_prev1, ti_curr)
    end
end

"""
    ratfn_leastsquares!(ws, f, w)

Internal version of `ratfn_leastsquares` that uses a workspace to avoid allocations.
"""
function ratfn_leastsquares!(ws::RemezWorkspace{T}, f::Function,
                             w = (x, y) -> one(T)) where {T<:AbstractFloat}
    n, d = ws.n, ws.d
    lo, hi = ws.lo, ws.hi
    
    fill!(ws.sums, zero(T))
    
    @inbounds for x in ws.grid
        y = f(x)
        weight = w(x, y)
        x_mapped = (2 * x - (hi + lo)) / (hi - lo)
        
        tk2 = one(T)
        tk1 = x_mapped
        
        y0 = weight
        y1 = y * weight
        y2 = y * y * weight
        
        ws.sums[1, 1] += tk2 * y0
        ws.sums[1, 2] += tk2 * y1
        ws.sums[1, 3] += tk2 * y2
        
        if size(ws.sums, 1) >= 2
            ws.sums[2, 1] += tk1 * y0
            ws.sums[2, 2] += tk1 * y1
            ws.sums[2, 3] += tk1 * y2
        end
        
        for k in 3:size(ws.sums, 1)
            tk = 2 * x_mapped * tk1 - tk2
            ws.sums[k, 1] += tk * y0
            ws.sums[k, 2] += tk * y1
            ws.sums[k, 3] += tk * y2
            tk2 = tk1
            tk1 = tk
        end
    end

    function get_sum_titj(i, j, m)
        s1 = ws.sums[i + j + 1, m]
        s2 = ws.sums[abs(i - j) + 1, m]
        return (s1 + s2) / 2
    end

    @inbounds for i in 0:n
        row = 1 + i
        for j in 0:n
            ws.matrix[row, 1 + j] = get_sum_titj(i, j, 1)
        end
        for j in 1:d
            ws.matrix[row, 1 + n + j] = -get_sum_titj(i, j, 2)
        end
        ws.vector[row] = get_sum_titj(i, 0, 2)
    end
    
    @inbounds for i in 1:d
        row = 1 + n + i
        for j in 0:n
            ws.matrix[row, 1 + j] = -get_sum_titj(i, j, 2)
        end
        for j in 1:d
            ws.matrix[row, 1 + n + j] = get_sum_titj(i, j, 3)
        end
        ws.vector[row] = -get_sum_titj(i, 0, 3)
    end

    # Use LU factorization to solve linear system in-place
    # ws.matrix is overwritten by lu!
    # ws.vector is overwritten by ldiv! with the solution
    F = lu!(ws.matrix)
    ldiv!(F, ws.vector)
    ws.all_coeffs .= ws.vector

    @inbounds ws.nc_cheb .= ws.all_coeffs[1:n+1]
    ws.dc_cheb[1] = one(T)
    @inbounds ws.dc_cheb[2:d+1] .= ws.all_coeffs[n+2:n+d+1]
    
    return ws.nc_cheb, ws.dc_cheb
end

"""
    nc, dc = ratfn_leastsquares(f, xvals, n, d, w)

Public wrapper for least-squares rational fit using Chebyshev basis internally.
"""
function ratfn_leastsquares(f::Function, xvals::AbstractVector{T}, n::Int, d::Int,
                             w = (x, y) -> one(T)) where {T<:AbstractFloat}
    lo, hi = minimum(xvals), maximum(xvals)
    ws = RemezWorkspace{T}((lo, hi), n, d)
    if length(xvals) != length(ws.grid)
        ws.grid = Vector{T}(xvals) # Copy if length differs
    else
        ws.grid .= xvals
    end
    nc_cheb, dc_cheb = ratfn_leastsquares!(ws, f, w)
    
    nc = Vector{T}(undef, n + 1)
    dc = Vector{T}(undef, d + 1)
    cheb_to_power_inplace!(nc, nc_cheb, ws.lo, ws.hi, ws.ti_prev2, ws.ti_prev1, ws.ti_curr)
    cheb_to_power_inplace!(dc, dc_cheb, ws.lo, ws.hi, ws.ti_prev2, ws.ti_prev1, ws.ti_curr)
    return nc, dc
end

"""
    ratfn_equal_deviation!(ws, f, coords, prev_err, w)

Internal version of `ratfn_equal_deviation` that uses a workspace.
"""
function ratfn_equal_deviation!(ws::RemezWorkspace{T}, f::Function, coords::AbstractVector{T},
                                 prev_err::T, w = (x, y) -> one(T)) where {T<:AbstractFloat}
    n, d = ws.n, ws.d
    lo, hi = ws.lo, ws.hi
    
    if d == 0
        currsign = one(T)
        @inbounds for i in 1:n+2
            x = coords[i]
            x_mapped = (2 * x - (hi + lo)) / (hi - lo)
            
            tk2 = one(T)
            ws.poly_matrix[i, 1] = tk2
            if n >= 1
                tk1 = x_mapped
                ws.poly_matrix[i, 2] = tk1
                for j in 2:n
                    tk = 2 * x_mapped * tk1 - tk2
                    ws.poly_matrix[i, 1 + j] = tk
                    tk2 = tk1
                    tk1 = tk
                end
            end
            
            y = f(x)
            ws.poly_vector[i] = y
            ws.poly_matrix[i, n+2] = currsign / w(x, y)
            currsign = -currsign
        end

        # Solve system in-place
        F = lu!(ws.poly_matrix)
        ldiv!(F, ws.poly_vector)
        outvector = ws.poly_vector

        @inbounds ws.nc_cheb .= outvector[1:n+1]
        ws.dc_cheb[1] = one(T)
        return ws.nc_cheb, ws.dc_cheb, outvector[n+2]
    else
        function try_e(e)
            currsign = one(T)
            @inbounds for i in 1:n+d+1
                x = coords[i]
                y = f(x)
                y_adj = y - currsign * e / w(x, y)
                x_mapped = (2 * x - (hi + lo)) / (hi - lo)
                
                ws.c_vals[1] = one(T)
                if max(n, d) >= 1
                    ws.c_vals[2] = x_mapped
                end
                for deg in 2:max(n, d)
                    ws.c_vals[deg + 1] = 2 * x_mapped * ws.c_vals[deg] - ws.c_vals[deg - 1]
                end
                
                for j in 0:n
                    ws.matrix[i, 1 + j] = ws.c_vals[1 + j]
                end
                for j in 1:d
                    ws.matrix[i, 1 + n + j] = -ws.c_vals[1 + j] * y_adj
                end
                ws.vector[i] = y_adj
                currsign = -currsign
            end

            F = lu!(ws.matrix)
            ldiv!(F, ws.vector)
            ws.all_coeffs .= ws.vector

            tx = coords[n+d+2]
            @inbounds ws.nc_cheb .= view(ws.all_coeffs, 1:n+1)
            ws.dc_cheb[1] = one(T)
            @inbounds ws.dc_cheb[2:d+1] .= view(ws.all_coeffs, n+2:n+d+1)
            
            y = f(tx)
            last_err = (ratfn_cheb_eval(ws.nc_cheb, ws.dc_cheb, tx, lo, hi) - y) * w(tx, y) * -currsign
            return last_err
        end

        epsbits = precision(T)
        threshold = T(2)^(-epsbits / 2)
        e0 = prev_err
        e1 = try_e(e0)
        
        if abs(e1 - e0) > threshold
            s = sign(e1 - e0)
            while abs(e1 - e0) > threshold
                emid = (e1 + e0) / 2
                enew = try_e(emid)
                if sign(enew - emid) == s
                    e0 = emid
                else
                    e1 = emid
                end
            end
        end

        return ws.nc_cheb, ws.dc_cheb
    end
end

"""
Find the extrema of a function within a given interval, writing to workspace buffers.
"""
function find_extrema!(ws::RemezWorkspace{T}, f) where {T<:AbstractFloat}
    grid = ws.grid
    len = length(grid)
    count = 0
    
    y0 = f(grid[1])
    y1 = f(grid[2])
    
    if abs(y0) >= abs(y1)
        count += 1
        ws.extrema_x[count] = grid[1]
        ws.extrema_y[count] = y0
    end

    yp = y0
    yi = y1
    
    for i in 2:len-1
        yn = f(grid[i+1])
        
        if (yi > yp && yi >= yn) || (yi < yp && yi <= yn)
            xp, xi, xn = grid[i-1], grid[i], grid[i+1]
            if yi > yp && yi >= yn
                # Maximize f
                x, y = goldensection(f, xp, xi, xn)
                count += 1
                ws.extrema_x[count] = x
                ws.extrema_y[count] = y
            else
                # Maximize -f using helper to avoid closure allocation
                x, y = goldensection_signed(f, xp, xi, xn, -one(T))
                count += 1
                ws.extrema_x[count] = x
                ws.extrema_y[count] = y
            end
        end
        yp = yi
        yi = yn
    end
    
    if abs(yi) >= abs(yp)
        count += 1
        ws.extrema_x[count] = grid[len]
        ws.extrema_y[count] = yi
    end
    
    ws.num_extrema[] = count
    return count
end

function goldensection_signed(f, a::T, b::T, c::T, s::T) where {T<:AbstractFloat}
    epsbits = precision(T)
    threshold = abs(c - a) * (T(2)^(-epsbits / 2))
    one_over_phi = T(2) / (one(T) + sqrt(T(5)))

    if abs(b-a) < abs(c-a)
        a,  c  = c,  a
    end

    fa, fb, fc = s*f(a), s*f(b), s*f(c)

    while abs(c-a) > threshold
        d = a + (b-a) * one_over_phi
        fd = s*f(d)
        if fd > fb
            a,  b,  c  = a,  d,  b
            fa, fb, fc = fa, fd, fb
        else
            a,  b,  c  = c,  b,  d
            fa, fb, fc = fc, fb, fd
        end
    end
    return (b, s*fb) # return true f(b) value? No, return (b, f(b)). s*fb is the maximized value.
    # if s is -1, maximized value is -f(b). We want f(b), so we return -maximized.
    # So return (b, s*fb * s) = (b, fb).
end

"""
Winnow extrema using workspace buffers.
"""
function winnow_extrema!(ws::RemezWorkspace{T}, count::Int, n::Int) where {T<:AbstractFloat}
    # Uses ws.extrema_x[1:count], ws.extrema_y[1:count]
    # Writes result to ws.extrema_x[1:n], ws.extrema_y[1:n]
    
    best_score = ws.winnow_best_score
    best_idx_at_i = ws.winnow_best_idx
    back_links = ws.winnow_back_links
    
    fill!(best_score, -T(Inf))
    fill!(best_idx_at_i, 0)
    # back_links doesn't need init if we write carefully?
    # Better to zero it for safety if needed, but it's large.
    # We only read what we write.
    
    # ... DP implementation ...
    # We need to be careful with indices.
    # ws.extrema_y is 1-based.
    
    for idx in 1:count
        y = ws.extrema_y[idx]
        s = (y > 0) ? 1 : ((y < 0) ? 2 : 0)
        s == 0 && continue
        abs_y = abs(y)

        # Base case: i = 1
        if abs_y > best_score[1, s]
            best_score[1, s] = abs_y
            best_idx_at_i[1, s] = idx
        end

        # Recursive step: i > 1
        others = 3 - s
        for i in 2:n
            if best_score[i-1, others] > -T(Inf)
                score = min(abs_y, best_score[i-1, others])
                if score > best_score[i, s]
                    best_score[i, s] = score
                    best_idx_at_i[i, s] = idx
                    back_links[idx, i] = best_idx_at_i[i-1, others]
                end
            end
        end
    end

    final_sign = (best_score[n, 1] > best_score[n, 2]) ? 1 : 2
    if best_score[n, final_sign] == -T(Inf)
        error("winnow_extrema: could not find enough alternating extrema (expected $n).")
    end

    curr_idx = best_idx_at_i[n, final_sign]
    for i in n:-1:1
        ws.extrema_x[i] = ws.extrema_x[curr_idx]
        ws.extrema_y[i] = ws.extrema_y[curr_idx]
        if i > 1
            curr_idx = back_links[curr_idx, i]
        end
    end
    
    ws.num_extrema[] = n
    return n
end

"""
Golden-section search to find a maximum of a function.
"""
function goldensection(f::Function, a::T, b::T, c::T) where {T<:AbstractFloat}
    epsbits = precision(T)
    threshold = abs(c - a) * (T(2)^(-epsbits / 2))
    one_over_phi = T(2) / (one(T) + sqrt(T(5)))

    if abs(b-a) < abs(c-a)
        a,  c  = c,  a
    end

    fa, fb, fc = f(a), f(b), f(c)

    while abs(c-a) > threshold
        d = a + (b-a) * one_over_phi
        fd = f(d)
        if fd > fb
            a,  b,  c  = a,  d,  b
            fa, fb, fc = fa, fd, fb
        else
            a,  b,  c  = c,  b,  d
            fa, fb, fc = fc, fb, fd
        end
    end
    return (b, fb)
end

"""
Find the extrema of a function within a given interval.
"""
function find_extrema(f::Function, grid::AbstractVector{T}) where {T<:AbstractFloat}
    len = length(grid)
    extrema = Tuple{T, T}[]
    sizehint!(extrema, 2 * len ÷ 256 + 2)
    
    y0 = f(grid[1])
    y1 = f(grid[2])
    
    if abs(y0) >= abs(y1)
        push!(extrema, (grid[1], y0))
    end

    yp = y0
    yi = y1
    for i in 2:len-1
        yn = f(grid[i+1])
        
        if (yi > yp && yi >= yn) || (yi < yp && yi <= yn)
            xp, xi, xn = grid[i-1], grid[i], grid[i+1]
            if yi > yp && yi >= yn
                push!(extrema, goldensection(f, xp, xi, xn))
            else
                x, y = goldensection(x->-f(x), xp, xi, xn)
                push!(extrema, (x, -y))
            end
        end
        yp = yi
        yi = yn
    end
    
    if abs(yi) >= abs(yp)
        push!(extrema, (grid[len], yi))
    end
    
    sort!(extrema, by=x->x[1])
    return extrema
end

"""
Winnow a list of a function's extrema to give a subsequence of alternating signs.
"""
function winnow_extrema(extrema::AbstractVector{Tuple{T,T}}, n::Int) where {T<:AbstractFloat}
    len = length(extrema)
    # dynamic programming state: (score, backpointer_to_index)
    best_score = fill(-T(Inf), n, 2)
    best_idx_at_i = fill(0, n, 2)
    back_links = zeros(Int, len, n)

    for (idx, (x, y)) in enumerate(extrema)
        s = (y > 0) ? 1 : ((y < 0) ? 2 : 0)
        s == 0 && continue
        abs_y = abs(y)

        # Base case: i = 1
        if abs_y > best_score[1, s]
            best_score[1, s] = abs_y
            best_idx_at_i[1, s] = idx
            # back_links[idx, 1] is 0
        end

        # Recursive step: i > 1
        others = 3 - s
        for i in 2:n
            if best_score[i-1, others] > -T(Inf)
                score = min(abs_y, best_score[i-1, others])
                if score > best_score[i, s]
                    best_score[i, s] = score
                    best_idx_at_i[i, s] = idx
                    back_links[idx, i] = best_idx_at_i[i-1, others]
                end
            end
        end
    end

    final_sign = (best_score[n, 1] > best_score[n, 2]) ? 1 : 2
    if best_score[n, final_sign] == -T(Inf)
        error("winnow_extrema: could not find enough alternating extrema (expected $n).")
    end

    ret = Vector{Tuple{T,T}}(undef, n)
    curr_idx = best_idx_at_i[n, final_sign]
    for i in n:-1:1
        ret[i] = extrema[curr_idx]
        if i > 1
            curr_idx = back_links[curr_idx, i]
        end
    end
    return ret
end

"""
    ratfn_minimax!(ws, f, w)

Internal version of `ratfn_minimax` using workspace.
"""
function ratfn_minimax!(ws::RemezWorkspace{T}, f, w = (x, y) -> one(T)) where {T<:AbstractFloat}
    ratfn_leastsquares!(ws, f, w)
    epsbits = precision(T)
    threshold = T(2)^(-epsbits / 3)
    max_iters = 100
    local max_err
    
    # Error function
    err_func = x -> (ratfn_cheb_eval(ws.nc_cheb, ws.dc_cheb, x, ws.lo, ws.hi) - f(x)) * w(x, f(x))

    for iter in 1:max_iters
        # Update extrema
        count = find_extrema!(ws, err_func)
        
        # Winnow
        n_needed = ws.n + ws.d + 2
        count = winnow_extrema!(ws, count, n_needed)

        # Calculate error stats from winnowed extrema
        min_err = abs(ws.extrema_y[1])
        max_err = abs(ws.extrema_y[1])
        for i in 2:count
             val = abs(ws.extrema_y[i])
             if val < min_err min_err = val end
             if val > max_err max_err = val end
        end

        variation = (max_err - min_err) / max_err
        if variation < threshold
            extrema = [(ws.extrema_x[i], ws.extrema_y[i]) for i in 1:count]
            return (copy(ws.nc_cheb), copy(ws.dc_cheb), max_err, extrema)
        end

        ratfn_equal_deviation!(ws, f, view(ws.extrema_x, 1:count), max_err, w)
    end
    
    # Final step
    count = find_extrema!(ws, err_func)
    n_needed = ws.n + ws.d + 2
    count = winnow_extrema!(ws, count, n_needed)
    
    max_err = zero(T)
    for i in 1:count
        val = abs(ws.extrema_y[i])
        if val > max_err max_err = val end
    end
    
    extrema = [(ws.extrema_x[i], ws.extrema_y[i]) for i in 1:count]
    return (copy(ws.nc_cheb), copy(ws.dc_cheb), max_err, extrema)
end

"""
    N,D,E,X = ratfn_minimax_cheb(f, interval, n, d, w)

Find a minimax rational-function approximation in Chebyshev basis.
"""
function ratfn_minimax_cheb(f, interval, n::Int, d::Int,
                            w = (x, y) -> one(float(real(first(interval)))))
    T = typeof(float(real(first(interval))))
    ws = RemezWorkspace{T}(interval, n, d)
    return ratfn_minimax!(ws, f, w)
end

"""
    N,D,E,X = ratfn_minimax(f, interval, n, d, w)

Top-level function for minimax approximation.
"""
function ratfn_minimax(f, interval, n::Int, d::Int,
                       w = (x, y) -> one(float(real(first(interval)))))
    nc_cheb, dc_cheb, max_err, extrema = ratfn_minimax_cheb(f, interval, n, d, w)
    T = typeof(max_err)
    lo, hi = T(minimum(interval)), T(maximum(interval))
    
    nc = Vector{T}(undef, n + 1)
    dc = Vector{T}(undef, d + 1)

    ti_prev2 = Vector{T}(undef, max(n, d) + 1)
    ti_prev1 = Vector{T}(undef, max(n, d) + 1)
    ti_curr = Vector{T}(undef, max(n, d) + 1)
    
    cheb_to_power_inplace!(nc, nc_cheb, lo, hi, ti_prev2, ti_prev1, ti_curr)
    cheb_to_power_inplace!(dc, dc_cheb, lo, hi, ti_prev2, ti_prev1, ti_curr)
    
    return nc, dc, max_err, extrema
end

"""
    worst = wellcond(coeffs, lo, hi)

Check if a polynomial is well-conditioned.
"""
function wellcond(coeffs, lo, hi)
    x = max(abs(lo), abs(hi))
    worst, so_far = 0.0, 0.0
    for i = length(coeffs):-1:1
        coeff = abs(coeffs[i])
        so_far *= x
        if coeff != 0
            worst = max(worst, so_far / coeff)
            so_far += coeff
        end
    end
    return worst
end

end # module
