# Multi-centre Coulomb potential V(x) = -Σ_i Z_i / |x - a_i| and the
# element-level quantities needed by the estimates.

"""
    Coulomb{D}(centers, charges)

`centers` is a D × nc matrix, `charges` a length-nc vector (Z_i > 0).
"""
struct Coulomb{D}
    A::Matrix{Float64}
    Z::Vector{Float64}
end
Coulomb(A::AbstractMatrix, Z::AbstractVector) =
    Coulomb{size(A, 1)}(Matrix{Float64}(A), Vector{Float64}(Z))

@inline function Vval(pot::Coulomb{D}, x) where {D}
    s = 0.0
    @inbounds for i in eachindex(pot.Z)
        r2 = 0.0
        for d in 1:D
            r2 += (x[d] - pot.A[d, i])^2
        end
        s -= pot.Z[i] / sqrt(r2)
    end
    return s
end

# Index (1..D+1) of the local vertex of element e that coincides with a
# centre, or 0 if none.
function singular_vertex(pot::Coulomb{D}, m::SimplexMesh{D}, e::Int) where {D}
    @inbounds for i in 1:D+1, c in eachindex(pot.Z)
        r2 = 0.0
        for d in 1:D
            r2 += (m.X[d, m.T[i, e]] - pot.A[d, c])^2
        end
        r2 < (1e-12 * m.hK[e])^2 && return i
    end
    return 0
end

"Minimum over centres of dist(a_i, K)."
function min_center_dist(pot::Coulomb{D}, m::SimplexMesh{D}, e::Int) where {D}
    dmin = Inf
    for c in eachindex(pot.Z)
        dmin = min(dmin, dist_point_simplex(m, e, pot.A[:, c]))
    end
    return dmin
end

# Physical quadrature points of element e for a collapsed rule, with local
# vertex `iv` used as the collapse vertex.
function _element_points(m::SimplexMesh{D}, e::Int, rule::CollapsedRule{D}, iv::Int) where {D}
    order = iv == 0 ? collect(1:D+1) : vcat(iv, [i for i in 1:D+1 if i != iv])
    nq = length(rule.w)
    P = zeros(D, nq)
    @inbounds for q in 1:nq, j in 1:D+1
        vj = m.T[order[j], e]
        b = rule.bary[q, j]
        for d in 1:D
            P[d, q] += b * m.X[d, vj]
        end
    end
    return P, order
end

"""
    QuadSet(D; n_sing, n_near, n_far, q_sing)

Bundle of collapsed rules: singular elements (a centre is a vertex),
near elements (dist(a, K) < 2 h_K) and far elements.
"""
struct QuadSet{D}
    sing::CollapsedRule{D}
    near::CollapsedRule{D}
    far::CollapsedRule{D}
end
QuadSet(D::Integer; n_sing = 12, n_near = 8, n_far = 4, q_sing = 1) =
    QuadSet{D}(collapsed_rule(D, n_sing; q = q_sing), collapsed_rule(D, n_near),
               collapsed_rule(D, n_far))

function _pick_rule(qs::QuadSet, pot, m, e)
    iv = singular_vertex(pot, m, e)
    iv > 0 && return qs.sing, iv
    # cheap nearness test with the centroid
    D = size(m.X, 1)
    dmin = Inf
    for c in eachindex(pot.Z)
        r2 = 0.0
        for d in 1:D
            xc = 0.0
            for i in 1:D+1
                xc += m.X[d, m.T[i, e]]
            end
            r2 += (xc / (D + 1) - pot.A[d, c])^2
        end
        dmin = min(dmin, sqrt(r2))
    end
    return (dmin < 3 * m.hK[e] ? qs.near : qs.far), 0
end

"""
    cell_averages(pot, m; qs) -> c_h

c_h|_K = |K|^{-1} ∫_K V dx with collapsed quadrature (Duffy at the
singular vertex for elements touching a nucleus).
"""
function cell_averages(pot::Coulomb{D}, m::SimplexMesh{D};
                       qs::QuadSet{D} = QuadSet(D)) where {D}
    ne = nelem(m); c = zeros(ne)
    Threads.@threads for e in 1:ne
        rule, iv = _pick_rule(qs, pot, m, e)
        P, _ = _element_points(m, e, rule, iv)
        s = 0.0
        for q in eachindex(rule.w)
            s += rule.w[q] * Vval(pot, @view P[:, q])
        end
        c[e] = s
    end
    return c
end

"""
    lp_dev_integral(pot, m, e, cK, p; qs) -> ∫_K |V - c_K|^p dx
"""
function lp_dev_integral(pot::Coulomb{D}, m::SimplexMesh{D}, e::Int, cK::Float64,
                         p::Float64; qs::QuadSet{D}) where {D}
    rule, iv = _pick_rule(qs, pot, m, e)
    P, _ = _element_points(m, e, rule, iv)
    s = 0.0
    for q in eachindex(rule.w)
        s += rule.w[q] * abs(Vval(pot, @view P[:, q]) - cK)^p
    end
    return s * m.vol[e]
end

"""
    sup_dev(pot, m, e, cK) -> upper bound of sup_K |V - c_K|

Uses V ∈ [-Σ Z_i/dmin_i, -Σ Z_i/dmax_i] on K (exact range for one centre).
Returns Inf if a centre lies in K.
"""
function sup_dev(pot::Coulomb{D}, m::SimplexMesh{D}, e::Int, cK::Float64) where {D}
    lo = 0.0; hi = 0.0
    for c in eachindex(pot.Z)
        a = pot.A[:, c]
        dmin = dist_point_simplex(m, e, a)
        dmin == 0.0 && return Inf
        lo -= pot.Z[c] / dmin
        hi -= pot.Z[c] / maxdist_point_simplex(m, e, a)
    end
    return max(hi - cK, cK - lo)
end
