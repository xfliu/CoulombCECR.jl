# Collapsed (Duffy) Gauss rules on simplices.
#
# The collapsed rule maps the unit square/cube onto a simplex with the
# first vertex as the collapse point. Its Jacobian (∝ s in 2D, ∝ s²t in 3D)
# cancels a 1/|x - v1| singularity at that vertex, so the same rule serves
# for smooth integrands and for integrands singular at the first vertex.
# An optional radial grading s = u^q smooths stronger singularities such
# as |V - c|^p with V ~ 1/r.

using LinearAlgebra: SymTridiagonal, eigen

"""
    gauss_legendre01(n) -> (x, w)

n-point Gauss–Legendre rule on [0, 1] (Golub–Welsch).
"""
function gauss_legendre01(n::Integer)
    n == 1 && return [0.5], [1.0]
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    F = eigen(SymTridiagonal(zeros(n), β))
    x = (F.values .+ 1) ./ 2
    w = F.vectors[1, :] .^ 2
    return x, w
end

"""
    CollapsedRule{D}

Reference points and weights for the collapsed rule on the unit D-simplex,
stored as barycentric-like parameters. `pts[q]` holds the coefficients
`(1-s, s(1-t), st)` in 2D or `(1-s, s(1-t), st(1-r), str)` in 3D, so that
`x = Σ_i pts[q][i] v_i`. Weights include the Jacobian and are normalised to
sum to 1, so `∫_K f ≈ |K| Σ_q w_q f(x_q)`.
"""
struct CollapsedRule{D}
    bary::Matrix{Float64}   # nq × (D+1)
    w::Vector{Float64}      # nq, sum = 1
end

function collapsed_rule(D::Integer, n::Integer; q::Real = 1)
    x, w = gauss_legendre01(n)
    if D == 2
        nq = n * n
        B = zeros(nq, 3); W = zeros(nq); k = 0
        for i in 1:n, j in 1:n
            u = x[i]; s = u^q; ds = q * u^(q - 1)
            t = x[j]
            k += 1
            B[k, :] .= (1 - s, s * (1 - t), s * t)
            W[k] = w[i] * w[j] * 2 * s * ds        # dx = 2|K| s ds dt
        end
        return CollapsedRule{2}(B, W ./ sum(W))
    elseif D == 3
        nq = n^3
        B = zeros(nq, 4); W = zeros(nq); k = 0
        for i in 1:n, j in 1:n, l in 1:n
            u = x[i]; s = u^q; ds = q * u^(q - 1)
            t = x[j]; r = x[l]
            k += 1
            B[k, :] .= (1 - s, s * (1 - t), s * t * (1 - r), s * t * r)
            W[k] = w[i] * w[j] * w[l] * 6 * s^2 * t * ds   # dx = 6|K| s² t ds dt dr
        end
        return CollapsedRule{3}(B, W ./ sum(W))
    end
    throw(ArgumentError("D must be 2 or 3"))
end
