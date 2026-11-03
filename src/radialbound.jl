# Continuous form bound for V^- by Neumann decoupling into balls around the
# centres and a radial Rayleigh–Ritz solve on each ball:
#
#   (V^- u, u) ≤ ε ‖∇u‖² + C_V(ε) ‖u‖²   for all u ∈ H¹(Ω).
#
# * `radial_neumann_ground`: lowest Neumann eigenvalue of -Δ - Z/|x| on a ball
#   B(0, ρ) ⊂ R^dim (dim = 2, 3). The ground state is radial (simple and
#   rotation invariant), so a Rayleigh–Ritz over polynomials f(r) of degree n
#   (Legendre basis in 2r/ρ - 1, weight r^(dim-1)) suffices; Neumann is the
#   natural boundary condition, nothing is imposed.
# * `decoupled_form_bound`: the constant C_V(ε) (balls B_i = B(a_i, ρ_i),
#   remainder Ω \ ∪B_i bounded by the maximum principle), optimised over a
#   common scaling ρ_i = t ρ_max,i.
#
# Floating-point stage: the Ritz value is an (exponentially convergent) upper
# approximation of ν, reported with the estimate |ν(n) - ν(n/2)|.

"Legendre values P_j(ξ) and derivatives P_j'(ξ), j = 0..n, into P, dP (length n+1)."
function _legendre_PdP!(P::Vector{Float64}, dP::Vector{Float64}, n::Int, ξ::Float64)
    P[1] = 1.0; dP[1] = 0.0
    n == 0 && return P, dP
    P[2] = ξ; dP[2] = 1.0
    for j in 1:n-1                      # P_{j+1} from P_j, P_{j-1}
        P[j+2] = ((2j + 1) * ξ * P[j+1] - j * P[j]) / (j + 1)
        dP[j+2] = dP[j] + (2j + 1) * P[j+1]
    end
    return P, dP
end

"Smallest Ritz value of -Δ - Z/r on radial polynomials of degree n on B(0, ρ) ⊂ R^dim."
function _radial_ritz(dim::Int, Z::Float64, rho::Float64, n::Int)
    (dim == 2 || dim == 3) || throw(ArgumentError("dim must be 2 or 3"))
    # integrands are polynomials of degree ≤ 2n + dim - 1: n + 3 Gauss points are exact
    x, w = gauss_legendre01(n + 3)
    A = zeros(n + 1, n + 1); B = zeros(n + 1, n + 1)
    P = zeros(n + 1); dP = zeros(n + 1)
    d = 2 / rho                          # d/dr = (2/ρ) d/dξ
    for q in eachindex(x)
        r = rho * x[q]
        _legendre_PdP!(P, dP, n, 2x[q] - 1)
        wq = rho * w[q]
        wk = wq * r^(dim - 1); wv = wq * Z * r^(dim - 2)
        for j in 1:n+1, i in 1:n+1
            A[i, j] += wk * d^2 * dP[i] * dP[j] - wv * P[i] * P[j]
            B[i, j] += wk * P[i] * P[j]
        end
    end
    return LinearAlgebra.eigvals(LinearAlgebra.Symmetric(A), LinearAlgebra.Symmetric(B))[1]
end

"""
    radial_neumann_ground(dim, Z, rho; n=40) -> (nu, err)

Lowest Neumann eigenvalue ν of -Δ - Z/|x| on B(0, rho) ⊂ R^dim (dim = 2, 3),
by Rayleigh–Ritz over radial polynomials of degree n; `err = |ν(n) - ν(n÷2)|`.
"""
function radial_neumann_ground(dim::Integer, Z::Real, rho::Real; n::Int = 40)
    nu = _radial_ritz(Int(dim), Float64(Z), Float64(rho), n)
    nu2 = _radial_ritz(Int(dim), Float64(Z), Float64(rho), max(n ÷ 2, 1))
    return (nu = nu, err = abs(nu - nu2))
end

"""
    decoupled_form_bound(centers, charges, eps, rho; bdry_dist, domain_is_ball=false,
                         optimise=true, nt=100, n=40) -> NamedTuple (C, rho, ...)

Continuous constant C_V(ε) with (V^- u,u) ≤ ε‖∇u‖² + C_V‖u‖² on H¹(Ω) for
V^- = Σ_j Z_j/|x - a_j| (`centers` dim × nc, `charges` Z_j > 0), by Neumann
decoupling into the balls B_i = B(a_i, ρ_i):

* on B_i: V^- ≤ Z_i/|x - a_i| + b_i, b_i = Σ_{j≠i} Z_j/(|a_i - a_j| - ρ_i), and
  ∫_{B_i} Z_i u²/|x - a_i| ≤ ε ∫_{B_i} |∇u|² - ε ν_i ∫_{B_i} u²,
  ν_i = radial_neumann_ground(dim, Z_i/ε, ρ_i);
* on Ω \\ ∪B_i: V^- ≤ s_rem = max(max_i [Z_i/ρ_i + b_i], Σ_j Z_j/dist(a_j, ∂Ω))
  (maximum principle: 1/|x| is harmonic in 3D, subharmonic in 2D).

C_V = max(max_i(-ε ν_i + b_i), s_rem). With `domain_is_ball` (Ω = B(a_1, R),
one centre, ρ_1 = R) the remainder is empty and C_V = -ε ν_1.

`rho` are the radii (optimise = false) or the maximal admissible radii
ρ_max,i (optimise = true: ρ_i = t ρ_max,i, t over a grid of `nt` values in
(0, 1] followed by a golden-section refinement). Admissibility
ρ_i ≤ dist(a_i, ∂Ω), ρ_i + ρ_j ≤ |a_i - a_j| is checked.
"""
function decoupled_form_bound(centers::AbstractMatrix, charges::AbstractVector, eps::Real,
                              rho::AbstractVector; bdry_dist::AbstractVector,
                              domain_is_ball::Bool = false, optimise::Bool = true,
                              nt::Int = 100, n::Int = 40)
    dim, nc = size(centers)
    length(charges) == nc == length(rho) == length(bdry_dist) ||
        throw(DimensionMismatch("centers/charges/rho/bdry_dist"))
    all(charges .> 0) || throw(ArgumentError("charges must be positive"))
    eps > 0 || throw(ArgumentError("eps must be positive"))
    dist(i, j) = sqrt(sum(abs2, centers[:, i] .- centers[:, j]))
    for i in 1:nc
        rho[i] <= bdry_dist[i] * (1 + 1e-12) || throw(ArgumentError("ball $i leaves Ω"))
        for j in i+1:nc
            rho[i] + rho[j] <= dist(i, j) * (1 + 1e-12) || throw(ArgumentError("balls $i, $j overlap"))
        end
    end
    if domain_is_ball
        nc == 1 || throw(ArgumentError("domain_is_ball needs exactly one centre"))
        r = radial_neumann_ground(dim, charges[1] / eps, rho[1]; n = n)
        return (C = -eps * r.nu, rho = [Float64(rho[1])], C_ball = [-eps * r.nu], b = [0.0],
                s_rem = -Inf, nu = [r.nu], nu_err = [r.err], t = 1.0)
    end
    s_bdry = sum(charges[j] / bdry_dist[j] for j in 1:nc)
    function at(radii)
        b = [sum((charges[j] / (dist(i, j) - radii[i]) for j in 1:nc if j != i); init = 0.0)
             for i in 1:nc]
        rs = [radial_neumann_ground(dim, charges[i] / eps, radii[i]; n = n) for i in 1:nc]
        Cb = [-eps * rs[i].nu + b[i] for i in 1:nc]
        srem = max(maximum(charges[i] / radii[i] + b[i] for i in 1:nc), s_bdry)
        return (C = max(maximum(Cb), srem), rho = collect(Float64.(radii)), C_ball = Cb, b = b,
                s_rem = srem, nu = [r.nu for r in rs], nu_err = [r.err for r in rs])
    end
    if !optimise
        return merge(at(rho), (t = 1.0,))
    end
    f(t) = at(t .* rho).C
    ts = range(1 / nt, 1, length = nt)
    vals = f.(ts)
    ib = argmin(vals)
    # golden-section refinement on the bracketing grid interval
    a = ts[max(ib - 1, 1)]; bnd = ts[min(ib + 1, nt)]
    g = (sqrt(5) - 1) / 2
    x1 = bnd - g * (bnd - a); x2 = a + g * (bnd - a); f1 = f(x1); f2 = f(x2)
    for _ in 1:40
        if f1 <= f2
            bnd = x2; x2 = x1; f2 = f1; x1 = bnd - g * (bnd - a); f1 = f(x1)
        else
            a = x1; x1 = x2; f1 = f2; x2 = a + g * (bnd - a); f2 = f(x2)
        end
    end
    tbest, Cbest = ts[ib], vals[ib]
    for (tt, ff) in ((x1, f1), (x2, f2))
        ff < Cbest && ((tbest, Cbest) = (tt, ff))
    end
    return merge(at(tbest .* rho), (t = tbest,))
end
