# Form-bound constants  (c_h^- u, u) ≤ ε ‖∇u‖² + C_ε ‖u‖²  on H¹(Ω).
#
# * `analytic + Lemma 3`: C from a bound for V^- plus d_h.
# * `certify_ceps_rung`:  paper Sec. 4.4 — a lower bound for ν_1 of
#   -Δ + q_h, q_h = -c_h^-/ε, by Theorem 4.2, given a preliminary bound
#   (ε_*, C_*) with ε_* < ε. Gives C_ε^cert = max(0, -ε ν_low).
# * `ceps_ladder`: repeated rungs over an increasing ε sequence.
# * `ceps_p1_galerkin`: Galerkin (P1, Neumann) value of the optimal C_ε —
#   a LOWER estimate, kept only for comparison with the original code.

"""
    certify_ceps_rung(vm, m, c, eps, eps_pre, C_pre; delta) -> NamedTuple
"""
function certify_ceps_rung(vm, m::SimplexMesh, c::AbstractVector, eps::Float64,
                           eps_pre::Float64, C_pre::Float64; delta::Float64 = 1e-3)
    eps_pre < eps || throw(ArgumentError("need eps_pre < eps"))
    q = -max.(-c, 0.0) ./ eps                 # q_h = -c_h^-/ε  (≤ 0)
    epst = eps_pre / eps; Ct = C_pre / eps    # assumption for q_h^-
    sig = Ct + delta
    nu, info = cecr_eigs(vm, q, sig, 1)
    Ch = interp_constant(m)
    Gam = reaction_constant(m, q)              # = Γ_h(c_h)/ε
    nu_low = cecr_lower(nu, sig, Ch, Gam, epst)[1]
    C = max(0.0, -eps * nu_low)
    return (C = C, nu_sigma = nu[1], nu_low = nu_low, sigma = sig, eps = eps,
            eps_pre = eps_pre, C_pre = C_pre, A = Gam / (1 - epst))
end

"""
    ceps_ladder(vm, m, c, eps_seq, C_first; delta) -> Vector of rung results

`C_first` must be a valid constant for ε = eps_seq[1] (e.g. from
Lemma 3). Each later rung uses the previous rung as preliminary bound.
"""
function ceps_ladder(vm, m::SimplexMesh, c::AbstractVector, eps_seq::Vector{Float64},
                     C_first::Float64; delta::Float64 = 1e-3)
    out = Any[(C = C_first, eps = eps_seq[1], source = "preliminary")]
    for j in 2:length(eps_seq)
        r = certify_ceps_rung(vm, m, c, eps_seq[j], eps_seq[j-1], out[end].C; delta = delta)
        push!(out, r)
    end
    return out
end

"""
    ceps_p1_galerkin(m, c, eps; shift) -> Float64

max over P1 (Neumann) of ((c_h^- u,u) - ε‖∇u‖²)/‖u‖² — a lower estimate
of the optimal C_ε (Galerkin restriction), NOT a certified bound.
"""
function ceps_p1_galerkin(m::SimplexMesh, c::AbstractVector, eps::Float64; shift::Float64)
    K, M, W = p1_matrices(m; cpw = max.(-c, 0.0))
    # smallest eigenvalue of (εK - W) x = μ M x;  C = -μ_1
    λ, _ = smallest_eigs(eps * K - W, M, 1; shift = shift)
    return -λ[1]
end
