# Mesh constants of the revised paper.

"C_h = h_max / π (Payne–Weinberger, Prop. 1)."
interp_constant(m::SimplexMesh) = maximum(m.hK) / π

"Γ_h = max_K c_h^-|_K h_K² / π² (Def. 1)."
reaction_constant(m::SimplexMesh, c::AbstractVector) =
    maximum(max.(-c, 0.0) .* m.hK .^ 2) / π^2

"""
    cecr_lower(mu_sigma, sigma, Ch, Gamma, eps) -> L^{μ,σ}

Theorem 4.2: L = μ^σ / ((1 + A_h)(1 + C_h² μ^σ)) - σ with A_h = Γ_h/(1-ε).
"""
function cecr_lower(mu_sigma, sigma, Ch, Gamma, eps)
    A = Gamma / (1 - eps)
    return @. mu_sigma / ((1 + A) * (1 + Ch^2 * mu_sigma)) - sigma
end

"Theorem 5.4: L_k = (1 - ε_h)(L^{μ,σ} + σ) - σ."
corrected_lower(Lmu, sigma, eps_h) = @. (1 - eps_h) * (Lmu + sigma) - sigma
