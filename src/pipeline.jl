# End-to-end evaluation of the two-sided bound (paper Thm 5.4) on one mesh.

using Printf

"""
    CaseSpec

Physical and analytic parameters of one test problem (paper Table 6.1).
"""
Base.@kwdef struct CaseSpec
    name::String
    dim::Int
    centers::Matrix{Float64}          # dim × nc
    charges::Vector{Float64}
    eps::Float64                      # ε in Assumption 2
    Ceps::Float64                     # analytic C_ε (≥ C0 + d_h)
    sigma::Float64                    # analytic shift σ > C_ε
    eps0::Float64                     # ε₀ of the continuous bound for V^-
    C0::Float64                       # C₀ of the continuous bound for V^-
    S::Float64                        # embedding constant in d_h (S_8 or C_6)
    p::Float64                        # Hölder exponent (4/3 or 3/2)
    per_centre::Bool = false          # local embedding constant per centre
    r_loc::Float64 = Inf              # radius of validity of a local constant
    neig::Int = 3
    on_boundary::Function = x -> false  # geometric boundary test (conformity check)
    neumann_p1::Bool = false          # also compute λ^{N,P1}
    max_rungs::Int = 3                # ladders with 0..max_rungs intermediate ε values
    cont_bounds::Vector{Tuple{Float64,Float64}} = Tuple{Float64,Float64}[]
                                      # extra continuous bounds (ε, C) for V^-
                                      # (e.g. certified on a finer mesh)
    symmetry::Vector{Int} = Int[]     # coordinate planes x_i = 0 of mirror symmetry of
                                      # domain, potential and mesh (empty: none)
end

"""
    cecr_eigs_case(spec, vm, c, sigma, k; kw...)

CECR eigenvalues for a case: full solve, or the symmetry-sector solve
(`cecr_eigs_symmetric`) when `spec.symmetry` is non-empty.
"""
cecr_eigs_case(spec::CaseSpec, vm, c, sigma, k; kw...) =
    isempty(spec.symmetry) ? cecr_eigs(vm, c, sigma, k; kw...) :
                             cecr_eigs_symmetric(vm, c, sigma, k, spec.symmetry; kw...)

load_vfem_mesh(dir, dim) = dim == 2 ? mesh2d_load_ne(dir) : mesh_load_from_folder(dir)

"""
    run_case(spec, meshdir; verbose=true) -> Dict

Computes every quantity reported in the paper tables for one mesh.
"""
function run_case(spec::CaseSpec, meshdir::AbstractString; verbose::Bool = true)
    t0 = time()
    log(s) = verbose && (println(s); flush(stdout))
    R = Dict{String,Any}("case" => spec.name, "mesh" => basename(rstrip(meshdir, '/')))
    vm = load_vfem_mesh(meshdir, spec.dim)
    m = SimplexMesh(vm)
    pot = Coulomb(spec.centers, spec.charges)
    R["n_elem"] = nelem(m); R["n_node"] = nnode(m); R["h_max"] = maximum(m.hK)
    R["hK_min"] = minimum(m.hK)
    R["interior_free_facets"] = interior_free_facets(m, spec.on_boundary)
    R["interior_free_facets"] > 0 &&
        @warn "mesh has $(R["interior_free_facets"]) unmatched interior facets: P1 bounds are not conforming"
    if R["interior_free_facets"] > 0
        m = SimplexMesh{spec.dim}(m.X, m.T, m.vol, m.hK, BitVector([spec.on_boundary(m.X[:, i]) for i in 1:nnode(m)]))
    end
    log(@sprintf("[%s] %s: ne=%d nv=%d hmax=%.4f  (load %.1fs)", spec.name, R["mesh"],
                 nelem(m), nnode(m), R["h_max"], time() - t0))

    c = cell_averages(pot, m); R["t_ch"] = time() - t0
    Ch = interp_constant(m); Gam = reaction_constant(m, c)
    R["C_h"] = Ch; R["Gamma_h"] = Gam; R["A_h"] = Gam / (1 - spec.eps)

    # ---- (S3) perturbation d_h ------------------------------------------
    dh = direct_dh(pot, m, c, spec.S, spec.p; per_centre = spec.per_centre, r_loc = spec.r_loc)
    R["d_h"] = dh.dh; R["d_h_patch"] = dh.patch; R["d_h_off"] = dh.off
    R["rho_opt"] = dh.rho; R["n_patch"] = dh.npatch; R["patch_rmax"] = dh.rmax
    # Lemma 3 admissibility of the analytic (ε, C_ε)
    R["lemma3_ok"] = (spec.eps0 + dh.dh <= spec.eps) && (spec.C0 + dh.dh <= spec.Ceps)
    log(@sprintf("  Gamma_h=%.3e  d_h=%.4e (patch %.3e, off %.3e, %d patch elts)  lemma3=%s",
                 Gam, dh.dh, dh.patch, dh.off, dh.npatch, R["lemma3_ok"]))

    # ---- analytic shift ---------------------------------------------------
    kap = min(1 - spec.eps, spec.sigma - spec.Ceps)
    epsh = dh.dh / kap
    mu, inf1 = cecr_eigs_case(spec, vm, c, spec.sigma, spec.neig)
    Lmu = cecr_lower(mu, spec.sigma, Ch, Gam, spec.eps)
    R["kappa"] = kap; R["eps_h"] = epsh
    R["mu_minus_sigma"] = mu .- spec.sigma; R["L_mu_sigma"] = Lmu
    R["L"] = corrected_lower(Lmu, spec.sigma, epsh); R["cecr_resid"] = inf1.residuals
    R["n_dof_cecr"] = size(cecr_matrices(vm, c)[1], 1)
    log(@sprintf("  mu-sigma=%s  L=%s", string(round.(R["mu_minus_sigma"], digits = 4)),
                 string(round.(R["L"], digits = 4))))

    # ---- upper bounds -------------------------------------------------------
    shiftP1 = -(spec.C0 + 0.5)           # λ_1 ≥ -C0 by the continuous form bound
    U, infU = p1_bounds(m, pot, spec.neig; bc = :dirichlet, shift = shiftP1)
    R["U_dirichlet"] = U
    if spec.neumann_p1
        UN, _ = p1_bounds(m, pot, spec.neig; bc = :neumann, shift = shiftP1)
        R["U_neumann"] = UN
    end
    log("  U_D=" * string(round.(U, digits = 4)))

    # ---- form-bound constants ----------------------------------------------
    # Each continuous bound (e0, C0) for V^- gives, by Lemma 3, the discrete
    # bound (e0 + d_h, C0 + d_h); it is used directly (if e0 + d_h ≤ ε) and as
    # the starting rung of ladders up to ε (Sec. 4.4).
    cands = Any[]
    starts = vcat([(spec.eps0, spec.C0, "analytic")],
                  [(e, C, "transfer") for (e, C) in spec.cont_bounds])
    for (e0, C0, src) in starts
        ef = e0 + dh.dh; Cf = C0 + dh.dh
        ef <= spec.eps && push!(cands, (C = Cf, how = "$(src):direct", eps_start = ef, rungs = Any[]))
        ef < spec.eps || continue
        for nm in 0:spec.max_rungs
            seq = [ef + (spec.eps - ef) * k / (nm + 1) for k in 0:nm+1]
            try
                lad = ceps_ladder(vm, m, c, seq, Cf)
                push!(cands, (C = lad[end].C, how = "$(src):ladder$(nm+1)", eps_start = ef,
                              rungs = [(eps = r.eps, C = r.C) for r in lad]))
            catch err
                log("  ladder from $(src) with $(nm) mids failed: $(err)")
            end
        end
    end
    ib = argmin([x.C for x in cands])
    Ccert = cands[ib].C
    R["C_cert"] = Ccert; R["C_cert_how"] = cands[ib].how
    R["C_cert_candidates"] = [Dict("C" => x.C, "how" => x.how,
                                   "rungs" => [[r.eps, r.C] for r in x.rungs]) for x in cands]
    ia = findall(x -> startswith(x.how, "analytic"), cands)
    Can = isempty(ia) ? NaN : minimum(cands[i].C for i in ia)
    R["C_cert_analytic"] = Can                 # best constant without transferred bounds
    R["C_p1_galerkin"] = ceps_p1_galerkin(m, c, spec.eps; shift = -(spec.C0 + dh.dh + 0.5))
    log(@sprintf("  C_cert(Sec.4.4)=%.4f [%s]   C_P1(Galerkin, lower est.)=%.4f",
                 Ccert, cands[ib].how, R["C_p1_galerkin"]))

    # ---- optimised shift with the certified constant -------------------------
    tags = Any[("opt", Ccert), ("optP1", R["C_p1_galerkin"])]
    isfinite(Can) && Can != Ccert && push!(tags, ("optA", Can))   # analytic-start-only L_opt
    for (tag, Cv) in tags
        sopt = Cv + (1 - spec.eps)
        kopt = min(1 - spec.eps, sopt - Cv)
        muo, _ = cecr_eigs_case(spec, vm, c, sopt, spec.neig)
        Lmuo = cecr_lower(muo, sopt, Ch, Gam, spec.eps)
        R["sigma_"*tag] = sopt; R["eps_h_"*tag] = dh.dh / kopt
        R["L_"*tag] = corrected_lower(Lmuo, sopt, dh.dh / kopt)
    end
    log("  L_opt=" * string(round.(R["L_opt"], digits = 4)) *
        "  (with P1 C: " * string(round.(R["L_optP1"], digits = 4)) * ")")
    if !haskey(R, "L_optA") && isfinite(Can)
        R["sigma_optA"] = R["sigma_opt"]; R["eps_h_optA"] = R["eps_h_opt"]; R["L_optA"] = R["L_opt"]
    end
    R["wall_s"] = time() - t0
    return R
end
