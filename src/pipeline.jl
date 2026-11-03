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
                                      # (max_rungs = 0: no ladders at all)
    cont_bounds::Vector{Tuple{Float64,Float64}} = Tuple{Float64,Float64}[]
                                      # extra continuous bounds (ε, C) for V^-
                                      # (e.g. certified on a finer mesh)
    symmetry::Vector{Int} = Int[]     # coordinate planes x_i = 0 of mirror symmetry of
                                      # domain, potential and mesh (empty: none)
    # ---- radial (Neumann-decoupling) form bound, src/radialbound.jl ----
    bdry_dist::Vector{Float64} = Float64[]   # dist(a_j, ∂Ω) per centre
    rho_max::Vector{Float64} = Float64[]     # maximal admissible ball radius per centre
                                             # (empty: no radial bound)
    domain_is_ball::Bool = false             # Ω = B(a_1, rho_max[1])
    eps_grid::Vector{Float64} = collect(0.30:0.025:0.95)   # per-k choice of ε
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
    radial_form_bound(spec, eps) -> NamedTuple

Continuous bound (V^- u,u) ≤ eps‖∇u‖² + C_V‖u‖² of `decoupled_form_bound`
with the geometry data of `spec` (radii optimised).
"""
radial_form_bound(spec::CaseSpec, eps::Real) =
    decoupled_form_bound(spec.centers, spec.charges, eps, spec.rho_max;
                         bdry_dist = spec.bdry_dist, domain_is_ball = spec.domain_is_ball)

"""
    interp_shifted(solves, s) -> Vector

Piecewise-linear interpolation (linear extrapolation beyond the end points)
in σ of μ^σ - σ from `solves = [(σ_j, μ^σ_j - σ_j), ...]`.
"""
function interp_shifted(solves, s::Real)
    sig = [x[1] for x in solves]; p = sortperm(sig)
    sig = sig[p]; vals = [solves[i][2] for i in p]
    keep = [1; [j for j in 2:length(sig) if sig[j] - sig[j-1] > 1e-12]]
    sig = sig[keep]; vals = vals[keep]
    length(sig) == 1 && return copy(vals[1])
    j = clamp(searchsortedlast(sig, s), 1, length(sig) - 1)
    θ = (s - sig[j]) / (sig[j+1] - sig[j])
    return (1 - θ) .* vals[j] .+ θ .* vals[j+1]
end

"Theorem 5.4 bound with the choice σ = C + κ, κ = 1 - ε (A_h = Γ_h/(1-ε), ε_h = d_h/κ)."
lower_bound_at(mu_shifted, sigma, Ch, Gam, eps, dh) =
    corrected_lower(cecr_lower(mu_shifted .+ sigma, sigma, Ch, Gam, eps), sigma, dh / (1 - eps))

"""
    run_case(spec, meshdir; verbose=true) -> Dict

Computes every quantity reported in the paper tables for one mesh.
"""
function run_case(spec::CaseSpec, meshdir::AbstractString; verbose::Bool = true)
    t0 = time()
    log(s) = verbose && (println(s); flush(stdout))
    R = Dict{String,Any}("case" => spec.name, "mesh" => basename(rstrip(meshdir, '/')))
    timing = Dict{String,Float64}()
    tic = Ref(time())
    lap(key) = (timing[key] = time() - tic[]; tic[] = time())
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
    lap("load")

    c = cell_averages(pot, m); R["t_ch"] = time() - t0
    Ch = interp_constant(m); Gam = reaction_constant(m, c)
    R["C_h"] = Ch; R["Gamma_h"] = Gam; R["A_h"] = Gam / (1 - spec.eps)
    eigf = (vm_, q_, s_, k_) -> cecr_eigs_case(spec, vm_, q_, s_, k_)
    lap("cell_averages")

    # ---- (S3) perturbation d_h ------------------------------------------
    dh = direct_dh(pot, m, c, spec.S, spec.p; per_centre = spec.per_centre, r_loc = spec.r_loc)
    R["d_h"] = dh.dh; R["d_h_patch"] = dh.patch; R["d_h_off"] = dh.off
    R["rho_opt"] = dh.rho; R["n_patch"] = dh.npatch; R["patch_rmax"] = dh.rmax
    # Lemma 3 admissibility of the analytic (ε, C_ε)
    R["lemma3_ok"] = (spec.eps0 + dh.dh <= spec.eps) && (spec.C0 + dh.dh <= spec.Ceps)
    log(@sprintf("  Gamma_h=%.3e  d_h=%.4e (patch %.3e, off %.3e, %d patch elts)  lemma3=%s",
                 Gam, dh.dh, dh.patch, dh.off, dh.npatch, R["lemma3_ok"]))
    lap("d_h")

    # ---- analytic shift ---------------------------------------------------
    solves = Any[]                       # (σ, μ^σ - σ) of all CECR solves with c_h
    kap = min(1 - spec.eps, spec.sigma - spec.Ceps)
    epsh = dh.dh / kap
    mu, inf1 = cecr_eigs_case(spec, vm, c, spec.sigma, spec.neig)
    push!(solves, (spec.sigma, mu .- spec.sigma))
    Lmu = cecr_lower(mu, spec.sigma, Ch, Gam, spec.eps)
    R["kappa"] = kap; R["eps_h"] = epsh
    R["mu_minus_sigma"] = mu .- spec.sigma; R["L_mu_sigma"] = Lmu
    R["L"] = corrected_lower(Lmu, spec.sigma, epsh); R["cecr_resid"] = inf1.residuals
    R["n_dof_cecr"] = size(cecr_matrices(vm, c)[1], 1)
    log(@sprintf("  mu-sigma=%s  L=%s", string(round.(R["mu_minus_sigma"], digits = 4)),
                 string(round.(R["L"], digits = 4))))
    lap("cecr_analytic")

    # ---- upper bounds -------------------------------------------------------
    shiftP1 = -(spec.C0 + 0.5)           # λ_1 ≥ -C0 by the continuous form bound
    U, infU = p1_bounds(m, pot, spec.neig; bc = :dirichlet, shift = shiftP1)
    R["U_dirichlet"] = U
    if spec.neumann_p1
        UN, _ = p1_bounds(m, pot, spec.neig; bc = :neumann, shift = shiftP1)
        R["U_neumann"] = UN
    end
    log("  U_D=" * string(round.(U, digits = 4)))
    lap("p1_upper")

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
        (ef < spec.eps && spec.max_rungs > 0) || continue
        for nm in 0:spec.max_rungs
            seq = [ef + (spec.eps - ef) * k / (nm + 1) for k in 0:nm+1]
            try
                lad = ceps_ladder(vm, m, c, seq, Cf; eigfun = eigf)
                push!(cands, (C = lad[end].C, how = "$(src):ladder$(nm+1)", eps_start = ef,
                              rungs = [(eps = r.eps, C = r.C) for r in lad]))
            catch err
                log("  ladder from $(src) with $(nm) mids failed: $(err)")
            end
        end
    end
    # radial (Neumann-decoupling) continuous bound at ε - d_h, transferred by Lemma 3
    has_radial = !isempty(spec.rho_max)
    if has_radial && spec.eps - dh.dh > 0
        rb = radial_form_bound(spec, spec.eps - dh.dh)
        push!(cands, (C = rb.C + dh.dh, how = "radial:direct", eps_start = spec.eps - dh.dh,
                      rungs = Any[]))
        R["radial_C_V"] = rb.C; R["radial_rho"] = rb.rho; R["radial_nu"] = rb.nu
        R["radial_nu_err"] = rb.nu_err; R["radial_s_rem"] = isfinite(rb.s_rem) ? rb.s_rem : nothing   # -Inf: no remainder
        R["C_cert_radial"] = rb.C + dh.dh
        # λ_1 ≥ -C_V(1) for the continuous problem (form bound with ε = 1)
        R["L1_trivial_radial"] = -radial_form_bound(spec, 1.0).C
        log(@sprintf("  radial: C_V(ε-d_h)=%.4f (rho=%s, s_rem=%.4f, ν err %.1e)  C=%.4f;  -C_V(1)=%.4f",
                     rb.C, string(round.(rb.rho, digits = 3)), rb.s_rem, maximum(rb.nu_err),
                     rb.C + dh.dh, R["L1_trivial_radial"]))
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
    lap("ceps")

    # ---- optimised shift with the certified constant -------------------------
    tags = Any[("opt", Ccert), ("optP1", R["C_p1_galerkin"])]
    isfinite(Can) && Can != Ccert && push!(tags, ("optA", Can))   # analytic-start-only L_opt
    for (tag, Cv) in tags
        sopt = Cv + (1 - spec.eps)
        kopt = min(1 - spec.eps, sopt - Cv)
        muo, _ = cecr_eigs_case(spec, vm, c, sopt, spec.neig)
        push!(solves, (sopt, muo .- sopt))
        Lmuo = cecr_lower(muo, sopt, Ch, Gam, spec.eps)
        R["sigma_"*tag] = sopt; R["eps_h_"*tag] = dh.dh / kopt
        R["mu_minus_sigma_"*tag] = muo .- sopt
        R["L_"*tag] = corrected_lower(Lmuo, sopt, dh.dh / kopt)
    end
    log("  L_opt=" * string(round.(R["L_opt"], digits = 4)) *
        "  (with P1 C: " * string(round.(R["L_optP1"], digits = 4)) * ")")
    if !haskey(R, "L_optA") && isfinite(Can)
        R["sigma_optA"] = R["sigma_opt"]; R["eps_h_optA"] = R["eps_h_opt"]; R["L_optA"] = R["L_opt"]
    end
    lap("cecr_opt")

    # ---- per-k choice of ε (radial bound) ------------------------------------
    # For each admissible ε of the grid: C(ε) = C_V(ε - d_h) + d_h (Lemma 3),
    # σ(ε) = C(ε) + 1 - ε, κ = 1 - ε. L_k(ε) is predicted from μ^σ - σ
    # interpolated in σ; each k gets its own ε_k (each L_k is a separate valid
    # bound) and is then recomputed exactly at σ(ε_k). At ε = spec.eps the
    # constant is min(radial, C_cert), so L_best ≥ L_opt up to the prediction error.
    if has_radial
        grid = Any[]
        for e in sort(unique([spec.eps_grid; spec.eps]))
            (e - dh.dh > 0.05 && dh.dh / (1 - e) < 0.5) || continue
            rb = radial_form_bound(spec, e - dh.dh)
            Ce = rb.C + dh.dh; how = "radial:direct"
            # at the target ε every certified candidate (ladders, ...) is available
            if abs(e - spec.eps) < 1e-9 && Ccert < Ce
                Ce = Ccert; how = cands[ib].how
            end
            se = Ce + 1 - e
            Lp = lower_bound_at(interp_shifted(solves, se), se, Ch, Gam, e, dh.dh)
            push!(grid, (eps = e, C = Ce, C_V = rb.C, C_how = how, sigma = se, rho = rb.rho,
                         L_pred = Lp))
        end
        if !isempty(grid)
            kbest = [argmax([g.L_pred[k] for g in grid]) for k in 1:spec.neig]
            exact = Dict{Int,Any}()
            for gi in unique(kbest)
                g = grid[gi]
                j = findfirst(x -> abs(x[1] - g.sigma) <= 1e-12, solves)
                ms = if j === nothing
                    mu_e, _ = cecr_eigs_case(spec, vm, c, g.sigma, spec.neig)
                    push!(solves, (g.sigma, mu_e .- g.sigma)); mu_e .- g.sigma
                else
                    solves[j][2]
                end
                exact[gi] = (mu_shifted = ms, L = lower_bound_at(ms, g.sigma, Ch, Gam, g.eps, dh.dh))
            end
            Lb = [exact[kbest[k]].L[k] for k in 1:spec.neig]
            Lpk = [grid[kbest[k]].L_pred[k] for k in 1:spec.neig]
            R["eps_best"] = [grid[i].eps for i in kbest]
            R["C_best"] = [grid[i].C for i in kbest]
            R["C_best_how"] = [grid[i].C_how for i in kbest]
            R["sigma_best"] = [grid[i].sigma for i in kbest]
            R["L_best_pred"] = Lpk
            R["L_best"] = Lb
            R["L_best_pred_minus_exact"] = Lpk .- Lb
            R["perk_solves"] = [Dict("eps" => grid[gi].eps, "sigma" => grid[gi].sigma,
                                     "C" => grid[gi].C, "mu_minus_sigma" => exact[gi].mu_shifted,
                                     "L" => exact[gi].L) for gi in sort(collect(keys(exact)))]
            R["perk_grid"] = [Dict("eps" => g.eps, "C" => g.C, "C_V" => g.C_V, "C_how" => g.C_how, "sigma" => g.sigma,
                                   "rho" => g.rho, "L_pred" => g.L_pred) for g in grid]
            log("  per-k: eps=" * string(R["eps_best"]) * "  L_best=" *
                string(round.(Lb, digits = 4)) * "  pred-exact=" *
                string(round.(R["L_best_pred_minus_exact"], sigdigits = 2)))
        end
    end
    lap("perk")
    R["timing"] = timing
    R["wall_s"] = time() - t0
    return R
end
