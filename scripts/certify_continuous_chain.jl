# Certify continuous form bounds  (V^- u,u) <= eps_c |grad u|^2 + C_c |u|^2  on Omega
# using a FINE mesh, with ONE chained Sec. 4.4 ladder through all requested eps_f
# (cheaper variant of certify_continuous.jl: every rung is a valid bound, so a
# single ladder ef0 -> (n_pre intermediate rungs) -> eps_1 -> eps_2 -> ... yields a
# certified C_f for every target; certify_continuous.jl instead runs 5 separate
# ladders per target, i.e. ~15 large eigensolves per target).
#
# usage: julia --project=. scripts/certify_continuous_chain.jl <case> <meshdir> <eps_f,...> [outdir] [n_pre]
# Output: <outdir>/contbound_<case>.json (same format as certify_continuous.jl).
include(joinpath(@__DIR__, "cases.jl"))
using JSON, Printf
const CC = CoulombCECR

case = ARGS[1]; rel = ARGS[2]
eps_list = sort(parse.(Float64, split(ARGS[3], ",")))
outdir = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "..", "results")
npre = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
spec = CASES[case]
dir = isabspath(rel) ? rel : joinpath(get(ENV, "CECR_MESH_ROOT", ""), rel)
t0 = time()
vm = load_vfem_mesh(dir, spec.dim); m = SimplexMesh(vm)
pot = Coulomb(spec.centers, spec.charges)
c = cell_averages(pot, m)
dh = direct_dh(pot, m, c, spec.S, spec.p; per_centre = spec.per_centre, r_loc = spec.r_loc)
nfree = interior_free_facets(m, spec.on_boundary)
@printf("fine mesh %s: ne=%d hmax=%.4f d_f=%.4e interior_free_facets=%d (%.0fs)\n", rel, CC.nelem(m),
        maximum(m.hK), dh.dh, nfree, time() - t0); flush(stdout)
nfree == 0 || @warn "fine mesh is not conforming / has holes: continuous bound refers to the mesh domain"
ef0 = spec.eps0 + dh.dh; Cf0 = spec.C0 + dh.dh
targets = filter(>(ef0), eps_list)
seq = vcat([ef0 + (targets[1] - ef0) * k / (npre + 1) for k in 0:npre], targets)
function run_chain(vm, m, c, seq, Cf0)
    out = Any[(C = Cf0, eps = seq[1])]
    for j in 2:length(seq)
        t1 = time()
        r = certify_ceps_rung(vm, m, c, seq[j], seq[j-1], out[end].C)
        push!(out, r)
        @printf("  rung %d: eps=%.4f  C=%.6f  nu_low=%.6f  A=%.3e  (%.0fs)\n", j - 1, r.eps, r.C,
                r.nu_low, r.A, time() - t1); flush(stdout)
    end
    return out
end
lad = run_chain(vm, m, c, seq, Cf0)
function collect_bounds(lad, targets, m, c, dh, Cf0)
    bounds = Any[]
    for ef in targets
        r = lad[findfirst(x -> x.eps == ef, lad)]
        Cp1 = ceps_p1_galerkin(m, c, ef; shift = -(Cf0 + 0.5))
        @printf("  eps_f=%.4f  C_f(cert)=%.6f [chain]  C_f(P1 Galerkin)=%.6f  ->  continuous (%.6f, %.6f)\n",
                ef, r.C, Cp1, ef + dh.dh, r.C + dh.dh); flush(stdout)
        push!(bounds, Dict("eps_f" => ef, "C_f" => r.C, "how" => "chain", "C_f_p1" => Cp1,
                           "eps_cont" => ef + dh.dh, "C_cont" => r.C + dh.dh))
    end
    return bounds
end
bounds = collect_bounds(lad, targets, m, c, dh, Cf0)
mkpath(outdir)
open(joinpath(outdir, "contbound_$(case).json"), "w") do io
    JSON.print(io, Dict("case" => case, "fine_mesh" => rel, "d_f" => dh.dh, "d_f_patch" => dh.patch,
                        "d_f_off" => dh.off, "interior_free_facets" => nfree,
                        "n_elem" => CC.nelem(m), "h_max" => maximum(m.hK), "bounds" => bounds,
                        "chain" => [[r.eps, r.C] for r in lad], "wall_s" => time() - t0), 2)
end
println("DONE_MARKER")
