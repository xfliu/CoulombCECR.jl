# Certify continuous form bounds  (V^- u,u) ≤ ε_c ‖∇u‖² + C_c ‖u‖²  on Ω
# using a FINE mesh: Sec. 4.4 ladder for c_f^- plus Lemma 3 (d_f).
#
# usage: julia --project=. scripts/certify_continuous.jl <case> <meshdir-rel|abs> <eps_f,...> [outdir] [eps0]
#   eps0 (optional): start the fine-mesh ladders from the analytic bound (eps0, C0FUN[case](eps0))
#   instead of (spec.eps0, spec.C0); needed when eps_f must be < spec.eps0 + d_f.
include(joinpath(@__DIR__, "cases.jl"))
using JSON, Printf
const CC = CoulombCECR

case = ARGS[1]; rel = ARGS[2]
eps_list = parse.(Float64, split(ARGS[3], ","))
outdir = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "..", "results")
spec = CASES[case]
haskey(ENV, "CECR_MAX_RUNGS") && (spec = with_fields(spec; max_rungs = parse(Int, ENV["CECR_MAX_RUNGS"])))
if length(ARGS) >= 5
    e0 = parse(Float64, ARGS[5]); spec = with_fields(spec; eps0 = e0, C0 = C0FUN[case](e0))
    @printf("analytic start: eps0=%.4f C0=%.4f\n", spec.eps0, spec.C0); flush(stdout)
end
dir = joinpath(get(ENV, "CECR_MESH_ROOT", ""), rel)
vm = load_vfem_mesh(dir, spec.dim); m = SimplexMesh(vm)
pot = Coulomb(spec.centers, spec.charges)
c = cell_averages(pot, m)
dh = direct_dh(pot, m, c, spec.S, spec.p; per_centre = spec.per_centre, r_loc = spec.r_loc)
@printf("fine mesh %s: ne=%d hmax=%.4f d_f=%.4e\n", rel, CC.nelem(m), maximum(m.hK), dh.dh); flush(stdout)
ef0 = spec.eps0 + dh.dh; Cf0 = spec.C0 + dh.dh
bounds = Any[]
for ef in eps_list
    ef > ef0 || continue
    best = Inf; how = ""
    for nm in 0:spec.max_rungs+1
        seq = [ef0 + (ef - ef0) * k / (nm + 1) for k in 0:nm+1]
        try
            lad = ceps_ladder(vm, m, c, seq, Cf0)
            if lad[end].C < best
                best = lad[end].C; how = "ladder$(nm+1)"
            end
        catch err
            println("  ladder failed: ", err)
        end
    end
    Cp1 = ceps_p1_galerkin(m, c, ef; shift = -(Cf0 + 0.5))
    @printf("  eps_f=%.4f  C_f(cert)=%.4f [%s]  C_f(P1 Galerkin)=%.4f  ->  continuous (%.4f, %.4f)\n",
            ef, best, how, Cp1, ef + dh.dh, best + dh.dh); flush(stdout)
    push!(bounds, Dict("eps_f" => ef, "C_f" => best, "how" => how, "C_f_p1" => Cp1,
                       "eps_cont" => ef + dh.dh, "C_cont" => best + dh.dh))
end
mkpath(outdir)
open(joinpath(outdir, "contbound_$(case).json"), "w") do io
    JSON.print(io, Dict("case" => case, "fine_mesh" => rel, "d_f" => dh.dh, "eps0" => spec.eps0, "C0" => spec.C0,
                        "n_elem" => CC.nelem(m), "h_max" => maximum(m.hK), "bounds" => bounds), 2)
end
println("DONE_MARKER")
