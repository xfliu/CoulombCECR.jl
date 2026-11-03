# Full vs symmetry-sector CECR eigenvalues on a mirror-symmetric H2+ mesh.
# usage: julia -t N --project=. scripts/check_symmetry.jl <mode> <mesh-dir> <out.json> [sigmas] [k]
#   mode = full   : cecr_eigs (all DOFs)
#        = sym    : cecr_eigs_symmetric over all 8 sectors of Z_2^3
#        = sector : fully symmetric sector only (cost probe), plus direct_dh timing
#   sigmas: comma-separated list (default 5.4,2.1546), k default 3
include(joinpath(@__DIR__, "cases.jl"))
using JSON

todict(x) = x
todict(x::NamedTuple) = Dict(string(k) => todict(v) for (k, v) in pairs(x))
todict(x::AbstractVector) = [todict(v) for v in x]

function main(args)
    mode, dir, out = args[1:3]
    sigmas = length(args) >= 4 ? parse.(Float64, split(args[4], ",")) : [5.4, 2.1546]
    k = length(args) >= 5 ? parse(Int, args[5]) : 3
    spec = CASES["3d_h2plus"]
    R = Dict{String,Any}("mode" => mode, "mesh" => abspath(dir), "nthreads" => Threads.nthreads(), "k" => k)
    t = time(); vm = load_vfem_mesh(dir, 3); m = SimplexMesh(vm); R["t_load"] = time() - t
    pot = Coulomb(spec.centers, spec.charges)
    t = time(); c = cell_averages(pot, m); R["t_ch"] = time() - t
    R["n_elem"] = vm.NumElt; R["NumF"] = vm.NumF; R["ndof"] = vm.NumF + vm.NumElt
    R["n_node"] = vm.NumNode; R["h_max"] = maximum(m.hK)
    R["maxrss_before_solve_GB"] = Sys.maxrss() / 2^30
    println("loaded $(dir): ne=$(vm.NumElt) ndof=$(R["ndof"]) t_load=$(R["t_load"]) t_ch=$(R["t_ch"])"); flush(stdout)
    runs = Any[]
    for σ in sigmas
        GC.gc(); t = time()
        if mode == "full"
            mu, inf = cecr_eigs(vm, c, σ, k)
            push!(runs, Dict("sigma" => σ, "mu" => mu, "mu_minus_sigma" => mu .- σ, "t_solve" => time() - t,
                             "info" => todict(inf), "maxrss_GB" => Sys.maxrss() / 2^30))
        else
            secs = mode == "sector" ? [[1, 1, 1]] : nothing
            mu, inf = cecr_eigs_symmetric(vm, c, σ, k, [1, 2, 3]; sectors = secs, verbose = true)
            push!(runs, Dict("sigma" => σ, "mu" => mu, "mu_minus_sigma" => mu .- σ, "t_solve" => time() - t,
                             "info" => todict(inf), "maxrss_GB" => Sys.maxrss() / 2^30))
        end
        println("sigma=$σ mu=$(runs[end]["mu"]) t=$(runs[end]["t_solve"])"); flush(stdout)
    end
    R["runs"] = runs
    if mode == "sector"
        t = time()
        dh = direct_dh(pot, m, c, spec.S, spec.p; per_centre = spec.per_centre, r_loc = spec.r_loc)
        R["t_dh"] = time() - t; R["d_h"] = dh.dh; R["d_h_patch"] = dh.patch; R["d_h_off"] = dh.off
        println("d_h=$(dh.dh) t_dh=$(R["t_dh"])")
    end
    R["maxrss_GB"] = Sys.maxrss() / 2^30
    open(out, "w") do io
        JSON.print(io, R, 2)
    end
    println("DONE_MARKER")
end
main(ARGS)
