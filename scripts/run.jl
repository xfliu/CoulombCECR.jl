# usage: julia -t N --project=. scripts/run.jl <case> <level-index | mesh-dir> [outdir]
#   <mesh-dir>: any mesh folder (absolute, or relative to the cwd) instead of a level of MESHES[case]
#   ENV CECR_MAX_RUNGS overrides spec.max_rungs (number of intermediate ε values in the ladders)
include(joinpath(@__DIR__, "cases.jl"))
using JSON

case = ARGS[1]; lev = tryparse(Int, ARGS[2])
outdir = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "..", "results")
root = get(ENV, "CECR_MESH_ROOT", "")
spec = with_transfer(CASES[case], outdir)
haskey(ENV, "CECR_MAX_RUNGS") && (spec = with_fields(spec; max_rungs = parse(Int, ENV["CECR_MAX_RUNGS"])))
dir = lev === nothing ? abspath(ARGS[2]) : joinpath(root, MESHES[case][lev])
R = run_case(spec, dir)
mkpath(outdir)
fn = joinpath(outdir, "$(case)_$(R["mesh"]).json")
open(fn, "w") do io
    JSON.print(io, R, 2)
end
println("WROTE ", fn)
println("DONE_MARKER")
