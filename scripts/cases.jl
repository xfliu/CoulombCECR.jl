# Test problems of the paper (Table 6.1). Mesh folders are looked up under
# ENV["CECR_MESH_ROOT"] (see README for the expected layout).

using CoulombCECR

const K3 = 1 / sqrt(3π) * (4 / sqrt(π))^(1 / 3)          # Talenti constant
const C6_BALL6 = 0.642                                     # (5.5), R = 6
const C6_LOCAL1 = sqrt(2) * K3                             # (5.6), r = 1
const R_IN_BALL6 = 5.98795                                # inradius of the 3D hydrogen polyhedron (rounded down)
const A_ball = (8π / 3)^(2 / 3) * C6_BALL6^2
const A_loc = (8π / 3)^(2 / 3) * C6_LOCAL1^2

# Lemma 4(a) / 4(b) at ε₀ = 0.5
C0_hydrogen3d(e0) = A_ball / e0 + e0
function C0_h2plus3d(e0; d = 4.0)
    ρ = e0 / A_loc
    @assert ρ <= d / 2
    return e0 + 1 / ρ + 1 / (d - ρ)
end

const CASES = Dict(
    "2d_one_centre" => CaseSpec(name = "2d_one_centre", dim = 2,
        bdry_dist = [5.0], rho_max = [5.0],
        centers = reshape([0.0, 0.0], 2, 1), charges = [1.0],
        eps = 0.55, Ceps = 12.0, sigma = 12.5, eps0 = 0.5, C0 = 10.99,
        S = 1.466, p = 4 / 3, on_boundary = x -> abs(abs(x[1]) - 5) < 1e-9 || abs(abs(x[2]) - 5) < 1e-9),
    "2d_two_centre" => CaseSpec(name = "2d_two_centre", dim = 2,
        bdry_dist = [5.0, 5.0], rho_max = [2.0, 2.0],
        centers = [-2.0 2.0; 0.0 0.0], charges = [1.0, 1.0],
        eps = 0.55, Ceps = 12.0, sigma = 12.5, eps0 = 0.5, C0 = 11.54,
        S = 1.462, p = 4 / 3, on_boundary = x -> abs(abs(x[1]) - 7) < 1e-9 || abs(abs(x[2]) - 5) < 1e-9),
    "3d_hydrogen" => CaseSpec(name = "3d_hydrogen", dim = 3,
        # The meshed domain is the polyhedron P inscribed in B(0,6) (icosphere
        # boundary, vertices on |x| = 6); its inradius is R_IN_BALL6 = 5.98795 (rounded
        # down). The radial form bound uses the ball B(0, R_IN_BALL6) in P, and the
        # patch term of d_h the local embedding constant (patch in B(0,1), B(0,2) in P).
        bdry_dist = [R_IN_BALL6], rho_max = [R_IN_BALL6], domain_is_ball = false,
        centers = reshape([0.0, 0.0, 0.0], 3, 1), charges = [1.0],
        eps = 0.6, Ceps = 5.0, sigma = 5.4, eps0 = 0.5, C0 = C0_hydrogen3d(0.5),
        S = C6_LOCAL1, p = 1.5, per_centre = true, r_loc = 1.0,
        neumann_p1 = true, on_boundary = x -> sqrt(sum(abs2, x)) > 6 - 1e-6),
    "3d_h2plus" => CaseSpec(name = "3d_h2plus", dim = 3,
        bdry_dist = [6.0, 6.0], rho_max = [2.0, 2.0],
        centers = [-2.0 2.0; 0.0 0.0; 0.0 0.0], charges = [1.0, 1.0],
        eps = 0.6, Ceps = 5.0, sigma = 5.4, eps0 = 0.5, C0 = C0_h2plus3d(0.5),
        S = C6_LOCAL1, p = 1.5, per_centre = true, r_loc = 1.0,
        on_boundary = x -> abs(abs(x[1]) - 8) < 1e-9 || abs(abs(x[2]) - 6) < 1e-9 || abs(abs(x[3]) - 6) < 1e-9),
)

# mesh folders used for the paper tables (relative to CECR_MESH_ROOT)
const MESHES = Dict(
    "2d_one_centre" => ["case7_2d_hydrogen_square/meshes/square_h040_opt",
                        "case7_2d_hydrogen_square/meshes/square_h020_opt",
                        "case7_2d_hydrogen_square/meshes/square_h010_opt"],
    "2d_two_centre" => ["case8_2d_h2plus_box/meshes/box_h040_opt",
                        "case8_2d_h2plus_box/meshes/box_h020_opt"],
    "3d_hydrogen"   => ["case5_3d_hydrogen_cube/meshes_R6_ball/ball_R6_h140",
                        "case5_3d_hydrogen_cube/meshes_R6_ball/ball_R6_h120",
                        "case5_3d_hydrogen_cube/meshes_R6_ball/ball_R6_h100",
                        "case5_3d_hydrogen_cube/meshes_R6_ball/ball_R6_h080"],
    "3d_h2plus"     => ["case6_3d_h2plus_box/meshes_R6/box_R6_h100_opt"],
)

# analytic continuous bound C0(ε0) for V^- (Lemma 4) per case, for other ε0 than spec.eps0
const C0FUN = Dict("3d_hydrogen" => C0_hydrogen3d, "3d_h2plus" => C0_h2plus3d)

"copy of a CaseSpec with some fields replaced"
with_fields(spec::CaseSpec; kw...) =
    CaseSpec(; (f => (haskey(kw, f) ? kw[f] : getfield(spec, f)) for f in fieldnames(CaseSpec))...)

# Continuous form bounds for V^- certified on a fine mesh (scripts/certify_continuous.jl);
# appended to the case specs when the file exists.
using JSON
function with_transfer(spec::CaseSpec, resdir)
    fn = joinpath(resdir, "contbound_$(spec.name).json")
    isfile(fn) || return spec
    B = JSON.parsefile(fn)
    cb = [(Float64(b["eps_cont"]), Float64(b["C_cont"])) for b in B["bounds"]]
    return CaseSpec(; (f => getfield(spec, f) for f in fieldnames(CaseSpec) if f != :cont_bounds)...,
                    cont_bounds = cb)
end
