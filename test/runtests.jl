using Test, CoulombCECR, LinearAlgebra, SparseArrays
using VFEM: mesh2d_triangle_uniform
import VFEM

@testset "CoulombCECR" begin
    @testset "Gauss–Legendre / collapsed rules" begin
        x, w = gauss_legendre01(6)
        @test sum(w) ≈ 1
        @test sum(w .* x .^ 11) ≈ 1 / 12
        for D in (2, 3), q in (1, 2, 3)
            r = collapsed_rule(D, 6; q = q)
            @test sum(r.w) ≈ 1
            # ∫_K λ_1 dx / |K| = 1/(D+1)
            @test sum(r.w .* r.bary[:, 2]) ≈ 1 / (D + 1) atol = 1e-12
        end
    end

    @testset "singular cell average, 2D right triangle" begin
        # ∫_K 1/|x| over K = conv{0,(1,0),(0,1)} equals sqrt(2)·asinh(1)
        X = [0.0 1.0 0.0; 0.0 0.0 1.0]; T = reshape([1, 2, 3], 3, 1)
        m = CoulombCECR.SimplexMesh(X, T)
        pot = Coulomb(reshape([0.0, 0.0], 2, 1), [1.0])
        c = cell_averages(pot, m)
        @test c[1] ≈ -sqrt(2) * asinh(1) / 0.5 rtol = 1e-8
    end

    @testset "singular cell average, 3D corner tetrahedron" begin
        # ∫ 1/|x| over the unit corner tetrahedron, reference by heavy quadrature
        X = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]; T = reshape([1, 2, 3, 4], 4, 1)
        m = CoulombCECR.SimplexMesh(Float64.(X), T)
        pot = Coulomb(reshape([0.0, 0.0, 0.0], 3, 1), [1.0])
        c1 = cell_averages(pot, m)
        qs = CoulombCECR.QuadSet(3; n_sing = 30)
        c2 = cell_averages(pot, m; qs = qs)
        @test c1[1] ≈ c2[1] rtol = 1e-8
    end

    @testset "point–simplex distance" begin
        X = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]; T = reshape([1, 2, 3, 4], 4, 1)
        m = CoulombCECR.SimplexMesh(Float64.(X), T)
        @test CoulombCECR.dist_point_simplex(m, 1, [1.0, 1.0, 1.0]) ≈ (3 - 1) / sqrt(3)
        @test CoulombCECR.dist_point_simplex(m, 1, [0.1, 0.1, 0.1]) == 0.0
        @test CoulombCECR.dist_point_simplex(m, 1, [-1.0, 0.0, 0.0]) ≈ 1.0
    end

    @testset "eigensolver vs dense" begin
        n = 200
        A = spdiagm(-1 => -ones(n - 1), 0 => 2ones(n), 1 => -ones(n - 1))
        M = spdiagm(0 => ones(n))
        λ, info = smallest_eigs(A, M, 3; shift = -0.1)
        λd = eigvals(Matrix(A))[1:3]
        @test λ ≈ λd rtol = 1e-9
    end

    @testset "P1 Coulomb matrix vs VFEM exact singular moments (2D)" begin
        vm, _ = mesh2d_triangle_uniform(8)
        m = CoulombCECR.SimplexMesh(vm)
        a = m.X[:, 1 + size(m.X, 2) ÷ 2]                  # a mesh vertex
        pot = Coulomb(reshape(a, 2, 1), [1.0])
        K, M, W = CoulombCECR.p1_matrices(m; pot = pot)
        Dex = VFEM._assemble_Dh_2d(vm, reshape(a, 1, 2), [1.0])   # ∫ φ_i φ_j / |x-a|
        @test norm(Matrix(W + Dex)) / norm(Matrix(Dex)) < 1e-5
    end

    @testset "CECR on a uniform square: Laplace Neumann eigenvalues" begin
        vm, _ = mesh2d_triangle_uniform(16)
        m = CoulombCECR.SimplexMesh(vm)
        c = zeros(CoulombCECR.nelem(m))
        mu, _ = cecr_eigs(vm, c, 1.0, 3)          # σ = 1: eigenvalues 1 + λ_N
        Ch = interp_constant(m)
        L = cecr_lower(mu, 1.0, Ch, 0.0, 0.5)
        @test mu[1] ≈ 1.0 atol = 1e-8              # constant mode
        @test all(L .<= mu .- 1.0 .+ 1e-12)
    end

    @testset "radial Neumann ground state" begin
        r3 = radial_neumann_ground(3, 1.0, 6.0)
        @test r3.nu ≈ -0.3045981336 atol = 1e-9
        @test r3.err < 1e-9
        r2 = radial_neumann_ground(2, 1.0, 20.0)
        @test r2.nu ≈ -1.0 atol = 1e-6            # whole-plane 2D hydrogen: -Z²
        # scaling ν(Z, ρ) = ν(Zρ, 1)/ρ²
        @test radial_neumann_ground(3, 2.0, 3.0).nu ≈ radial_neumann_ground(3, 6.0, 1.0).nu / 9 rtol = 1e-10
        # Neumann ground state lies below the whole-space value -Z²/4 (3D) only by the boundary effect
        @test radial_neumann_ground(3, 1.0, 30.0).nu ≈ -0.25 atol = 1e-8
    end

    @testset "decoupled form bound (reference values)" begin
        C2 = [-2.0 2.0; 0.0 0.0; 0.0 0.0]
        b = decoupled_form_bound(C2, [1.0, 1.0], 0.6, [1.95, 1.95]; bdry_dist = [6.0, 6.0],
                                 optimise = false)
        @test b.C ≈ 1.3168 atol = 1e-4
        bo = decoupled_form_bound(C2, [1.0, 1.0], 0.6, [2.0, 2.0]; bdry_dist = [6.0, 6.0])
        @test bo.C <= b.C + 1e-12                    # optimisation over radii
        bh = decoupled_form_bound(zeros(3, 1), [1.0], 0.6, [6.0]; bdry_dist = [6.0],
                                  domain_is_ball = true)
        @test bh.C ≈ 0.4218 atol = 1e-4
        @test bh.C ≈ -0.6 * radial_neumann_ground(3, 1 / 0.6, 6.0).nu
        @test_throws ArgumentError decoupled_form_bound(C2, [1.0, 1.0], 0.6, [2.5, 2.5];
                                                        bdry_dist = [6.0, 6.0], optimise = false)
        # P1 Galerkin check of the 3D ball bound is replaced here by a 2D sanity check:
        # on B(0,ρ) ⊂ R², ε=1: C_V = -ν ≥ the whole-plane value Z² = 1
        b2 = decoupled_form_bound(zeros(2, 1), [1.0], 1.0, [5.0]; bdry_dist = [5.0],
                                  domain_is_ball = true)
        @test b2.C >= 1.0 - 1e-12
    end

    @testset "radial bound dominates the discrete Galerkin C_ε (2D square)" begin
        # (V^- u,u) ≤ ε‖∇u‖² + C_V‖u‖² must hold for P1 functions: compare with
        # the P1 Galerkin value of the optimal constant for the exact V^- (lower estimate)
        vm, _ = mesh2d_triangle_uniform(16)        # unit square [0,1]²
        m = CoulombCECR.SimplexMesh(vm)
        a = [0.5, 0.5]
        pot = Coulomb(reshape(a, 2, 1), [1.0])
        K, M, W = CoulombCECR.p1_matrices(m; pot = pot)
        eps = 0.6
        λ, _ = smallest_eigs(eps * K + W, M, 1; shift = -60.0)   # W = -∫ V^- φφ
        Cgal = -λ[1]
        rb = decoupled_form_bound(reshape(a, 2, 1), [1.0], eps, [0.5]; bdry_dist = [0.5])
        @test rb.C >= Cgal
    end

    @testset "interpolation of μ^σ - σ in σ" begin
        S = [(1.0, [0.0, 1.0]), (3.0, [2.0, 3.0])]
        @test interp_shifted(S, 2.0) ≈ [1.0, 2.0]
        @test interp_shifted(S, 4.0) ≈ [3.0, 4.0]          # linear extrapolation
        @test interp_shifted([(1.0, [5.0])], 7.0) ≈ [5.0]
    end
end
