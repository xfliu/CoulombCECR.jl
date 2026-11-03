# Mirror-symmetry splitting: sector eigenvalues = full eigenvalues on small
# mirror-symmetric meshes (included from runtests.jl).
using Test, CoulombCECR, LinearAlgebra
using VFEM: mesh_load_from_folder, mesh2d_load_ne
using DelimitedFiles: writedlm

"reflect a simplex mesh (X: nv×D, T: ne×(D+1)) across x_a = 0 for a in axes, merging on-plane nodes"
function reflect_mesh(X::Matrix{Float64}, T::Matrix{Int}, axes)
    for a in axes
        nv = size(X, 1)
        on = X[:, a] .== 0
        newid = zeros(Int, nv); nxt = nv
        for v in 1:nv
            newid[v] = on[v] ? v : (nxt += 1)
        end
        Y = X[.!on, :]; Y[:, a] .*= -1
        X = vcat(X, Y); T = vcat(T, newid[T])
    end
    return X, T
end

"Kuhn (6 tets per cube) mesh of [0,1]^3 with n^3 cubes, reflected to [-1,1]^3"
function sym_cube_mesh(n)
    id(i, j, k) = 1 + i + (n + 1) * (j + (n + 1) * k)
    X = zeros((n + 1)^3, 3)
    for i in 0:n, j in 0:n, k in 0:n
        X[id(i, j, k), :] = [i, j, k] ./ n
    end
    T = zeros(Int, 0, 4); E = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    perms = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    for i in 0:n-1, j in 0:n-1, k in 0:n-1, p in perms
        v0 = [i, j, k]; v1 = v0 + E[p[1]]; v2 = v1 + E[p[2]]; v3 = v2 + E[p[3]]
        T = vcat(T, [id(v0...) id(v1...) id(v2...) id(v3...)])
    end
    return reflect_mesh(X, T, 1:3)
end

"criss-cross mesh of [0,1]^2 (4 triangles per square), reflected to [-1,1]^2"
function sym_square_mesh(n)
    id(i, j) = 1 + i + (n + 1) * j
    X = zeros((n + 1)^2 + n^2, 2)
    for i in 0:n, j in 0:n
        X[id(i, j), :] = [i, j] ./ n
    end
    T = zeros(Int, 0, 3); c0 = (n + 1)^2
    for i in 0:n-1, j in 0:n-1
        c = c0 + 1 + i + n * j; X[c, :] = ([i, j] .+ 0.5) ./ n
        a, b, d, e = id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)
        T = vcat(T, [a b c; b d c; d e c; e a c])
    end
    return reflect_mesh(X, T, 1:2)
end

function write_mesh(X, T)
    dir = mktempdir()
    writedlm(joinpath(dir, "nodes.dat"), X, ' '); writedlm(joinpath(dir, "elements.dat"), T, ' ')
    return dir
end

@testset "mirror-symmetry splitting" begin
    @testset "3D, Z_2^3 and subgroups" begin
        X, T = sym_cube_mesh(2)
        @test size(T, 1) == 8 * 48
        vm = mesh_load_from_folder(write_mesh(X, T))
        m = SimplexMesh(vm)
        pot = Coulomb([-0.5 0.5; 0.0 0.0; 0.0 0.0], [1.0, 1.0])
        c = cell_averages(pot, m)
        for σ in (5.4, 3.0)
            mu, _ = cecr_eigs(vm, c, σ, 8)
            ms, inf = cecr_eigs_symmetric(vm, c, σ, 8, [1, 2, 3])
            @test ms ≈ mu rtol = 1e-9
            @test maximum(inf.invariance_A) < 1e-12 && maximum(inf.invariance_M) < 1e-12
            @test sum(s.dim for s in inf.sectors) == vm.NumF + vm.NumElt
            @test inf.sector[1] == "x1+,x2+,x3+"
            for ax in ([2], [1, 3])
                ms2, _ = cecr_eigs_symmetric(vm, c, σ, 8, ax)
                @test ms2 ≈ mu rtol = 1e-9
            end
        end
        # one sector only: its smallest eigenvalue is the global ground state here
        ms1, inf1 = cecr_eigs_symmetric(vm, c, 5.4, 1, [1, 2, 3]; sectors = [[1, 1, 1]])
        mu, _ = cecr_eigs(vm, c, 5.4, 1)
        @test ms1[1] ≈ mu[1] rtol = 1e-9
        @test length(inf1.sectors) == 1
        # an asymmetric potential is detected through the orbit average of c
        pot2 = Coulomb(reshape([0.5, 0.0, 0.0], 3, 1), [1.0])
        c2 = cell_averages(pot2, m)
        _, inf2 = cecr_eigs_symmetric(vm, c2, 5.4, 2, [1])
        @test inf2.c_orbit_maxchange > 1e-3
    end
    @testset "2D criss-cross square" begin
        X, T = sym_square_mesh(3)
        vm = mesh2d_load_ne(write_mesh(X, T))
        m = SimplexMesh(vm)
        pot = Coulomb([-1/3 1/3; 0.0 0.0], [1.0, 1.0])
        c = cell_averages(pot, m)
        mu, _ = cecr_eigs(vm, c, 12.5, 6)
        ms, inf = cecr_eigs_symmetric(vm, c, 12.5, 6, [1, 2])
        @test ms ≈ mu rtol = 1e-9
        @test maximum(inf.invariance_A) < 1e-12
    end
    @testset "non-symmetric mesh is rejected" begin
        X, T = sym_cube_mesh(1)
        X2 = copy(X); X2[1, :] .+= [0.1, 0.05, 0.02]          # move the node at the origin
        vm = mesh_load_from_folder(write_mesh(X2, T))
        @test_throws ErrorException cecr_eigs_symmetric(vm, zeros(vm.NumElt), 1.0, 2, [1])
    end
end
