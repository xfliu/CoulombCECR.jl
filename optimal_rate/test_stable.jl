using Test, LinearAlgebra, SparseArrays, JSON, CoulombCECR, VFEM
include(joinpath(@__DIR__,"stable_fem.jl"))
const OF=OptimalRateFEM
const CC=CoulombCECR

function one_triangle(P)
    edges=[1 2;1 3;2 3]
    vm=VFEM.Mesh2D(Matrix{Float64}(P),reshape([1,2,3],1,3),edges,edges,[1,2,3],reshape([3,2,1],1,3),3,1,3,3)
    return vm,SimplexMesh(vm)
end
function local_forms(P)
    vm,m=one_triangle(P)
    return OF.cecr_parts(vm,m,zeros(1))
end

metrics=Dict{String,Any}()
@testset "Optimal-rate numerical validation" begin
    @testset "ECR duality, constant mode and positive mass" begin
        for P in ([0. 0.;1. 0.;.25 .75],[0. 0.;1. .1;-.2 .5])
            Y=(P .- P[1:1,:])./maximum(norm(P[i,:]-P[j,:]) for i in 1:3 for j in 1:3)
            C=VFEM._local_ecr_bernstein_coefficients(Y,Float64.(VFEM._ECR_C_CR))
            D=vcat([0. 1/3 1/3 0 1/3 0;1/3 0 1/3 0 0 1/3;1/3 1/3 0 1/3 0 0],fill(1/6,1,6))
            @test norm(D*C-I)<1e-13
            K,M,_,_=local_forms(P)
            @test norm(K*ones(4))<1e-12
            @test minimum(eigvals(Symmetric(Matrix(M))))>0
            @test abs(sum(M)-abs(det(hcat(P[2,:]-P[1,:],P[3,:]-P[1,:])))/2)<1e-13
        end
    end
    @testset "Translation and isotropic scaling" begin
        P=[0. 0.;1. 0.;.25 .75]
        K0,M0,_,_=local_forms(P)
        worstK=0.;worstM=0.
        for scale in (1.,2.0^-20,2.0^-35,2.0^-40), center in (-2.,0.,2.)
            Q=scale.*P .+ [center 0.]
            K,M,_,_=local_forms(Q)
            ek=norm(K-K0)/norm(K0);em=norm(M/scale^2-M0)/norm(M0)
            worstK=max(worstK,ek);worstM=max(worstM,em)
            @test ek<1e-12
            @test em<1e-12
        end
        metrics["tiny_translation_stiffness_relative_error"]=worstK
        metrics["tiny_translation_mass_relative_error"]=worstM
    end
    @testset "Agreement with original VFEM on an ordinary mesh" begin
        vm,_=VFEM.mesh2d_triangle_uniform(5)
        m=SimplexMesh(vm);c=fill(-.2,CC.nelem(m));sig=2.
        K,M,R,P=OF.cecr_parts(vm,m,c)
        Aref,Mref=CC.cecr_matrices(vm,c.+sig)
        da=norm(K+R+sig*P-Aref)/norm(Aref);dm=norm(M-Mref)/norm(Mref)
        metrics["ordinary_vfem_stiffness_relative_error"]=da
        metrics["ordinary_vfem_mass_relative_error"]=dm
        @test da<1e-11
        @test dm<1e-11
        lambda,info=OF.eigs(K+R+sig*P,M,3)
        dense=eigvals(Symmetric(Matrix(K+R+sig*P)),Symmetric(Matrix(M)))[1:3]
        @test lambda≈dense rtol=1e-10 atol=1e-11
        metrics["sparse_dense_eigenvalue_max_difference"]=maximum(abs.(lambda.-dense))
        @test maximum(info.backward_residuals)<1e-9
    end
    @testset "Singular quadrature and center-relative evaluation" begin
        worst=0.
        for scale in (1.,2.0^-35), center in (-2.,0.,2.)
            vm,m=one_triangle(scale.*[0. 0.;1. 0.;0. 1.] .+ [center 0.])
            pot=Coulomb(reshape([center,0.],2,1),[1.])
            c=OF.averages(pot,m)[1]
            exact=-2sqrt(2)*asinh(1)/scale
            err=abs(c/exact-1);worst=max(worst,err)
            @test err<1e-10
            qs=CC.QuadSet(2;n_sing=32,n_near=20,n_far=10)
            ch=OF.averages(pot,m;qs=qs)[1]
            @test c≈ch rtol=1e-10
            K,M,W=OF.p1_parts(pot,m)
            Kh,Mh,Wh=OF.p1_parts(pot,m;qs=qs)
            @test norm(W-Wh)/norm(Wh)<1e-9
        end
        metrics["singular_cell_average_analytic_relative_error"]=worst
    end
    @testset "All generated meshes are conforming with shrinking core" begin
        reps=filter(x->startswith(basename(x),"2d_") && isfile(joinpath(x,"mesh_report.json")),readdir(joinpath(@__DIR__,"meshes");join=true))
        @test length(reps)>=8
        for dir in reps
            r=JSON.parsefile(joinpath(dir,"mesh_report.json"))
            @test r["interior_free_edges"]==0
            @test r["nonmanifold_edges"]==0
            @test r["nodes_outside"]==0
            @test r["min_area"]>0
            @test r["min_angle_deg"]>10.
            @test r["theta_off_core"]<1.
            @test r["core_rmax"]≈r["beta"]*r["h"]^4 rtol=2e-5
        end
    end
end
metrics["status"]="passed"
open(joinpath(@__DIR__,"results","validation.json"),"w") do io
    JSON.print(io,metrics,2)
end
println(JSON.json(metrics))
