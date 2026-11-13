# Independent numerical validation of the new 3D optimal-rate kernels.
include(joinpath(@__DIR__,"stable_fem_3d.jl"))
using .OptimalRateFEM3D, .OptimalRateFEM
using CoulombCECR, VFEM, LinearAlgebra, SparseArrays, DelimitedFiles, Test, JSON
const OF3=OptimalRateFEM3D
const CC=CoulombCECR
BLAS.set_num_threads(1)
const diagnostic=Dict{String,Any}()

function loaded_mesh(X,T)
    mktempdir() do d
        writedlm(joinpath(d,"nodes.dat"),X)
        writedlm(joinpath(d,"elements.dat"),T)
        return CC.load_vfem_mesh(d,3)
    end
end

function mean_monomials(P)
    n=size(P,1)
    return [1.;vec(sum(P,dims=1))./n;
            (sum(abs2,P)+sum(abs2,sum(P,dims=1)))/(n*(n+1))]
end

# Independent ECR reconstruction in monomials: no Bernstein moment code.
function monomial_reference(X)
    D=zeros(5,5)
    for i in 1:4
        D[i,:]=mean_monomials(X[setdiff(1:4,[i]),:])
    end
    D[5,:]=mean_monomials(X)
    C=D\Matrix{Float64}(I,5,5)
    @test norm(D*C-I)<2e-13
    rule=CC.collapsed_rule(3,7)
    Q=rule.bary*X
    V=hcat(ones(size(Q,1)),Q,sum(abs2,Q,dims=2))*C
    vol=abs(det(hcat(X[2,:]-X[1,:],X[3,:]-X[1,:],X[4,:]-X[1,:])))/6
    K=zeros(5,5)
    for q in eachindex(rule.w)
        x,y,z=Q[q,:]
        G=[0. 1. 0. 0. 2x;0. 0. 1. 0. 2y;0. 0. 0. 1. 2z]*C
        K .+= vol*rule.w[q].*(G'*G)
    end
    M=vol.*(V'*Diagonal(rule.w)*V)
    return K,M,D,C
end

# Evaluate the actual Bernstein coefficients without VFEM evaluation routines.
function bernstein_values(lambda,C)
    alpha=VFEM.ijkl_list(2)
    B=[2/prod(factorial.(alpha[j,:]))*prod(lambda[i]^alpha[j,i] for i in 1:4)
       for j in axes(alpha,1)]
    return C'*B
end

function independent_duality(Y)
    C=OF3.local_coefficients(Y,VFEM._ecr3d_cr_coeffs(Float64))
    E=zeros(5,5)
    face=CC.collapsed_rule(2,6)
    for i in 1:4
        ids=setdiff(1:4,[i])
        for q in eachindex(face.w)
            lambda=zeros(4);lambda[ids]=face.bary[q,:]
            E[i,:].+=face.w[q].*bernstein_values(lambda,C)
        end
    end
    cell=CC.collapsed_rule(3,6)
    for q in eachindex(cell.w)
        E[5,:].+=cell.w[q].*bernstein_values(cell.bary[q,:],C)
    end
    return norm(E-I)
end

function reflected_cube()
    # An independent construction: all coordinate-order simplices of each
    # reflected octant, then exact tuple merging on the coordinate planes.
    permutations=((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1))
    nodes=NTuple{3,Float64}[];lookup=Dict{NTuple{3,Float64},Int}()
    cells=NTuple{4,Int}[]
    function index(p)
        key=Tuple(Float64.(p))
        return get!(lookup,key) do
            push!(nodes,key);length(nodes)
        end
    end
    for sx in (-1,1),sy in (-1,1),sz in (-1,1),order in permutations
        p=zeros(Int,3);ids=[index(p)]
        for i in order
            p[i]=(sx,sy,sz)[i];push!(ids,index(p))
        end
        push!(cells,Tuple(ids))
    end
    return [nodes[i][j] for i in eachindex(nodes),j in 1:3],
           [cells[i][j] for i in eachindex(cells),j in 1:4]
end

@testset "Independent optimal-rate 3D validation" begin
    X=[.125 .25 0.;1. .125 .25;.25 1.25 .125;0. .25 1.]
    T=reshape([1,2,3,4],1,4)
    vm=loaded_mesh(X,T);m=SimplexMesh(vm)
    Xvm=vm.NodeList[vm.ElementList[1,:],:]
    K,M,W,P=OF3.cecr_parts(vm,m,[-.37])
    ids=[vm.Element2Facet[1,:];vm.NumF+1]
    @testset "Independent monomial and face/cell moments" begin
        Kr,Mr,_,_=monomial_reference(Xvm)
        @test norm(Matrix(K[ids,ids])-Kr)/norm(Kr)<2e-12
        @test norm(Matrix(M[ids,ids])-Mr)/norm(Mr)<2e-12
        Y=(Xvm.-Xvm[1:1,:])./m.hK[1]
        err=independent_duality(Y)
        diagnostic["face_cell_duality_error"]=err
        @test err<3e-13
        @test minimum(eigvals(Symmetric(Matrix(M))))>0
        @test norm(K*ones(size(K,1)))<1e-12*norm(K)
        @test abs(sum(M)-m.vol[1])<2e-14
        @test nnz(W)==1 && nnz(P)==1
        @test W[vm.NumF+1,vm.NumF+1]≈-.37*m.vol[1]
        @test P[vm.NumF+1,vm.NumF+1]≈m.vol[1]
        Ka,Ma,_=VFEM.create_matrix_cecr_3d(vm,[-.37])
        ek=norm(K+W-Ka)/norm(Ka);em=norm(M-Ma)/norm(Ma)
        diagnostic["ordinary_vfem_stiffness_relative_error"]=ek
        diagnostic["ordinary_vfem_mass_relative_error"]=em
        @test ek<3e-12 && em<3e-12
    end

    @testset "Dyadic translation and physical scaling" begin
        worstK=0.;worstM=0.
        for power in (-10,-24,-40),a in ([2.,.5,-.25],[-2.,.5,-.25])
            scale=2.0^power
            vms=loaded_mesh(scale.*X.+a',T);ms=SimplexMesh(vms)
            Ks,Ms,_,_=OF3.cecr_parts(vms,ms,[0.])
            ek=norm(Ks/scale-K)/norm(K);em=norm(Ms/scale^3-M)/norm(M)
            worstK=max(worstK,ek);worstM=max(worstM,em)
            @test ek<2e-11 && em<2e-11
        end
        diagnostic["tiny_scale_stiffness_relative_error"]=worstK
        diagnostic["tiny_scale_mass_relative_error"]=worstM
    end

    @testset "Coulomb and P1 quadrature" begin
        Y=[0. 1. 0. 0.;0. 0. 1. 0.;0. 0. 0. 1.]
        mt=SimplexMesh(Y,reshape([1,2,3,4],4,1));pot=Coulomb(zeros(3,1),[1.])
        qlow=CC.QuadSet(3;n_sing=16,n_near=12,n_far=6)
        qhigh=CC.QuadSet(3;n_sing=40,n_near=20,n_far=10)
        c=OF3.averages(pot,mt;qs=qlow);ch=OF3.averages(pot,mt;qs=qhigh)
        ec=abs(c[1]/ch[1]-1);diagnostic["singular_average_refinement_relative_error"]=ec
        @test ec<1e-10
        Kp,Mp,Wp=OF3.p1_parts(pot,mt;qs=qlow)
        Kh,Mh,Wh=OF3.p1_parts(pot,mt;qs=qhigh)
        ew=norm(Wp-Wh)/norm(Wh)
        diagnostic["singular_P1_refinement_relative_error"]=ew
        @test ew<1e-9
        @test norm(Kp-Kh)<1e-14 && norm(Mp-Mh)<1e-14
        @test maximum(eigvals(Symmetric(Matrix(Wp))))<0
        Ka,Ma,Wa=CC.p1_matrices(mt;pot=pot,qs=qlow)
        @test norm(Kp-Ka)<2e-13 && norm(Mp-Ma)<2e-13 && norm(Wp-Wa)<2e-13
        for power in (-24,-40)
            scale=2.0^power;a=[2.,.5,-.25]
            tiny=SimplexMesh(scale.*Y.+a,reshape([1,2,3,4],4,1))
            pt=Coulomb(reshape(a,3,1),[1.])
            ct=OF3.averages(pt,tiny;qs=qlow)
            @test abs(ct[1]*scale/c[1]-1)<2e-13
            kt,mtiny,wt=OF3.p1_parts(pt,tiny;qs=qlow)
            @test norm(kt/scale-Kp)/norm(Kp)<2e-13
            @test norm(mtiny/scale^3-Mp)/norm(Mp)<2e-13
            @test norm(wt/scale^2-Wp)/norm(Wp)<2e-13
        end
    end

    @testset "Local disjoint support uses max, not sum" begin
        Y=[0. 1. 0. 0.;0. 0. 1. 0.;0. 0. 0. 1.]
        a=[-2. 2.;0. 0.;0. 0.];scale=2.0^-4
        XX=hcat(scale.*Y.+a[:,1],scale.*Y.+a[:,2])
        mm=SimplexMesh(XX,[1 5;2 6;3 7;4 8]);pot=Coulomb(a,[1.,2.])
        c=OF3.averages(pot,mm)
        qs=CC.QuadSet(3;n_sing=16,n_near=12,n_far=6,q_sing=2)
        p=1.5;S=sqrt(2)/sqrt(3pi)*(4/sqrt(pi))^(1/3)
        vals=[CC.lp_dev_integral(pot,mm,e,c[e],p;qs=qs) for e in 1:2]
        dh=OF3.perturbation(pot,mm,c,S,p;qs=qs,r_loc=1.)
        expected=S^2*maximum(vals)^(1/p)
        @test dh.npatch==2 && dh.off==0
        @test dh.rmax<=1.
        @test abs(dh.dh/expected-1)<1e-9
        @test dh.dh<S^2*sum(vals)^(1/p)
        diagnostic["local_patch_maxnorm_relative_error"]=abs(dh.dh/expected-1)
        outside=OF3.perturbation(pot,mm,c,S,p;qs=qs,r_loc=scale/2)
        @test !isfinite(outside.dh)
    end

    @testset "Full spectrum equals all eight parity sectors" begin
        Xc,Tc=reflected_cube();v=loaded_mesh(Xc,Tc);mc=SimplexMesh(v)
        @test CC.nelem(mc)==48
        @test interior_free_facets(mc,x->any(abs.(x).==1))==0
        cc=fill(-.3,CC.nelem(mc));Kc,Mc,Wc,Pc=OF3.cecr_parts(v,mc,cc)
        H=Kc+Wc;sigma=2.
        sectors,info=OF3.sector_parts(v,H,Mc,Pc)
        @test length(sectors)==8
        @test sum(size(s.H,1) for s in sectors)==size(H,1)
        @test all(maximum(x)<1e-12 for x in values(info.invariance_errors))
        dense=eigvals(Symmetric(Matrix(H+sigma*Pc)),Symmetric(Matrix(Mc)))
        union=sort(vcat([eigvals(Symmetric(Matrix(s.H+sigma*s.P)),Symmetric(Matrix(s.M))) for s in sectors]...))
        ee=maximum(abs.(union.-dense))/maximum(abs.(dense))
        diagnostic["all_sector_full_spectrum_relative_error"]=ee
        @test ee<2e-12
        mu,si=OF3.sector_eigs(sectors,sigma,8)
        @test norm(mu-dense[1:8],Inf)<2e-11
        @test length(si.sectors)==8 && length(mu)==8
        diagnostic["all_sector_first8_absolute_error"]=norm(mu-dense[1:8],Inf)
        @test minimum(dense)>0
    end
end

diagnostic["status"]="passed"
mkpath(joinpath(@__DIR__,"results_complete"))
open(joinpath(@__DIR__,"results_complete/validation_3d.json"),"w") do io
    JSON.print(io,diagnostic,2)
end
println(JSON.json(diagnostic))
