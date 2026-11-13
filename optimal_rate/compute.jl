# Usage: julia -t 16 --project=. optimal_rate/compute.jl CASE MESHDIR RESULT.json
# CASE is 2d_one_centre or 2d_two_centre. CECR_QUAD_CHECK=1 repeats
# averages, d_h and P1 upper bounds with higher quadrature orders.
include(joinpath(@__DIR__,"stable_fem.jl"))
using .OptimalRateFEM, CoulombCECR, VFEM, LinearAlgebra, SparseArrays, Random, JSON, Printf
const OF=OptimalRateFEM
const CC=CoulombCECR
Random.seed!(20261110)
BLAS.set_num_threads(1)

function selftest()
    using_test=nothing
    vm,_=VFEM.mesh2d_triangle_uniform(8)
    m=SimplexMesh(vm); z=zeros(CC.nelem(m))
    K,M,_,_=OF.cecr_parts(vm,m,z)
    Kr,Mr,_=VFEM.create_matrix_cecr(vm,z)
    @assert norm(K-Kr)/norm(Kr)<1e-11
    @assert norm(M-Mr)/norm(Mr)<1e-11
    @assert norm(K*ones(size(K,1)))<1e-10*norm(K)
    s=1e-7; nodes=vm.nodes.*s .+ [-2. 0.]
    vms=VFEM.Mesh2D(nodes,vm.elements,vm.edges,vm.bd_edges,vm.bd_edge_ids,
        vm.tri2edge,vm.nv,vm.nt,vm.ne,vm.nb)
    ms=SimplexMesh(vms); Ks,Ms,_,_=OF.cecr_parts(vms,ms,z)
    @assert norm(Ks-K)/norm(K)<1e-7
    @assert norm(Ms./s^2-M)/norm(M)<1e-7
    tri=SimplexMesh([0. 1. 0.;0. 0. 1.],reshape([1,2,3],3,1))
    pot=Coulomb(zeros(2,1),[1.])
    c=OF.averages(pot,tri)
    @assert isapprox(c[1],-2sqrt(2)*asinh(1);rtol=1e-12)
    q=CC.QuadSet(2;n_sing=16,n_near=10,n_far=5)
    pot2=Coulomb(reshape([.5,.5],2,1),[1.])
    Kp,Mp,Wp=OF.p1_parts(pot2,m;qs=q)
    Ka,Ma,Wa=CC.p1_matrices(m;pot=pot2,qs=q)
    @assert norm(Kp-Ka)/norm(Ka)<1e-12
    @assert norm(Mp-Ma)/norm(Ma)<1e-12
    @assert norm(Wp-Wa)/norm(Wa)<1e-11
    println("SELFTEST PASS: baseline assembly, translation/scale invariance, singular average, P1 potential")
end

function main(args)
    if length(args)==1 && args[1]=="--self-test"
        selftest(); return
    end
    length(args)==3||error("CASE MESHDIR RESULT.json required")
    case,meshdir,output=args; t0=time();ts=t0
    case in ("2d_one_centre","2d_two_centre")||error("unsupported case $case")
    two=case=="2d_two_centre"
    A=two ? [-2. 2.;0. 0.] : zeros(2,1);Z=ones(size(A,2))
    S=two ? 1.462 : 1.466;eps=.55;sigfix=12.5;kap=1-eps;k=3
    reportfile=joinpath(meshdir,"mesh_report.json")
    rep=isfile(reportfile) ? JSON.parsefile(reportfile) : Dict{String,Any}()
    h=get(rep,"h",nothing)
    R=Dict{String,Any}("case"=>case,"mesh"=>basename(normpath(meshdir)),"h"=>h,
        "mesh_report"=>rep,"eps"=>eps,"p"=>4/3,"S"=>S,"arithmetic"=>"Float64",
        "method"=>"locally normalized CECR; centre-relative Coulomb quadrature")
    timing=Dict{String,Float64}()
    function lap(key)
        timing[key]=time()-ts;ts=time();flush(stdout)
    end
    vm=CC.load_vfem_mesh(meshdir,2);m=SimplexMesh(vm);pot=Coulomb(A,Z)
    R["n_elem"]=CC.nelem(m);R["n_node"]=CC.nnode(m)
    R["h_max"]=maximum(m.hK);R["hK_min"]=minimum(m.hK)
    onb=x->abs(abs(x[1])-(two ? 7 : 5))<1e-10||abs(abs(x[2])-5)<1e-10
    R["interior_free_facets"]=interior_free_facets(m,onb)
    R["interior_free_facets"]==0||error("mesh is not conforming")
    println("[$case] h=$h ne=$(R["n_elem"]) nv=$(R["n_node"])");lap("load")
    c=OF.averages(pot,m)
    Ch=interp_constant(m);Gam=reaction_constant(m,c)
    R["C_h"]=Ch;R["Gamma_h"]=Gam;R["A_h"]=Gam/kap;lap("averages")
    dh=OF.perturbation(pot,m,c,S,4/3)
    R["d_h"]=dh.dh;R["d_h_patch"]=dh.patch;R["d_h_off"]=dh.off
    R["rho_opt"]=isfinite(dh.rho) ? dh.rho : nothing
    R["patch_rmax"]=dh.rmax;R["n_patch"]=dh.npatch
    dh.dh<kap||error("perturbation is too large")
    println(@sprintf("d_h=%.8e (patch %.8e off %.8e), Gamma=%.8e",dh.dh,dh.patch,dh.off,Gam))
    lap("d_h")
    dist=fill(5.,length(Z));rho_max=fill(two ? 2. : 5.,length(Z))
    rb=decoupled_form_bound(A,Z,eps-dh.dh,rho_max;bdry_dist=dist)
    Cv=rb.C+dh.dh;sig=Cv+kap
    Cv<=sigfix-kap||error("fixed shift does not provide the requested coercivity margin")
    R["form_bound_admissible"]=true
    R["fixed_shift_margin"]=sigfix-Cv
    R["C_cert"]=Cv;R["C_cert_how"]="radial Neumann decoupling at eps-d_h"
    R["radial_C_V"]=rb.C;R["radial_rho"]=rb.rho;R["radial_nu_err"]=rb.nu_err
    R["sigma_opt"]=sig;R["sigma"]=sigfix;R["eps_h"]=dh.dh/kap;R["kappa"]=kap
    K,M,W,P=OF.cecr_parts(vm,m,c);R["n_dof_cecr"]=size(K,1)
    lap("cecr_assembly")
    mu,info=OF.eigs(K+W+sig*P,M,k)
    R["mu_minus_sigma_opt"]=mu.-sig
    R["L_opt"]=corrected_lower(cecr_lower(mu,sig,Ch,Gam,eps),sig,dh.dh/kap)
    R["cecr_resid_opt"]=info.residuals;R["cecr_backward_resid_opt"]=info.backward_residuals
    R["cecr_mvproducts_opt"]=info.mvproducts
    muf,infof=OF.eigs(K+W+sigfix*P,M,k)
    R["mu_minus_sigma"]=muf.-sigfix
    R["L"]=corrected_lower(cecr_lower(muf,sigfix,Ch,Gam,eps),sigfix,dh.dh/kap)
    R["cecr_resid"]=infof.residuals;R["cecr_backward_resid"]=infof.backward_residuals
    println("L_opt=$(R["L_opt"])");lap("cecr_solves")
    # Release the largest pencil before assembling P1 forms.
    K=nothing;M=nothing;W=nothing;P=nothing;GC.gc()
    Kp,Mp,Wp=OF.p1_parts(pot,m);Ap=Kp+Wp
    rb1=decoupled_form_bound(A,Z,1.,rho_max;bdry_dist=dist)
    shiftP=-rb1.C-.5
    un,infon=OF.eigs(Ap,Mp,k;shift=shiftP)
    ids=findall(.!m.bnodes)
    ud,infod=OF.eigs(Ap[ids,ids],Mp[ids,ids],k;shift=shiftP)
    R["U_neumann"]=un;R["U_dirichlet"]=ud
    R["p1_resid_neumann"]=infon.residuals;R["p1_resid_dirichlet"]=infod.residuals
    R["p1_backward_resid_neumann"]=infon.backward_residuals
    R["p1_backward_resid_dirichlet"]=infod.backward_residuals
    R["neumann_width"]=un.-R["L_opt"]
    R["dirichlet_width"]=ud.-R["L_opt"]
    all(R["neumann_width"].>0)||error("negative enclosure width")
    println("U_N=$un U_D=$ud");lap("p1_upper")
    if get(ENV,"CECR_QUAD_CHECK","0")=="1"
        qhi=CC.QuadSet(2;n_sing=24,n_near=16,n_far=8)
        chi=OF.averages(pot,m;qs=qhi)
        dhi=OF.perturbation(pot,m,chi,S,4/3;
            qs=CC.QuadSet(2;n_sing=32,n_near=20,n_far=10,q_sing=3))
        Kh,Mh,Wh=OF.p1_parts(pot,m;qs=qhi)
        unh,_=OF.eigs(Kh+Wh,Mh,k;shift=shiftP)
        udh,_=OF.eigs((Kh+Wh)[ids,ids],Mh[ids,ids],k;shift=shiftP)
        R["quadrature_check"]=Dict(
            "c_max_relative_change"=>maximum(abs.(chi.-c)./max.(abs.(chi),1e-300)),
            "d_h_high"=>dhi.dh,"d_h_relative_change"=>(dhi.dh-dh.dh)/dhi.dh,
            "U_neumann_high"=>unh,"U_neumann_change"=>unh.-un,
            "U_dirichlet_high"=>udh,"U_dirichlet_change"=>udh.-ud)
        println("quadrature_check=$(R["quadrature_check"])");lap("quadrature_check")
    end
    R["timing"]=timing;R["wall_s"]=time()-t0
    mkpath(dirname(abspath(output)))
    open(output,"w") do io
        JSON.print(io,R,2)
    end
    println("WROTE $output");println("DONE_MARKER")
end
main(ARGS)
