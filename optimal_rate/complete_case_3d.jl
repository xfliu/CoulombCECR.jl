# Complete optimal-rate H2+ case on the symmetric box mesh.
# julia -t 16 --project=. optimal_rate/complete_case_3d.jl MESHDIR RESULT.json
include(joinpath(@__DIR__,"stable_fem_3d.jl"))
using .OptimalRateFEM3D, .OptimalRateFEM, CoulombCECR, VFEM, LinearAlgebra, SparseArrays, Random, JSON, SHA, DelimitedFiles
const OF3=OptimalRateFEM3D
const CC=CoulombCECR
BLAS.set_num_threads(parse(Int,get(ENV,"CECR_BLAS_THREADS","4")))
Random.seed!(20261110)

function best_actual3d(base,candidates,k)
    choices=[base;candidates]
    return choices[argmax([s.L[k] for s in choices])]
end

function write_result3d(path,R)
    mkpath(dirname(abspath(path)))
    open(path*".tmp","w") do io
        JSON.print(io,R,2)
    end
    mv(path*".tmp",path;force=true)
end

function reflected_test_mesh()
    # Kuhn triangulation of the positive unit cube, reflected exactly.
    # Explicit rows avoid array-comprehension layout ambiguity.
    X=zeros(8,3)
    id(i,j,k)=1+i+2j+4k
    for i in 0:1,j in 0:1,k in 0:1
        X[id(i,j,k),:]=[i,j,k]
    end
    perms=((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1))
    E=([1,0,0],[0,1,0],[0,0,1]);T=zeros(Int,6,4)
    for (it,p) in enumerate(perms)
        a=[0,0,0];b=a+E[p[1]];c=b+E[p[2]];d=[1,1,1]
        T[it,:]=[id(a...),id(b...),id(c...),id(d...)]
    end
    for ax in 1:3
        on=X[:,ax].==0;nv=size(X,1);pn=zeros(Int,nv);n=nv
        for v in 1:nv
            pn[v]=on[v] ? v : (n+=1)
        end
        Y=copy(X[.!on,:]);Y[:,ax].*=-1
        X=vcat(X,Y);T=vcat(T,pn[T])
    end
    return X,T
end

function selftest3d()
    X,T=reflected_test_mesh();dir=mktempdir()
    writedlm(joinpath(dir,"nodes.dat"),X);writedlm(joinpath(dir,"elements.dat"),T)
    vm=CC.load_vfem_mesh(dir,3);m=SimplexMesh(vm);c=fill(-.3,CC.nelem(m))
    K,M,W,P=OF3.cecr_parts(vm,m,c)
    Kr,Mr,_=VFEM.create_matrix_ecr_3d(vm)
    @assert norm(K-Kr)/norm(Kr)<1e-11
    @assert norm(M-Mr)/norm(Mr)<1e-11
    Ar,_,_=VFEM.create_matrix_cecr_3d(vm,c)
    @assert norm(K+W-Ar)/norm(Ar)<1e-11
    @assert norm(K*ones(size(K,1)))<1e-11*norm(K)
    @assert minimum(eigvals(Symmetric(Matrix(M))))>0
    Y=[0. 0. 0.;1. 0. 0.;0. 1. 0.;0. 0. 1.]
    C=OF3.local_coefficients(Y,VFEM._ecr3d_cr_coeffs(Float64))
    for j in 1:5
        @assert norm(VFEM._ecr3d_q_face_avgs(C[:,j])-[Float64(i==j) for i in 1:4])<1e-12
        @assert abs(VFEM._ecr3d_q_cell_avg(C[:,j])-Float64(j==5))<1e-12
    end
    sectors,si=OF3.sector_parts(vm,K+W,M,P)
    mu,info=OF3.sector_eigs(sectors,2.,3)
    ref=eigvals(Symmetric(Matrix(K+W+2P)),Symmetric(Matrix(M)))[1:3]
    @assert maximum(abs.(mu.-ref))<1e-10
    @assert length(info.sectors)==8
    @assert sum(size(s.H,1) for s in sectors)==size(K,1)
    tri=SimplexMesh(Matrix(Y'),reshape([1,2,3,4],4,1));pot=Coulomb(zeros(3,1),[1.])
    qs=CC.QuadSet(3;n_sing=12,n_near=8,n_far=4)
    cb=OF3.averages(pot,tri;qs=qs)
    cr=CC.cell_averages(pot,tri;qs=CC.QuadSet(3;n_sing=28,n_near=12,n_far=6))
    @assert isapprox(cb[1],cr[1];rtol=1e-8)
    s=2.0^-24;a=[2.,0.,0.]
    tiny=SimplexMesh(s.*Matrix(Y').+a,reshape([1,2,3,4],4,1))
    pct=Coulomb(reshape(a,3,1),[1.]);ct=OF3.averages(pct,tiny;qs=qs)
    @assert isapprox(ct[1]*s,cb[1];rtol=1e-13)
    Kp,Mp,Wp=OF3.p1_parts(pot,tri;qs=qs)
    Ka,Ma,Wa=CC.p1_matrices(tri;pot=pot,qs=qs)
    @assert norm(Kp-Ka)<1e-13 && norm(Mp-Ma)<1e-13 && norm(Wp-Wa)<1e-12
    @assert abs(sum(Mp)-1/6)<1e-14
    println("SELFTEST 3D PASS: normalized/reference CECR, duality, mass positivity, all-eight-sector spectrum, relative singular quadrature, P1 reference")
end

function main3d(args)
    if length(args)==1 && args[1]=="--self-test"
        selftest3d();return
    end
    length(args)==2 || error("MESHDIR RESULT.json required")
    meshdir,output=args;t0=time();ts=t0;timing=Dict{String,Float64}()
    function lap(key)
        timing[key]=time()-ts;ts=time();flush(stdout)
    end
    rep=JSON.parsefile(joinpath(meshdir,"mesh_report.json"));h=rep["h"]
    A=[-2. 2.;0. 0.;0. 0.];Z=[1.,1.];pot=Coulomb(A,Z);k=3
    K3=1/sqrt(3pi)*(4/sqrt(pi))^(1/3);S=sqrt(2)*K3;p=1.5;eps=.6;kap=.4
    R=Dict{String,Any}("case"=>"3d_h2plus","mesh"=>basename(normpath(meshdir)),"h"=>h,
        "mesh_report"=>rep,"complete_case"=>true,"fresh_physics_and_P1"=>true,
        "eps"=>eps,"p"=>p,"S"=>S,"per_centre"=>true,"r_loc"=>1.,"arithmetic"=>"Float64",
        "method"=>"locally normalized 3D CECR; all-eight-parity-sector union; centre-relative Coulomb quadrature",
        "per_k_selection"=>"Interpolation selects grid candidates; every reported bound is recomputed by an actual all-sector eigensolve; baseline retained if stronger.",
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
        "sector_parallel"=>parse(Int,get(ENV,"CECR_SECTOR_PARALLEL","4")))
    R["source_sha256"]=Dict(name=>bytes2hex(sha256(read(path))) for (name,path) in [
        "nodes.dat"=>joinpath(meshdir,"nodes.dat"),"elements.dat"=>joinpath(meshdir,"elements.dat"),
        "stable_fem.jl"=>joinpath(@__DIR__,"stable_fem.jl"),
        "stable_fem_3d.jl"=>joinpath(@__DIR__,"stable_fem_3d.jl"),
        "complete_case_3d.jl"=>@__FILE__])
    R["quadrature_settings"]=Dict("averages_P1"=>[12,8,4],"Lp_patch"=>[14,10,5],"singular_Lp_collapse_exponent"=>2)
    vm=CC.load_vfem_mesh(meshdir,3);m=SimplexMesh(vm)
    onb=x->abs(abs(x[1])-8)<1e-10 || abs(abs(x[2])-6)<1e-10 || abs(abs(x[3])-6)<1e-10
    R["interior_free_facets"]=interior_free_facets(m,onb)
    R["interior_free_facets"]==0 || error("nonconforming mesh")
    R["n_elem"]=CC.nelem(m);R["n_node"]=CC.nnode(m)
    R["h_max"]=maximum(m.hK);R["hK_min"]=minimum(m.hK)
    println("[3d_h2plus] complete h=$h ne=$(R["n_elem"]) nv=$(R["n_node"])");lap("load")
    sym=CC.symmetry_data(vm,[1,2,3])
    craw=OF3.averages(pot,m);c=CC.orbit_average(sym,craw)
    R["c_orbit_maxchange"]=maximum(abs.(c.-craw))
    R["c_orbit_maxchange_rel"]=R["c_orbit_maxchange"]/maximum(abs.(craw))
    Ch=interp_constant(m);Gam=reaction_constant(m,c)
    R["C_h"]=Ch;R["Gamma_h"]=Gam;R["A_h"]=Gam/kap;lap("averages_symmetry")
    dh=OF3.perturbation(pot,m,c,S,p;r_loc=1.)
    R["d_h"]=dh.dh;R["d_h_patch"]=dh.patch;R["d_h_off"]=dh.off
    R["rho_opt"]=isfinite(dh.rho) ? dh.rho : nothing
    R["patch_rmax"]=dh.rmax;R["n_patch"]=dh.npatch
    0<dh.dh<kap || error("baseline perturbation inadmissible: $(dh.dh)")
    R["eps_h"]=dh.dh/kap;R["kappa"]=kap
    println("d_h=$(dh.dh) (patch=$(dh.patch) off=$(dh.off)), Gamma_h=$Gam");lap("d_h")
    bd=[6.,6.];maxrho=[2.,2.]
    function form_at(e)
        rb=decoupled_form_bound(A,Z,e-dh.dh,maxrho;bdry_dist=bd)
        Cv=rb.C+dh.dh
        return (eps=e,C=Cv,sigma=Cv+1-e,C_V=rb.C,rho=rb.rho,nu_err=rb.nu_err)
    end
    bf=form_at(eps)
    forms=[form_at(e) for e in sort(unique([collect(.3:.025:.95);eps])) if e-dh.dh>.05 && dh.dh/(1-e)<.95]
    any(g->abs(g.eps-eps)<1e-12,forms) || error("baseline missing from admissible grid")
    R["C_cert"]=bf.C;R["C_cert_how"]="radial Neumann decoupling at eps-d_h"
    R["radial_C_V"]=bf.C_V;R["radial_rho"]=bf.rho;R["radial_nu_err"]=bf.nu_err
    R["sigma_opt"]=bf.sigma;R["form_bound_admissible"]=true;lap("radial_grid")
    K,M,W,P=OF3.cecr_parts(vm,m,c);H=K+W;K=nothing;W=nothing;GC.gc()
    R["n_dof_cecr"]=size(H,1)
    sectors,si=OF3.sector_parts(vm,H,M,P)
    R["symmetry_invariance_errors"]=si.invariance_errors;R["symmetry_node_tol"]=si.tol
    R["sector_dimensions"]=[size(s.H,1) for s in sectors]
    H=nothing;M=nothing;P=nothing;GC.gc();lap("cecr_assembly_projection")
    function solve_at(g)
        mu,info=OF3.sector_eigs(sectors,g.sigma,k)
        L=corrected_lower(cecr_lower(mu,g.sigma,Ch,Gam,g.eps),g.sigma,dh.dh/(1-g.eps))
        println("eps=$(g.eps) sigma=$(g.sigma): mu-sigma=$(mu.-g.sigma) L=$L");flush(stdout)
        return merge(g,(mu_minus_sigma=mu.-g.sigma,L=L,residuals=info.residuals,
            relative_residuals=info.relative_residuals,labels=info.labels,sectors=info.sectors))
    end
    baseline=solve_at(bf);exact=[baseline]
    R["L_opt"]=baseline.L;R["mu_minus_sigma_opt"]=baseline.mu_minus_sigma
    R["cecr_resid_opt"]=baseline.residuals;R["cecr_backward_resid_opt"]=baseline.relative_residuals
    R["cecr_sector_opt"]=baseline.labels
    anchor=forms[end]
    abs(anchor.eps-eps)>1e-12 && push!(exact,solve_at(anchor))
    pred=[CC.lower_bound_at(CC.interp_shifted([(s.sigma,s.mu_minus_sigma) for s in exact],g.sigma),g.sigma,Ch,Gam,g.eps,dh.dh) for g in forms]
    selected=[argmax([v[ik] for v in pred]) for ik in 1:k]
    for gi in unique(selected)
        any(s->abs(s.eps-forms[gi].eps)<1e-12,exact) || push!(exact,solve_at(forms[gi]))
    end
    best=[best_actual3d(baseline,exact,ik) for ik in 1:k]
    R["eps_best"]=[s.eps for s in best];R["C_best"]=[s.C for s in best]
    R["sigma_best"]=[s.sigma for s in best];R["eps_h_best"]=[dh.dh/(1-s.eps) for s in best]
    R["mu_minus_sigma_best"]=[best[ik].mu_minus_sigma[ik] for ik in 1:k]
    R["L_best"]=[best[ik].L[ik] for ik in 1:k]
    R["cecr_resid_best"]=[best[ik].residuals[ik] for ik in 1:k]
    R["cecr_relative_resid_best"]=[best[ik].relative_residuals[ik] for ik in 1:k]
    R["cecr_sector_best"]=[best[ik].labels[ik] for ik in 1:k]
    R["perk_solves"]=[Dict("eps"=>s.eps,"C"=>s.C,"sigma"=>s.sigma,"mu_minus_sigma"=>s.mu_minus_sigma,
        "L"=>s.L,"residuals"=>s.residuals,"relative_residuals"=>s.relative_residuals,"sector"=>s.labels,
        "sectors"=>[Dict(string(f)=>getfield(t,f) for f in fieldnames(typeof(t))) for t in s.sectors],
        "radial_rho"=>s.rho,"radial_nu_err"=>s.nu_err) for s in exact]
    R["perk_grid"]=[Dict("eps"=>g.eps,"C"=>g.C,"sigma"=>g.sigma,"radial_rho"=>g.rho,"L_pred"=>pred[j]) for (j,g) in enumerate(forms)]
    R["n_actual_cecr_solves"]=length(exact)
    @assert all(R["L_best"].>=R["L_opt"].-1e-12)
    println("per-k eps=$(R["eps_best"]) L_best=$(R["L_best"])");lap("cecr_solves")
    sectors=nothing;sym=nothing;GC.gc()
    Kp,Mp,Wp=OF3.p1_parts(pot,m);Ap=Kp+Wp
    rb1=decoupled_form_bound(A,Z,1.,maxrho;bdry_dist=bd);shiftP=-rb1.C-.5
    R["L1_trivial_radial"]=-rb1.C
    un,infon=OptimalRateFEM.eigs(Ap,Mp,k;shift=shiftP)
    ids=findall(.!m.bnodes);ud,infod=OptimalRateFEM.eigs(Ap[ids,ids],Mp[ids,ids],k;shift=shiftP)
    R["U_neumann"]=un;R["U_dirichlet"]=ud
    R["p1_resid_neumann"]=infon.residuals;R["p1_resid_dirichlet"]=infod.residuals
    R["p1_backward_resid_neumann"]=infon.backward_residuals;R["p1_backward_resid_dirichlet"]=infod.backward_residuals
    R["neumann_width"]=un.-baseline.L;R["dirichlet_width"]=ud.-baseline.L
    R["neumann_width_best"]=un.-R["L_best"];R["dirichlet_width_best"]=ud.-R["L_best"]
    @assert all(R["L_best"].<=un) && all(un.<=ud)
    sext=-2/sqrt(40.)
    R["exterior_threshold"]=sext;R["whole_space_confinement"]=[u<sext for u in ud]
    R["finite_neumann_intervals"]=[[R["L_best"][ik],un[ik]] for ik in 1:k]
    R["whole_space_intervals"]=Dict(string(ik)=>[R["L_best"][ik],ud[ik]] for ik in 1:k if ud[ik]<sext)
    R["isolation_by_next_lower"]=[un[ik]<R["L_best"][ik+1] for ik in 1:k-1]
    println("U_N=$un U_D=$ud");lap("p1_upper")
    R["timing"]=timing;R["wall_s"]=time()-t0
    write_result3d(output,R);println("WROTE $output");println("DONE_MARKER")
end
main3d(ARGS)
