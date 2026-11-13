# Complete optimal-rate computation, including actual per-k shifted solves.
# julia -t 16 --project=. optimal_rate/complete_case.jl CASE MESHDIR RESULT.json
# The original release and the earlier convergence-run data are not modified.
include(joinpath(@__DIR__, "stable_fem.jl"))
using .OptimalRateFEM, CoulombCECR, VFEM, LinearAlgebra, SparseArrays, Random, JSON, Printf, SHA
const OF = OptimalRateFEM
const CC = CoulombCECR
BLAS.set_num_threads(1)
Random.seed!(20261110)

function best_actual(base, candidates, k)
    choices = [base; candidates]
    j = argmax([x.L[k] for x in choices])
    return choices[j]
end

function selftest()
    using_test = nothing
    b = (eps=.55, C=2., sigma=2.45, mu_minus_sigma=[-1.,-.2,-.1],
         L=[-1.1,-.3,-.2], residuals=zeros(3), relative_residuals=zeros(3))
    a = merge(b, (eps=.9, L=[-1.2,-.25,-.18]))
    @assert best_actual(b,[a],1).eps == .55
    @assert best_actual(b,[a],2).eps == .9
    @assert best_actual(b,typeof(b)[],3).eps == .55
    mu = [1.5,2.,2.5]; sig=2.5; ch=.1; gam=.01; eps=.55; dh=.001
    @assert CC.lower_bound_at(mu.-sig,sig,ch,gam,eps,dh) ≈
        corrected_lower(cecr_lower(mu,sig,ch,gam,eps),sig,dh/(1-eps))
    println("SELFTEST PASS: actual-candidate fallback, metadata selection, bound formula")
end

function write_result(path, R)
    mkpath(dirname(abspath(path)))
    tmp = path * ".tmp"
    open(tmp,"w") do io
        JSON.print(io,R,2)
    end
    mv(tmp,path;force=true)
end

function main(args)
    if length(args)==1 && args[1]=="--self-test"
        selftest(); return
    end
    length(args)==3 || error("CASE MESHDIR RESULT.json required")
    case, meshdir, output = args
    case in ("2d_one_centre","2d_two_centre") || error("unsupported case $case")
    t0=time(); ts=t0; timing=Dict{String,Float64}()
    function lap(key)
        timing[key]=time()-ts; ts=time(); flush(stdout)
    end
    two=case=="2d_two_centre"
    A=two ? [-2. 2.;0. 0.] : zeros(2,1); Z=ones(size(A,2)); k=3
    S=two ? 1.462 : 1.466; p=4/3; eps=.55; kap=1-eps
    rep=JSON.parsefile(joinpath(meshdir,"mesh_report.json")); h=rep["h"]
    R=Dict{String,Any}("case"=>case,"mesh"=>basename(normpath(meshdir)),"h"=>h,
        "mesh_report"=>rep,"eps"=>eps,"p"=>p,"S"=>S,"arithmetic"=>"Float64",
        "method"=>"locally normalized CECR; centre-relative Coulomb quadrature",
        "complete_case"=>true,"fresh_physics_and_P1"=>true,
        "per_k_selection"=>"Interpolation selects grid candidates; actual shifted eigensolves provide every reported bound; baseline is retained if stronger.")
    R["source_sha256"]=Dict(
        "nodes.dat"=>bytes2hex(sha256(read(joinpath(meshdir,"nodes.dat")))),
        "elements.dat"=>bytes2hex(sha256(read(joinpath(meshdir,"elements.dat")))),
        "stable_fem.jl"=>bytes2hex(sha256(read(joinpath(@__DIR__,"stable_fem.jl")))),
        "complete_case.jl"=>bytes2hex(sha256(read(@__FILE__))))
    R["quadrature_settings"]=Dict("averages_P1"=>[16,10,5],
        "Lp_patch"=>[20,12,6],"singular_Lp_collapse_exponent"=>3)
    vm=CC.load_vfem_mesh(meshdir,2); m=SimplexMesh(vm); pot=Coulomb(A,Z)
    R["n_elem"]=CC.nelem(m); R["n_node"]=CC.nnode(m)
    R["h_max"]=maximum(m.hK); R["hK_min"]=minimum(m.hK)
    onb=x->abs(abs(x[1])-(two ? 7 : 5))<1e-10 || abs(abs(x[2])-5)<1e-10
    R["interior_free_facets"]=interior_free_facets(m,onb)
    R["interior_free_facets"]==0 || error("mesh is not conforming")
    println("[$case] complete h=$h ne=$(R["n_elem"]) nv=$(R["n_node"])"); lap("load")
    c=OF.averages(pot,m); Ch=interp_constant(m); Gam=reaction_constant(m,c)
    R["C_h"]=Ch; R["Gamma_h"]=Gam; R["A_h"]=Gam/kap; lap("averages")
    dh=OF.perturbation(pot,m,c,S,p)
    R["d_h"]=dh.dh; R["d_h_patch"]=dh.patch; R["d_h_off"]=dh.off
    R["rho_opt"]=isfinite(dh.rho) ? dh.rho : nothing
    R["patch_rmax"]=dh.rmax; R["n_patch"]=dh.npatch
    0<dh.dh<kap || error("inadmissible perturbation")
    R["kappa"]=kap; R["eps_h"]=dh.dh/kap
    println(@sprintf("d_h=%.10e (patch %.3e off %.3e), Gamma_h=%.10e",dh.dh,dh.patch,dh.off,Gam)); lap("d_h")

    bd=fill(5.,length(Z)); maxrho=fill(two ? 2. : 5.,length(Z))
    function form_at(e)
        rb=decoupled_form_bound(A,Z,e-dh.dh,maxrho;bdry_dist=bd)
        C=rb.C+dh.dh
        return (eps=e,C=C,sigma=C+1-e,C_V=rb.C,rho=rb.rho,nu_err=rb.nu_err)
    end
    baseform=form_at(eps)
    R["C_cert"]=baseform.C; R["C_cert_how"]="radial Neumann decoupling at eps-d_h"
    R["radial_C_V"]=baseform.C_V; R["radial_rho"]=baseform.rho
    R["radial_nu_err"]=baseform.nu_err; R["sigma_opt"]=baseform.sigma
    R["form_bound_admissible"]=true
    forms=[form_at(e) for e in sort(unique([collect(.30:.025:.95);eps]))
           if e-dh.dh>.05 && dh.dh/(1-e)<.5]
    any(g->abs(g.eps-eps)<1e-12,forms) || error("baseline missing from epsilon grid")
    lap("radial_grid")

    K,M,W,P=OF.cecr_parts(vm,m,c); H=K+W
    K=nothing; W=nothing; GC.gc()
    R["n_dof_cecr"]=size(H,1); lap("cecr_assembly")
    function solve_at(g)
        mu,info=OF.eigs(H+g.sigma*P,M,k)
        L=corrected_lower(cecr_lower(mu,g.sigma,Ch,Gam,g.eps),g.sigma,dh.dh/(1-g.eps))
        println(@sprintf("eps=%.3f sigma=%.8f: mu-sigma=%s L=%s",g.eps,g.sigma,string(mu.-g.sigma),string(L)));flush(stdout)
        return merge(g,(mu_minus_sigma=mu.-g.sigma,L=L,residuals=info.residuals,
                         relative_residuals=info.backward_residuals,mvproducts=info.mvproducts))
    end
    baseline=solve_at(baseform)
    R["mu_minus_sigma_opt"]=baseline.mu_minus_sigma; R["L_opt"]=baseline.L
    R["cecr_resid_opt"]=baseline.residuals
    R["cecr_backward_resid_opt"]=baseline.relative_residuals
    # Same candidate-selection principle as release src/pipeline.jl. The
    # baseline eigenvalues initialize the predicted shift dependence. An
    # additional actual anchor at the highest admissible epsilon improves
    # interpolation without an expensive eigensolve at every grid point.
    exact=[baseline]
    anchor=forms[end]
    if abs(anchor.eps-eps)>1e-12
        push!(exact,solve_at(anchor))
    end
    preds=[CC.lower_bound_at(CC.interp_shifted([(s.sigma,s.mu_minus_sigma) for s in exact],g.sigma),
                            g.sigma,Ch,Gam,g.eps,dh.dh) for g in forms]
    selected=[argmax([x[ik] for x in preds]) for ik in 1:k]
    for gi in unique(selected)
        g=forms[gi]
        any(s->abs(s.eps-g.eps)<1e-12,exact) || push!(exact,solve_at(g))
    end
    # All actual candidate solves are valid for every eigenvalue index.
    # Keeping the baseline explicitly prevents interpolation from degrading
    # the final bound or producing inconsistent epsilon/shift metadata.
    best=[best_actual(baseline,exact,ik) for ik in 1:k]
    R["eps_best"]=[s.eps for s in best]
    R["C_best"]=[s.C for s in best]; R["sigma_best"]=[s.sigma for s in best]
    R["eps_h_best"]=[dh.dh/(1-s.eps) for s in best]
    R["mu_minus_sigma_best"]=[best[ik].mu_minus_sigma[ik] for ik in 1:k]
    R["L_best"]=[best[ik].L[ik] for ik in 1:k]
    R["cecr_resid_best"]=[best[ik].residuals[ik] for ik in 1:k]
    R["cecr_relative_resid_best"]=[best[ik].relative_residuals[ik] for ik in 1:k]
    R["perk_solves"]=[Dict("eps"=>s.eps,"C"=>s.C,"sigma"=>s.sigma,
        "mu_minus_sigma"=>s.mu_minus_sigma,"L"=>s.L,"residuals"=>s.residuals,
        "relative_residuals"=>s.relative_residuals,"mvproducts"=>s.mvproducts,
        "radial_rho"=>s.rho,"radial_nu_err"=>s.nu_err) for s in exact]
    R["perk_grid"]=[Dict("eps"=>g.eps,"C"=>g.C,"sigma"=>g.sigma,
        "radial_rho"=>g.rho,"L_pred"=>preds[j]) for (j,g) in enumerate(forms)]
    R["n_actual_cecr_solves"]=length(exact)
    @assert all(R["L_best"].>=R["L_opt"].-1e-12)
    println("per-k eps=$(R["eps_best"]) L_best=$(R["L_best"])"); lap("cecr_solves")

    H=nothing; M=nothing; P=nothing; GC.gc()
    Kp,Mp,Wp=OF.p1_parts(pot,m); Ap=Kp+Wp
    rb1=decoupled_form_bound(A,Z,1.,maxrho;bdry_dist=bd)
    R["L1_trivial_radial"]=-rb1.C
    shiftP=-rb1.C-.5
    un,infon=OF.eigs(Ap,Mp,k;shift=shiftP)
    ids=findall(.!m.bnodes)
    ud,infod=OF.eigs(Ap[ids,ids],Mp[ids,ids],k;shift=shiftP)
    R["U_neumann"]=un; R["U_dirichlet"]=ud
    R["p1_resid_neumann"]=infon.residuals; R["p1_resid_dirichlet"]=infod.residuals
    R["p1_backward_resid_neumann"]=infon.backward_residuals
    R["p1_backward_resid_dirichlet"]=infod.backward_residuals
    R["neumann_width"]=un.-baseline.L; R["dirichlet_width"]=ud.-baseline.L
    R["neumann_width_best"]=un.-R["L_best"]
    R["dirichlet_width_best"]=ud.-R["L_best"]
    @assert all(R["L_best"].<=un) && all(un.<=ud)
    sext=two ? -2/sqrt(29.) : -.2
    R["exterior_threshold"]=sext
    R["whole_space_confinement"]=[u<sext for u in ud]
    R["finite_neumann_intervals"]=[[R["L_best"][ik],un[ik]] for ik in 1:k]
    R["whole_space_intervals"]=Dict(string(ik)=>[R["L_best"][ik],ud[ik]] for ik in 1:k if ud[ik]<sext)
    R["isolation_by_next_lower"]=[un[ik]<R["L_best"][ik+1] for ik in 1:k-1]
    println("U_N=$un U_D=$ud");lap("p1_upper")
    R["timing"]=timing;R["wall_s"]=time()-t0
    write_result(output,R)
    println("WROTE $output");println("DONE_MARKER")
end
main(ARGS)
