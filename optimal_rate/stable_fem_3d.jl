include(joinpath(@__DIR__,"stable_fem.jl"))
module OptimalRateFEM3D
using CoulombCECR, VFEM, LinearAlgebra, SparseArrays
using ..OptimalRateFEM
const CC=CoulombCECR

"Bernstein ECR coefficients in translated, isotropically scaled coordinates."
function local_coefficients(Y,Ccr)
    cq=VFEM._ecr3d_q_coeffs(Y)
    mf=VFEM._ecr3d_q_face_avgs(cq)
    mc=VFEM._ecr3d_q_cell_avg(cq)
    beta=mc-sum(mf)/4
    abs(beta)>1e-14 || error("degenerate normalized ECR bubble")
    psi=(cq-Ccr*mf)/beta
    return hcat(Ccr .- psi./4,psi)
end

function cecr_parts(vm,m,c)
    nt=vm.NumElt;nf=vm.NumF;nd=nf+nt
    I=zeros(Int,25nt);J=similar(I);kv=zeros(25nt);mv=similar(kv)
    G2=VFEM._bernstein_gram3d_ref(2,Float64)
    G1=VFEM._bernstein_gram3d_ref(1,Float64)
    D=VFEM._bernstein_deriv_matrices3d(2,Float64)
    Ccr=VFEM._ecr3d_cr_coeffs(Float64)
    Threads.@threads for e in 1:nt
        X=vm.NodeList[vm.ElementList[e,:],:]
        Y=(X.-X[1:1,:])./m.hK[e]
        C=local_coefficients(Y,Ccr)
        B=hcat(Y[2,:]-Y[1,:],Y[3,:]-Y[1,:],Y[4,:]-Y[1,:])
        # Columns of g are physical gradients of the four barycentric coordinates.
        g=(B'\[-1. 1. 0. 0.;-1. 0. 1. 0.;-1. 0. 0. 1.])./m.hK[e]
        gx=zeros(4,5);gy=zeros(4,5);gz=zeros(4,5)
        for i in 1:4
            DiC=D[i]*C
            gx .+= g[1,i].*DiC;gy .+=g[2,i].*DiC;gz .+=g[3,i].*DiC
        end
        Ke=m.vol[e].*(gx'*G1*gx+gy'*G1*gy+gz'*G1*gz)
        Me=m.vol[e].*(C'*G2*C)
        ids=(vm.Element2Facet[e,1],vm.Element2Facet[e,2],vm.Element2Facet[e,3],vm.Element2Facet[e,4],nf+e)
        pos=25(e-1)
        for j in 1:5,i in 1:5
            pos+=1;I[pos]=ids[i];J[pos]=ids[j];kv[pos]=Ke[i,j];mv[pos]=Me[i,j]
        end
    end
    K=sparse(I,J,kv,nd,nd);M=sparse(I,J,mv,nd,nd)
    cell=collect(nf+1:nf+nt)
    W=sparse(cell,cell,c.*m.vol,nd,nd)
    P=sparse(cell,cell,m.vol,nd,nd)
    return K,M,W,P
end

function relative_values(pot,m,e,rule,iv)
    ord=iv==0 ? collect(1:4) : [iv;[j for j in 1:4 if j!=iv]]
    v=zeros(length(rule.w))
    for ic in eachindex(pot.Z)
        dx=m.X[:,m.T[ord,e]].-pot.A[:,ic]
        for q in eachindex(v)
            r1=0.;r2=0.;r3=0.
            for j in 1:4
                b=rule.bary[q,j]
                r1+=b*dx[1,j];r2+=b*dx[2,j];r3+=b*dx[3,j]
            end
            v[q]-=pot.Z[ic]/sqrt(r1*r1+r2*r2+r3*r3)
        end
    end
    return v,ord
end

function averages(pot,m;qs=CC.QuadSet(3;n_sing=12,n_near=8,n_far=4))
    c=zeros(CC.nelem(m))
    Threads.@threads for e in eachindex(c)
        r,iv=CC._pick_rule(qs,pot,m,e)
        v,_=relative_values(pot,m,e,r,iv)
        c[e]=dot(r.w,v)
    end
    return c
end

function perturbation(pot,m,c,S,p; r_loc=1.,
        qs=CC.QuadSet(3;n_sing=14,n_near=10,n_far=5,q_sing=2))
    ne=CC.nelem(m);nc=length(pot.Z)
    d=zeros(ne);near=zeros(Int,ne);rm=zeros(ne);off=zeros(ne)
    Threads.@threads for e in 1:ne
        ds=[CC.dist_point_simplex(m,e,pot.A[:,i]) for i in 1:nc]
        ic=argmin(ds);d[e]=ds[ic];near[e]=ic
        rm[e]=CC.maxdist_point_simplex(m,e,pot.A[:,ic])
        off[e]=m.hK[e]/2*CC.sup_dev(pot,m,e,c[e])
    end
    perm=sortperm(d);suffix=zeros(ne+1)
    for j in ne:-1:1
        suffix[j]=max(suffix[j+1],off[perm[j]])
    end
    mand=count(==(0.),d);acc=zeros(nc);rmax=0.
    best=(dh=Inf,patch=NaN,off=NaN,rho=NaN,npatch=0,rmax=NaN)
    for j in 1:ne
        e=perm[j]
        rm[e]>r_loc && break
        r,iv=CC._pick_rule(qs,pot,m,e)
        v,_=relative_values(pot,m,e,r,iv)
        acc[near[e]]+=m.vol[e]*dot(r.w,abs.(v.-c[e]).^p)
        rmax=max(rmax,rm[e]);j<mand && continue
        pt=S^2*maximum(acc)^(1/p)
        pt>best.dh && break
        if pt+suffix[j+1]<best.dh
            best=(dh=pt+suffix[j+1],patch=pt,off=suffix[j+1],
                  rho=j<ne ? d[perm[j+1]] : Inf,npatch=j,rmax=rmax)
        end
    end
    return best
end

function p1_parts(pot,m;qs=CC.QuadSet(3;n_sing=12,n_near=8,n_far=4))
    nt=CC.nelem(m);nd=CC.nnode(m)
    I=zeros(Int,16nt);J=similar(I);kv=zeros(16nt);mv=similar(kv);wv=similar(kv)
    Gref=[-1. 1. 0. 0.;-1. 0. 1. 0.;-1. 0. 0. 1.]
    Threads.@threads for e in 1:nt
        ids=m.T[:,e];X=m.X[:,ids]
        B=hcat(X[:,2]-X[:,1],X[:,3]-X[:,1],X[:,4]-X[:,1])
        g=B'\Gref;Ke=m.vol[e].*(g'*g)
        Me=fill(m.vol[e]/20,4,4)
        for i in 1:4
            Me[i,i]*=2
        end
        r,iv=CC._pick_rule(qs,pot,m,e)
        vs,ord=relative_values(pot,m,e,r,iv);We=zeros(4,4)
        for q in eachindex(r.w),a in 1:4,b in 1:4
            We[ord[a],ord[b]]+=m.vol[e]*r.w[q]*vs[q]*r.bary[q,a]*r.bary[q,b]
        end
        pos=16(e-1)
        for b in 1:4,a in 1:4
            pos+=1;I[pos]=ids[a];J[pos]=ids[b];kv[pos]=Ke[a,b];mv[pos]=Me[a,b];wv[pos]=We[a,b]
        end
    end
    return sparse(I,J,kv,nd,nd),sparse(I,J,mv,nd,nd),sparse(I,J,wv,nd,nd)
end

"Project invariant CECR matrices into all eight parity sectors, once."
function sector_parts(vm,H,M,P;check_tol=1e-10)
    S=CC.symmetry_data(vm,[1,2,3])
    errs=Dict("H"=>CC.invariance_errors(S,H),"M"=>CC.invariance_errors(S,M),
              "P"=>CC.invariance_errors(S,P))
    all(maximum(v)<check_tol for v in values(errs)) || error("matrix symmetry check failed: $errs")
    chars=CC.all_characters(3);bases=CC.sector_bases(S,chars)
    sum(size(B,2) for B in bases)==size(H,1) || error("sector dimensions incomplete")
    sectors=Any[]
    for (ch,B) in zip(chars,bases)
        push!(sectors,(character=ch,label=CC.charlabel([1,2,3],ch),
            H=B'*H*B,M=B'*M*B,P=B'*P*B))
    end
    return sectors,(invariance_errors=errs,tol=S.tol,ndof=S.ndof)
end

"All-sector union, preserving full-spectrum multiplicities and indices."
function sector_eigs(sectors,sigma,k;tol=1e-11)
    values=Float64[];residuals=Float64[];relative=Float64[];labels=String[];details=Any[]
    function solve_one(s)
        A=s.H+sigma*s.P;dim=size(A,1)
        if dim<=200
            F=cholesky(Symmetric(A))
            E=eigen(Symmetric(Matrix(A)),Symmetric(Matrix(s.M)))
            n=min(k,dim);mu=E.values[1:n]
            eq=Float64[];rr=Float64[]
            for j in 1:n
                ax=A*E.vectors[:,j];mx=s.M*E.vectors[:,j];r=ax-mu[j]*mx
                push!(eq,norm(r)/norm(mx));push!(rr,norm(r)/(norm(ax)+abs(mu[j])*norm(mx)))
            end
            info=(residuals=eq,backward_residuals=rr,mvproducts=0,factor_nnz=0)
        else
            mu,info=OptimalRateFEM.eigs(A,s.M,min(k,dim);tol=tol)
        end
        println("  sector $(s.label) dim=$dim mu=$mu");flush(stdout)
        return (mu=mu,info=info)
    end
    solved=Vector{Any}(undef,length(sectors))
    workers=min(parse(Int,get(ENV,"CECR_SECTOR_PARALLEL","4")),Threads.nthreads(),length(sectors))
    workers>=1 || error("CECR_SECTOR_PARALLEL must be positive")
    Threads.@threads for worker in 1:workers
        for i in worker:workers:length(sectors)
            solved[i]=solve_one(sectors[i])
        end
    end
    for (s,out) in zip(sectors,solved)
        mu=out.mu;info=out.info;dim=size(s.H,1)
        append!(values,mu);append!(residuals,info.residuals)
        append!(relative,info.backward_residuals);append!(labels,fill(s.label,length(mu)))
        push!(details,(label=s.label,character=s.character,dimension=dim,mu=mu,
            residuals=info.residuals,relative_residuals=info.backward_residuals,
            factor_nnz=info.factor_nnz,mvproducts=info.mvproducts))
    end
    ix=sortperm(values)[1:k]
    return values[ix],(residuals=residuals[ix],relative_residuals=relative[ix],
                       labels=labels[ix],sectors=details)
end
end
