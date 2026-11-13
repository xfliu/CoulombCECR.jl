module OptimalRateFEM
using CoulombCECR, VFEM, LinearAlgebra, SparseArrays, ArnoldiMethod, Random
const CC = CoulombCECR

# ECR = P1 + span{|x|^2}. Translation and ISOTROPIC scaling leave this
# space unchanged. Raw global quadratic moments lose all small-cell digits.
function cecr_parts(vm, m, c)
    nt=vm.nt; nd=vm.ne+nt
    dm=VFEM.build_ecr_dof_ordering(vm.tri2edge,vm.ne)
    I=zeros(Int,16nt); J=similar(I); kv=zeros(16nt); mv=similar(kv)
    Ccr=Float64.(VFEM._ECR_C_CR)
    G1=Float64.(VFEM._ECR_G1_NUM)./12
    G2=Float64.(VFEM._ECR_G2_NUM)./90
    Threads.@threads for e in 1:nt
        P=vm.nodes[vm.elements[e,:],:]
        Y=(P .- P[1:1,:])./m.hK[e]
        a=abs(det(hcat(Y[2,:]-Y[1,:],Y[3,:]-Y[1,:])))/2
        C=VFEM._local_ecr_bernstein_coefficients(Y,Ccr)
        g=VFEM._local_grad_lambda(Y,a)./m.hK[e]
        gx=zeros(3,4); gy=similar(gx)
        for j in 1:4
            z=VFEM._bernstein_quadratic_grad_coeffs(C[:,j],g)
            gx[:,j]=z[:,1]; gy[:,j]=z[:,2]
        end
        Ke=m.vol[e].*(gx'*G1*gx+gy'*G1*gy)
        Me=m.vol[e].*(C'*G2*C)
        ids=dm.local_dof[e,:]
        k=16(e-1)
        for j in 1:4,i in 1:4
            k+=1; I[k]=ids[i]; J[k]=ids[j]; kv[k]=Ke[i,j]; mv[k]=Me[i,j]
        end
    end
    K=sparse(I,J,kv,nd,nd); M=sparse(I,J,mv,nd,nd)
    # The CECR reaction acts only on cell-average DOFs.
    R=sparse(dm.cell,dm.cell,c.*m.vol,nd,nd)
    P=sparse(dm.cell,dm.cell,m.vol,nd,nd)
    return K,M,R,P
end

# Evaluate x-a as a convex combination of VERTEX DISPLACEMENTS. This
# avoids forming x near +/-2 and then subtracting the nucleus position.
function relative_values(pot,m,e,rule,iv)
    ord=iv==0 ? collect(1:3) : [iv; [j for j in 1:3 if j!=iv]]
    nq=length(rule.w); vals=zeros(nq)
    for ic in eachindex(pot.Z)
        D=m.X[:,m.T[ord,e]] .- pot.A[:,ic]
        for q in 1:nq
            dx=0.; dy=0.
            for j in 1:3
                dx+=rule.bary[q,j]*D[1,j]; dy+=rule.bary[q,j]*D[2,j]
            end
            vals[q]-=pot.Z[ic]/hypot(dx,dy)
        end
    end
    return vals,ord
end

function averages(pot,m; qs=CC.QuadSet(2;n_sing=16,n_near=10,n_far=5))
    c=zeros(CC.nelem(m))
    Threads.@threads for e in eachindex(c)
        r,iv=CC._pick_rule(qs,pot,m,e)
        v,_=relative_values(pot,m,e,r,iv)
        c[e]=dot(r.w,v)
    end
    return c
end

function perturbation(pot,m,c,S,p;
        qs=CC.QuadSet(2;n_sing=20,n_near=12,n_far=6,q_sing=3))
    ne=CC.nelem(m); d=zeros(ne); rm=zeros(ne); off=zeros(ne)
    Threads.@threads for e in 1:ne
        ds=[CC.dist_point_simplex(m,e,pot.A[:,i]) for i in eachindex(pot.Z)]
        i=argmin(ds); d[e]=ds[i]
        rm[e]=CC.maxdist_point_simplex(m,e,pot.A[:,i])
        off[e]=m.hK[e]/2*CC.sup_dev(pot,m,e,c[e])
    end
    perm=sortperm(d); suffix=zeros(ne+1)
    for j in ne:-1:1
        suffix[j]=max(suffix[j+1],off[perm[j]])
    end
    mand=count(==(0.),d); accum=0.; rmax=0.
    best=(dh=Inf,patch=NaN,off=NaN,rho=NaN,npatch=0,rmax=NaN)
    for j in 1:ne
        e=perm[j]; r,iv=CC._pick_rule(qs,pot,m,e)
        v,_=relative_values(pot,m,e,r,iv)
        accum+=m.vol[e]*dot(r.w,abs.(v.-c[e]).^p)
        rmax=max(rmax,rm[e]); j<mand && continue
        pt=S^2*accum^(1/p)
        pt>best.dh && break
        if pt+suffix[j+1]<best.dh
            best=(dh=pt+suffix[j+1],patch=pt,off=suffix[j+1],
                  rho=j<ne ? d[perm[j+1]] : Inf,npatch=j,rmax=rmax)
        end
    end
    return best
end

# Coulomb P1 matrices integrate V itself, not the cell average c_h.
function p1_parts(pot,m; qs=CC.QuadSet(2;n_sing=16,n_near=10,n_far=5))
    nt=CC.nelem(m); nd=CC.nnode(m)
    I=zeros(Int,9nt); J=similar(I); kv=zeros(9nt); mv=similar(kv); wv=similar(kv)
    Threads.@threads for e in 1:nt
        ids=m.T[:,e]; X=m.X[:,ids]
        B=hcat(X[:,2]-X[:,1],X[:,3]-X[:,1])
        g=B'\[-1. 1. 0.;-1. 0. 1.]
        Ke=m.vol[e].*(g'*g)
        Me=m.vol[e]/12 .* [2. 1. 1.;1. 2. 1.;1. 1. 2.]
        r,iv=CC._pick_rule(qs,pot,m,e)
        vs,ord=relative_values(pot,m,e,r,iv); We=zeros(3,3)
        for q in eachindex(r.w),a in 1:3,b in 1:3
            We[ord[a],ord[b]]+=m.vol[e]*r.w[q]*vs[q]*r.bary[q,a]*r.bary[q,b]
        end
        k=9(e-1)
        for b in 1:3,a in 1:3
            k+=1; I[k]=ids[a];J[k]=ids[b];kv[k]=Ke[a,b];mv[k]=Me[a,b];wv[k]=We[a,b]
        end
    end
    return sparse(I,J,kv,nd,nd),sparse(I,J,mv,nd,nd),sparse(I,J,wv,nd,nd)
end

# Return eigen-equation residual AND normwise backward residual. Tiny-cell
# stiffness contributions make ||r||/||Mx|| alone an overly pessimistic test.
function eigs(A,M,k; shift=0.,tol=1e-11)
    A=CC.symmetrize(A);M=CC.symmetrize(M)
    F=cholesky(Symmetric(A-shift*M);check=false)
    issuccess(F)||error("eigensolver shift does not give SPD matrix: $shift")
    op=CC.ShiftInvert(F,M,size(A,1),zeros(size(A,1)))
    nev=min(k+3,size(A,1)-2)
    dec,hist=partialschur(op;nev=nev,tol=tol,which=:LM,
        mindim=min(max(2nev,20),size(A,1)-1),maxdim=min(max(4nev,40),size(A,1)),restarts=2000)
    hist.converged||error("Arnoldi failed: $hist")
    th,X=partialeigen(dec); vals=shift .+1 ./real.(th); ord=sortperm(vals)
    vals=vals[ord[1:k]];X=real.(X[:,ord[1:k]])
    eq=zeros(k); back=zeros(k)
    for i in 1:k
        ax=A*X[:,i];mx=M*X[:,i];r=ax-vals[i]*mx
        eq[i]=norm(r)/norm(mx)
        back[i]=norm(r)/(norm(ax)+abs(vals[i])*norm(mx))
    end
    return vals,(residuals=eq,backward_residuals=back,nconv=hist.nconverged,
                 mvproducts=hist.mvproducts,factor_nnz=nnz(F))
end
end
