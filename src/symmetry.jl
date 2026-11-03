# Mirror-symmetry splitting of the CECR eigenproblem.
#
# Let the mesh (hence domain), and the cell averages c_h, be invariant under the
# reflections R_i : x_i -> -x_i, i ∈ axes. Each R_i permutes nodes, elements and
# facets, hence the CECR DOFs (facet-midpoint values and cell averages are mapped
# onto facet-midpoint values and cell averages), and the permutation matrices
# commute with A and M. The group G = Z_2^r (r = length(axes)) acts on R^n; for
# each character χ ∈ {±1}^r the isotypic subspace is spanned by the orthonormal
# columns of P_χ (one column per DOF orbit O whose stabiliser lies in ker χ, with
# entries χ(g)/sqrt(|O|) at the orbit points g(d0)), and
#   spec(A, M) = ⋃_χ spec(P_χ' A P_χ, P_χ' M P_χ)   (with multiplicities).

using SparseArrays: sparse, SparseMatrixCSC, nnz
using LinearAlgebra: eigen, cholesky, Symmetric, norm
import VFEM

# ---------------------------------------------------------------- mesh access
_coords(vm::Mesh3D) = vm.NodeList            # nv × D
_coords(vm::Mesh2D) = vm.nodes
_elems(vm::Mesh3D) = vm.ElementList           # ne × (D+1)
_elems(vm::Mesh2D) = vm.elements
_facets(vm::Mesh3D) = vm.FacetList            # nf × D (rows sorted)
_facets(vm::Mesh2D) = vm.edges

"(facet DOF index vector, cell DOF index vector) of the VFEM CECR assembly"
_dofmaps(vm::Mesh3D) = (collect(1:vm.NumF), collect(vm.NumF+1:vm.NumF+vm.NumElt))
function _dofmaps(vm::Mesh2D)
    o = VFEM.build_ecr_dof_ordering(vm.tri2edge, vm.ne)   # as in create_matrix_enriched_crouzeix_raviart
    return (o.edge, o.cell)
end

function _min_edge(X::Matrix{Float64}, T::Matrix{Int})
    h = Inf; D = size(X, 2)
    @inbounds for e in axes(T, 1), i in 1:size(T, 2), j in i+1:size(T, 2)
        s = 0.0
        for d in 1:D
            s += (X[T[e, i], d] - X[T[e, j], d])^2
        end
        h = min(h, s)
    end
    return sqrt(h)
end

"""
    node_reflection(X, i, tol) -> p

Node permutation of the reflection x_i -> -x_i: X[p[v], :] ≈ R_i X[v, :] within
`tol` (hashing on a grid of size `tol`, neighbouring cells searched). Throws
if a reflected node has no partner.
"""
function node_reflection(X::Matrix{Float64}, i::Int, tol::Float64)
    nv, D = size(X)
    key(x) = ntuple(d -> round(Int, x[d] / tol), D)
    H = Dict{NTuple{D,Int},Int}()
    sizehint!(H, nv)
    for v in 1:nv
        H[key(@view X[v, :])] = v
    end
    p = zeros(Int, nv)
    offs = vec(collect(Iterators.product(ntuple(_ -> -1:1, D)...)))
    y = zeros(D)
    for v in 1:nv
        y .= @view X[v, :]; y[i] = -y[i]
        k0 = key(y); best = 0; bd = Inf
        for o in offs
            w = get(H, k0 .+ o, 0)
            w == 0 && continue
            dd = sqrt(sum(abs2, (@view X[w, :]) .- y))
            dd < bd && (bd = dd; best = w)
        end
        (best == 0 || bd > tol) &&
            error("node_reflection: node $v has no mirror image across x_$i = 0 (tol = $tol)")
        p[v] = best
    end
    p[p] == 1:nv || error("node_reflection: not an involution")
    return p
end

"permutation of the simplices (rows of `T`, any vertex order) induced by node permutation `pn`"
function simplex_permutation(T::Matrix{Int}, pn::Vector{Int}, what::String)
    nr, q = size(T)
    H = Dict{NTuple{q,Int},Int}()
    sizehint!(H, nr)
    for r in 1:nr
        H[Tuple(sort(T[r, :]))] = r
    end
    p = zeros(Int, nr)
    for r in 1:nr
        w = get(H, Tuple(sort(pn[T[r, :]])), 0)
        w == 0 && error("simplex_permutation: reflected $what $r is not a $what of the mesh")
        p[r] = w
    end
    return p
end

"""
    SymmetryData

Node/element/facet permutations of each generator reflection, the DOF
permutations of all |G| = 2^r group elements (bitmask g: bit j ↔ axes[j]).
"""
struct SymmetryData
    axes::Vector{Int}
    tol::Float64
    pnode::Vector{Vector{Int}}
    pelem::Vector{Vector{Int}}
    pfacet::Vector{Vector{Int}}
    pdof::Vector{Vector{Int}}          # index g+1, g = 0..2^r-1
    ndof::Int
    t_build::Float64
end

const _SYM_CACHE = IdDict{Any,SymmetryData}()

function symmetry_data(vm, axes::Vector{Int}; tol_rel::Float64 = 1e-6)
    key = (vm, Tuple(axes), tol_rel)
    haskey(_SYM_CACHE, key) && return _SYM_CACHE[key]
    t0 = time()
    X = _coords(vm); T = _elems(vm); F = _facets(vm)
    tol = tol_rel * _min_edge(X, T)
    fd, cd = _dofmaps(vm)
    n = length(fd) + length(cd)
    pn = Vector{Int}[]; pe = Vector{Int}[]; pf = Vector{Int}[]; gens = Vector{Int}[]
    for i in axes
        p = node_reflection(X, i, tol)
        e = simplex_permutation(T, p, "element")
        f = simplex_permutation(F, p, "facet")
        d = zeros(Int, n)
        d[fd] = fd[f]; d[cd] = cd[e]
        push!(pn, p); push!(pe, e); push!(pf, f); push!(gens, d)
    end
    r = length(axes)
    pd = Vector{Vector{Int}}(undef, 2^r)
    for g in 0:2^r-1
        q = collect(1:n)
        for j in 1:r
            (g >> (j - 1)) & 1 == 1 && (q = gens[j][q])
        end
        pd[g+1] = q
    end
    S = SymmetryData(copy(axes), tol, pn, pe, pf, pd, n, time() - t0)
    _SYM_CACHE[key] = S
    return S
end

"element orbit average of c (G acts on elements through the generator permutations)"
function orbit_average(S::SymmetryData, c::AbstractVector)
    r = length(S.axes)
    cs = zeros(length(c))
    for g in 0:2^r-1
        q = collect(1:length(c))
        for j in 1:r
            (g >> (j - 1)) & 1 == 1 && (q = S.pelem[j][q])
        end
        cs .+= c[q]
    end
    return cs ./ 2^r
end

"‖A[p,p] - A‖_F / ‖A‖_F for each generator DOF permutation"
invariance_errors(S::SymmetryData, A) =
    [norm(A[S.pdof[1+(1<<(j-1))], S.pdof[1+(1<<(j-1))]] - A) / norm(A) for j in 1:length(S.axes)]

χval(s::Vector{Int}, g::Int) = prod((s[j] == -1 && (g >> (j - 1)) & 1 == 1) ? -1 : 1 for j in eachindex(s); init = 1)

"all characters of Z_2^r, the fully symmetric one first"
all_characters(r::Int) = [[(m >> (j - 1)) & 1 == 1 ? -1 : 1 for j in 1:r] for m in 0:2^r-1]

"""
    sector_bases(S, chars) -> Vector{SparseMatrixCSC}

Orthonormal sparse bases P_χ of the isotypic subspaces for the characters `chars`.
"""
function sector_bases(S::SymmetryData, chars::Vector{Vector{Int}})
    n = S.ndof; G = length(S.pdof); ns = length(chars)
    I = [Int[] for _ in 1:ns]; J = [Int[] for _ in 1:ns]; V = [Float64[] for _ in 1:ns]
    ncol = zeros(Int, ns)
    img = zeros(Int, G); pts = Int[]; gof = Int[]
    for d in 1:n
        for g in 1:G
            img[g] = S.pdof[g][d]
        end
        minimum(img) == d || continue                 # d is the orbit representative
        empty!(pts); empty!(gof)
        for g in 1:G
            img[g] in pts || (push!(pts, img[g]); push!(gof, g - 1))
        end
        w = 1 / sqrt(length(pts))
        for (s, ch) in enumerate(chars)
            ok = all(χval(ch, g - 1) == 1 for g in 1:G if img[g] == d)   # stabiliser ⊂ ker χ
            ok || continue
            ncol[s] += 1
            for (q, g) in zip(pts, gof)
                push!(I[s], q); push!(J[s], ncol[s]); push!(V[s], χval(ch, g) * w)
            end
        end
    end
    return [sparse(I[s], J[s], V[s], n, ncol[s]) for s in 1:ns]
end

function _sector_eigs(As, Ms, k; shift, tol)
    m = size(As, 1)
    if m <= 200          # tiny sectors (unit tests): dense
        Ad = Matrix(Symmetric((As + As') / 2)); Md = Matrix(Symmetric((Ms + Ms') / 2))
        cholesky(Symmetric(Ad - shift * Md))       # throws unless A - shift M is SPD
        E = eigen(Symmetric(Ad), Symmetric(Md))
        kk = min(k, m)
        λ = E.values[1:kk]
        res = [norm(Ad * E.vectors[:, i] - λ[i] * (Md * E.vectors[:, i])) / norm(Md * E.vectors[:, i]) for i in 1:kk]
        return λ, (residuals = res, nconv = kk, mvproducts = 0, factor_nnz = 0)
    end
    return smallest_eigs(As, Ms, k; shift = shift, tol = tol)
end

charlabel(axes, s) = join(("x$(a)" * (si == 1 ? "+" : "-") for (a, si) in zip(axes, s)), ",")

"""
    cecr_eigs_symmetric(vm, c, sigma, k, axes; shift=0.0, tol=1e-11, sectors=nothing,
                        check_tol=1e-10, verbose=false) -> (μ^σ_{1..k}, info)

Smallest k CECR eigenvalues (same convention as `cecr_eigs`) on a mesh that is
mirror-symmetric with respect to the planes x_i = 0, i ∈ axes, computed sector by
sector (group Z_2^r, characters χ ∈ {±1}^r; χ_j = +1: even in x_{axes[j]}).
`c` is replaced by its orbit average (max change recorded in info). A and M are
checked to be invariant under every generator (relative Frobenius error <
`check_tol`, else an error is thrown). `sectors` restricts the solve to a list of
characters (diagnostics); the k smallest over the solved sectors are returned.
"""
function cecr_eigs_symmetric(vm, c::AbstractVector, sigma::Float64, k::Int, axes;
                             shift::Float64 = 0.0, tol::Float64 = 1e-11, sectors = nothing,
                             check_tol::Float64 = 1e-10, verbose::Bool = false)
    t0 = time(); ax = collect(Int, axes)
    S = symmetry_data(vm, ax)
    cs = orbit_average(S, c)
    dc = maximum(abs.(cs .- c)); dcrel = dc / max(maximum(abs.(c)), eps())
    t1 = time()
    A, M = cecr_matrices(vm, cs .+ sigma)
    t_asm = time() - t1
    t1 = time()
    errA = invariance_errors(S, A); errM = invariance_errors(S, M)
    t_check = time() - t1
    (maximum(errA) < check_tol && maximum(errM) < check_tol) ||
        error("cecr_eigs_symmetric: matrices not invariant (A: $errA, M: $errM)")
    chars = sectors === nothing ? all_characters(length(ax)) : [collect(Int, s) for s in sectors]
    t1 = time()
    Ps = sector_bases(S, chars)
    t_basis = time() - t1
    sum(size(P, 2) for P in Ps) == S.ndof || sectors !== nothing ||
        error("cecr_eigs_symmetric: sector dimensions do not add up")
    lam = Float64[]; lab = String[]; res = Float64[]; sinfo = Any[]
    for (s, P) in zip(chars, Ps)
        t1 = time()
        As = P' * A * P; Ms = P' * M * P
        t_proj = time() - t1
        t1 = time()
        λ, inf = _sector_eigs(As, Ms, k; shift = shift, tol = tol)
        t_eig = time() - t1
        lbl = charlabel(ax, s)
        push!(sinfo, (character = s, label = lbl, dim = size(P, 2), nnzA = nnz(As),
                      factor_nnz = get(inf, :factor_nnz, 0), t_project = t_proj, t_eig = t_eig,
                      maxrss_GB = Sys.maxrss() / 2^30, mu = λ, residuals = inf.residuals))
        verbose && (println("  sector $lbl dim=$(size(P,2)) t_eig=$(round(t_eig, digits=2))s mu=$(λ)"); flush(stdout))
        append!(lam, λ); append!(lab, fill(lbl, length(λ))); append!(res, inf.residuals)
        As = Ms = nothing; GC.gc()
    end
    p = sortperm(lam)[1:min(k, length(lam))]
    info = (residuals = res[p], sector = lab[p], sectors = sinfo, ndof = S.ndof,
            c_orbit_maxchange = dc, c_orbit_maxchange_rel = dcrel,
            invariance_A = errA, invariance_M = errM, node_tol = S.tol,
            t_symmetry = S.t_build, t_assemble = t_asm, t_check = t_check, t_basis = t_basis,
            t_total = time() - t0)
    return lam[p], info
end
