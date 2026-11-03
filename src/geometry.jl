# Simplex mesh geometry used by the estimates (independent of the FEM
# DOF layout, which is handled by VFEM.jl).

using LinearAlgebra: det, norm, dot, cross

"""
    SimplexMesh{D}

* `X`   :: D × nv node coordinates
* `T`   :: (D+1) × ne element connectivity (1-based), same element order as
           the VFEM mesh it was built from
* `vol` :: element volumes (areas in 2D)
* `hK`  :: element diameters (longest edge)
* `bnodes` :: boolean vector, true for nodes on the boundary
"""
struct SimplexMesh{D}
    X::Matrix{Float64}
    T::Matrix{Int}
    vol::Vector{Float64}
    hK::Vector{Float64}
    bnodes::BitVector
end

nelem(m::SimplexMesh) = size(m.T, 2)
nnode(m::SimplexMesh) = size(m.X, 2)

function _simplex_measures(X::Matrix{Float64}, T::Matrix{Int})
    D = size(X, 1); ne = size(T, 2)
    vol = zeros(ne); hK = zeros(ne)
    fact = D == 2 ? 0.5 : 1 / 6
    Threads.@threads for e in 1:ne
        v = @view T[:, e]
        J = zeros(D, D)
        for i in 1:D, d in 1:D
            J[d, i] = X[d, v[i+1]] - X[d, v[1]]
        end
        vol[e] = abs(det(J)) * fact
        h = 0.0
        for i in 1:D+1, j in i+1:D+1
            s = 0.0
            for d in 1:D
                s += (X[d, v[i]] - X[d, v[j]])^2
            end
            h = max(h, sqrt(s))
        end
        hK[e] = h
    end
    return vol, hK
end

function _boundary_nodes(T::Matrix{Int}, nv::Int)
    D = size(T, 1) - 1
    cnt = Dict{NTuple{3,Int},Int}()
    for e in axes(T, 2)
        v = T[:, e]
        for skip in 1:D+1
            f = sort([v[i] for i in 1:D+1 if i != skip])
            key = D == 2 ? (f[1], f[2], 0) : (f[1], f[2], f[3])
            cnt[key] = get(cnt, key, 0) + 1
        end
    end
    b = falses(nv)
    for (k, c) in cnt
        c == 1 || continue
        b[k[1]] = true; b[k[2]] = true
        k[3] > 0 && (b[k[3]] = true)
    end
    return b
end

function SimplexMesh(X::Matrix{Float64}, T::Matrix{Int})
    vol, hK = _simplex_measures(X, T)
    D = size(X, 1)
    return SimplexMesh{D}(X, T, vol, hK, _boundary_nodes(T, size(X, 2)))
end

SimplexMesh(m::Mesh2D) = SimplexMesh(Matrix(m.nodes'), Matrix(m.elements'))
SimplexMesh(m::Mesh3D) = SimplexMesh(Matrix(m.NodeList'), Matrix(m.ElementList'))

# ---------------------------------------------------------------------------
# Point–simplex distances (exact Euclidean distance; 0 if the point is inside)

function _dist_point_segment(p, a, b)
    ab = b .- a; t = clamp(dot(p .- a, ab) / dot(ab, ab), 0.0, 1.0)
    return norm(p .- (a .+ t .* ab))
end

# Closest point on a 3D triangle (Ericson, Real-Time Collision Detection §5.1.5)
function _dist_point_triangle3(p, a, b, c)
    ab = b - a; ac = c - a; ap = p - a
    d1 = dot(ab, ap); d2 = dot(ac, ap)
    (d1 <= 0 && d2 <= 0) && return norm(ap)
    bp = p - b; d3 = dot(ab, bp); d4 = dot(ac, bp)
    (d3 >= 0 && d4 <= d3) && return norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        return norm(p - (a + (d1 / (d1 - d3)) * ab))
    end
    cp = p - c; d5 = dot(ab, cp); d6 = dot(ac, cp)
    (d6 >= 0 && d5 <= d6) && return norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        return norm(p - (a + (d2 / (d2 - d6)) * ac))
    end
    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return norm(p - (b + w * (c - b)))
    end
    denom = 1 / (va + vb + vc)
    v = vb * denom; w = vc * denom
    return norm(p - (a + ab * v + ac * w))
end

function _barycentric(m::SimplexMesh{D}, e::Int, p) where {D}
    v = @view m.T[:, e]
    J = zeros(D, D)
    for i in 1:D, d in 1:D
        J[d, i] = m.X[d, v[i+1]] - m.X[d, v[1]]
    end
    l = J \ (p .- m.X[:, v[1]])
    return vcat(1 - sum(l), l)
end

"""
    dist_point_simplex(m, e, p) -> Float64

Euclidean distance from point `p` to element `e` (0 if `p ∈ K`).
"""
function dist_point_simplex(m::SimplexMesh{D}, e::Int, p) where {D}
    λ = _barycentric(m, e, p)
    minimum(λ) >= -1e-14 && return 0.0
    v = @view m.T[:, e]
    P = [m.X[:, v[i]] for i in 1:D+1]
    if D == 2
        return min(_dist_point_segment(p, P[1], P[2]),
                   _dist_point_segment(p, P[2], P[3]),
                   _dist_point_segment(p, P[1], P[3]))
    else
        return min(_dist_point_triangle3(p, P[2], P[3], P[4]),
                   _dist_point_triangle3(p, P[1], P[3], P[4]),
                   _dist_point_triangle3(p, P[1], P[2], P[4]),
                   _dist_point_triangle3(p, P[1], P[2], P[3]))
    end
end

"Largest vertex distance from `p` to element `e`."
function maxdist_point_simplex(m::SimplexMesh{D}, e::Int, p) where {D}
    r = 0.0
    for i in 1:D+1
        r = max(r, norm(m.X[:, m.T[i, e]] .- p))
    end
    return r
end

"""
    interior_free_facets(m, on_boundary) -> Int

Number of facets that belong to only one element but whose vertices do not
all satisfy `on_boundary(x)`. A conforming mesh of a domain without holes
returns 0. Non-conforming meshes (hanging faces) or meshes with holes make
the P1 space non-conforming / change the domain, so results on them are
flagged.
"""
function interior_free_facets(m::SimplexMesh{D}, on_boundary::Function) where {D}
    cnt = Dict{NTuple{3,Int},Int}()
    for e in axes(m.T, 2)
        v = m.T[:, e]
        for skip in 1:D+1
            f = sort([v[i] for i in 1:D+1 if i != skip])
            key = D == 2 ? (f[1], f[2], 0) : (f[1], f[2], f[3])
            cnt[key] = get(cnt, key, 0) + 1
        end
    end
    bad = 0
    for (k, c) in cnt
        c == 1 || continue
        ids = D == 2 ? (k[1], k[2]) : k
        all(on_boundary(m.X[:, i]) for i in ids) || (bad += 1)
    end
    return bad
end
