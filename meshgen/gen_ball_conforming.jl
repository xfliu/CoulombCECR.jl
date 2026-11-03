# Conforming graded tetrahedral mesh of the ball B(0,R) with the origin as a vertex.
#
# Same node set and grading parameters as the paper's generator
# (Code_Sorted/cases/case5_3d_hydrogen_cube/mesh_generator/gen_structured_ball.jl):
#   * icosphere of subdivision level n_sub (chosen so that h_max <= h_max_factor*h),
#   * radial shells r_1 = h_min = h^2/(10R), r_{k+1} = r_k + max(h_min, theta*r_k) up to
#     r_cutoff, then uniform steps dr_outer up to R,
#   * one cone of tets from the origin to the first shell (singular patch of radius h_min).
#
# Difference to the paper generator: every triangular prism between two shells is split
# into 3 tets by the GLOBAL vertex-index rule (sort the face vertices a<b<c, then
#   [a_k b_k c_k c_{k+1}], [a_k b_k b_{k+1} c_{k+1}], [a_k a_{k+1} b_{k+1} c_{k+1}] ),
# so the diagonal of every lateral quad (u,v), u<v, is u_k--v_{k+1}, independently of which
# of the two neighbouring prisms is split. The paper generator split each prism in the
# LOCAL face order (p0,p1,p2), so two prisms sharing a lateral quad generally chose
# different diagonals -> hanging (non-matching) faces (221,760 unmatched facets at h=1.4).
#
# usage: julia gen_ball_conforming.jl --h_target 1.4 --output_dir <dir> [--R 6 --theta 0.1 ...]
using Printf, LinearAlgebra, Dates

# icosahedron base and subdivision copied verbatim (same vertex order) from gen_structured_ball.jl,
# so the node set is identical to the paper meshes
function icosa_base()
    phi = (1.0 + sqrt(5.0)) / 2.0
    raw = [0.0 1.0 phi; 0.0 -1.0 phi; 0.0 1.0 -phi; 0.0 -1.0 -phi; 1.0 phi 0.0; -1.0 phi 0.0;
           1.0 -phi 0.0; -1.0 -phi 0.0; phi 0.0 1.0; -phi 0.0 1.0; phi 0.0 -1.0; -phi 0.0 -1.0]
    faces = [1 2 9; 1 9 5; 1 5 6; 1 6 10; 1 10 2; 2 7 9; 9 7 11; 9 11 5; 5 11 3; 5 3 6;
             6 3 12; 6 12 10; 10 12 8; 10 8 2; 2 8 7; 4 7 8; 4 11 7; 4 3 11; 4 12 3; 4 8 12]
    return raw ./ norm(raw[1, :]), faces
end

function subdiv_icosa(n::Int)
    V, F = icosa_base()
    key = Dict{NTuple{3,Int64},Int}(); verts = NTuple{3,Float64}[]; faces = NTuple{3,Int}[]
    function addv(v)
        v = v / norm(v)
        k = (round(Int64, v[1] * 1e10), round(Int64, v[2] * 1e10), round(Int64, v[3] * 1e10))
        haskey(key, k) && return key[k]
        push!(verts, (v[1], v[2], v[3])); key[k] = length(verts); return length(verts)
    end
    for f in axes(F, 1)
        A = V[F[f, 1], :]; B = V[F[f, 2], :]; C = V[F[f, 3], :]
        g = Dict{NTuple{2,Int},Int}()
        for i in 0:n, j in 0:(n - i)
            g[(i, j)] = addv(((n - i - j) * A + i * B + j * C) / n)
        end
        for i in 0:(n - 1), j in 0:(n - 1 - i)
            push!(faces, (g[(i, j)], g[(i + 1, j)], g[(i, j + 1)]))
            i + j + 2 <= n && push!(faces, (g[(i + 1, j)], g[(i + 1, j + 1)], g[(i, j + 1)]))
        end
    end
    return [verts[i][d] for i in eachindex(verts), d in 1:3], [faces[i][d] for i in eachindex(faces), d in 1:3]
end

function make_radial_levels(h_min, theta, R, dr_outer, r_cutoff)
    levels = Float64[]; r = h_min
    while r < R
        push!(levels, r)
        if r < r_cutoff
            rn = r + max(h_min, theta * r); r = rn > r_cutoff ? r_cutoff : rn
        else
            r = min(r + dr_outer, R)
        end
    end
    (isempty(levels) || levels[end] < R) && push!(levels, R)
    return levels
end

function build_ball(n_sub, r_levels)
    sv, sf = subdiv_icosa(n_sub); ns = size(sv, 1); nf = size(sf, 1); nsh = length(r_levels)
    X = zeros(1 + nsh * ns, 3)
    for k in 1:nsh, j in 1:ns
        X[1 + (k - 1) * ns + j, :] = sv[j, :] .* r_levels[k]
    end
    sid(k, j) = 1 + (k - 1) * ns + j
    T = zeros(Int, nf * (1 + 3 * (nsh - 1)), 4); nt = 0
    for f in 1:nf
        a, b, c = sort(sf[f, :])
        nt += 1; T[nt, :] = [1, sid(1, a), sid(1, b), sid(1, c)]
    end
    for k in 1:(nsh - 1), f in 1:nf
        a, b, c = sort(sf[f, :])                       # global-index order
        A0, B0, C0 = sid(k, a), sid(k, b), sid(k, c)
        A1, B1, C1 = sid(k + 1, a), sid(k + 1, b), sid(k + 1, c)
        nt += 1; T[nt, :] = [A0, B0, C0, C1]
        nt += 1; T[nt, :] = [A0, B0, B1, C1]
        nt += 1; T[nt, :] = [A0, A1, B1, C1]
    end
    return X, T, ns, nf
end

function measures(X, T)
    nt = size(T, 1); vol = zeros(nt); hK = zeros(nt); dets = zeros(nt)
    for i in 1:nt
        v = [X[T[i, j], :] for j in 1:4]
        d = det(hcat(v[2] - v[1], v[3] - v[1], v[4] - v[1])); dets[i] = d; vol[i] = abs(d) / 6
        hK[i] = maximum(norm(v[p] - v[q]) for p in 1:4 for q in p+1:4)
    end
    return vol, hK, dets
end

"facet census: (#facets, #single facets, #single facets not on the sphere, #facets shared by >2)"
function facet_census(X, T, R)
    cnt = Dict{NTuple{3,Int},Int}()
    for i in axes(T, 1), s in 1:4
        f = sort([T[i, j] for j in 1:4 if j != s]); k = (f[1], f[2], f[3])
        cnt[k] = get(cnt, k, 0) + 1
    end
    single = 0; bad = 0; over = 0
    for (k, c) in cnt
        c > 2 && (over += 1)
        c == 1 || continue
        single += 1
        all(norm(X[i, :]) > R - 1e-8 for i in k) || (bad += 1)
    end
    return length(cnt), single, bad, over
end

function main()
    a = Dict("R" => "6.0", "h_target" => "0.8", "theta" => "0.1", "h_min" => "-1", "n_sub" => "-1",
             "r_cutoff" => "4.0", "h_max_factor" => "1.05", "output_dir" => "")
    i = 1
    while i < length(ARGS)
        a[lstrip(ARGS[i], '-')] = ARGS[i + 1]; i += 2
    end
    R = parse(Float64, a["R"]); h = parse(Float64, a["h_target"]); theta = parse(Float64, a["theta"])
    rc = parse(Float64, a["r_cutoff"]); hf = parse(Float64, a["h_max_factor"])
    h_min = parse(Float64, a["h_min"]) > 0 ? parse(Float64, a["h_min"]) : h^2 / (10R)
    ang = acos(1 / sqrt(5)); chord = 2sin(ang / 2); cdist = sqrt((3 + 6cos(ang)) / 9)
    cu(ns) = chord / (ns * cdist)
    function dro(ns)
        F = hf * h; c = cu(ns); b = -R * c^2; cc = R^2 * c^2 - F^2; D = b^2 - 4cc
        D < 0 ? (0.0, false) : ((-b + sqrt(D)) / 2, true)
    end
    n_sub = parse(Int, a["n_sub"]) > 0 ? parse(Int, a["n_sub"]) :
            max(ceil(Int, ang / (2asin(min(theta / 2, 0.9999)))), 4, ceil(Int, R * chord / (cdist * hf * h)))
    d, ok = dro(n_sub)
    while !ok
        n_sub += 1; d, ok = dro(n_sub)
    end
    dr_outer = min(h, d)
    out = isempty(a["output_dir"]) ? @sprintf("ball_R%d_h%03d_conf", round(Int, R), round(Int, 100h)) : a["output_dir"]
    t0 = time(); X = zeros(0, 3); T = zeros(Int, 0, 4); hK = Float64[]; vol = Float64[]; dets = Float64[]; lev = Float64[]
    for attempt in 1:4
        lev = make_radial_levels(h_min, theta, R, dr_outer, rc)
        X, T, ns, nf = build_ball(n_sub, lev)
        vol, hK, dets = measures(X, T)
        @printf("[attempt %d] n_sub=%d shells=%d nodes=%d tets=%d hmax=%.4f\n", attempt, n_sub, length(lev), size(X, 1), size(T, 1), maximum(hK))
        maximum(hK) <= hf * h && break
        n_sub = ceil(Int, n_sub * maximum(hK) / (hf * h)) + 1
        d, ok = dro(n_sub); dr_outer = ok ? min(h, d) : h / 4
    end
    nflip = 0
    for i in axes(T, 1)
        if dets[i] < 0
            T[i, 2], T[i, 3] = T[i, 3], T[i, 2]; nflip += 1
        end
    end
    nfac, single, bad, over = facet_census(X, T, R)
    cen = [norm(sum(X[T[i, j], :] for j in 1:4) / 4) for i in axes(T, 1)]
    sing = [any(norm(X[T[i, j], :]) < 1e-12 for j in 1:4) for i in axes(T, 1)]
    ratio = maximum(hK[.!sing] ./ cen[.!sing])
    patchR = maximum(maximum(norm(X[T[i, j], :]) for j in 1:4) for i in findall(sing))
    nbd = count(i -> norm(X[i, :]) > R - 1e-8, axes(X, 1))
    @printf("facets=%d single=%d interior_free=%d over-shared=%d  min|det|=%.3e flipped=%d\n", nfac, single, bad, over, minimum(abs.(dets)), nflip)
    @printf("hmax=%.6f hK_min=%.6e  max h_K/r_K (off origin) = %.4f  singular patch radius = %.4e (%d tets)\n",
            maximum(hK), minimum(hK), ratio, patchR, count(sing))
    mkpath(out)
    open(joinpath(out, "nodes.dat"), "w") do f
        for i in axes(X, 1); @printf(f, "%.15e %.15e %.15e\n", X[i, 1], X[i, 2], X[i, 3]); end
    end
    open(joinpath(out, "elements.dat"), "w") do f
        for i in axes(T, 1); @printf(f, "%d %d %d %d\n", T[i, 1], T[i, 2], T[i, 3], T[i, 4]); end
    end
    info = Dict("generator" => "gen_ball_conforming.jl (global-index prism split)", "R_ball" => R, "h_target" => h,
                "theta" => theta, "h_min" => h_min, "n_sub" => n_sub, "dr_outer" => dr_outer, "r_cutoff" => rc,
                "h_max_factor" => hf, "n_r_levels" => length(lev), "n_nodes" => size(X, 1), "n_tets" => size(T, 1),
                "hmax_actual" => maximum(hK), "hK_min" => minimum(hK), "n_boundary" => nbd,
                "n_facets" => nfac, "n_single_facets" => single, "interior_free_facets" => bad,
                "max_hK_over_rK" => ratio, "singular_patch_radius" => patchR, "created_at" => string(now()))
    open(joinpath(out, "info.txt"), "w") do f
        for (k, v) in sort(collect(info), by = first); println(f, "$k = $v"); end
    end
    @printf("SUMMARY h=%.2f n_sub=%d nodes=%d tets=%d hmax=%.4f interior_free=%d  (%.1fs) -> %s\n",
            h, n_sub, size(X, 1), size(T, 1), maximum(hK), bad, time() - t0, out)
end
main()
