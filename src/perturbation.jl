# Direct patch/off-patch estimate of d_h (paper Thm 5.3):
#
#   |((V - c_h)u, u)| ≤ d_h ‖u‖²_{H¹},
#   d_h = S_r² ‖V - c_h‖_{L^p(ω)} + max_{K ⊄ ω} (h_K/2) ‖V - V_K‖_{L^∞(K)},
#
# valid for ANY union of elements ω containing the elements that touch a
# singular point. ω is optimised over the family ω_j = first j elements in
# the order of increasing distance to the nearest centre.
#
# `per_centre = true` uses a local embedding constant S_r valid on
# B(a_i, r_loc) with disjoint support balls (paper (5.6)); the patch term is
# then S_r² max_i ‖V - c_h‖_{L^p(ω ∩ near a_i)} and ω ⊂ ∪ B(a_i, r_loc) is
# enforced.

"""
    direct_dh(pot, m, c, S, p; per_centre=false, r_loc=Inf, qs) -> NamedTuple

Returns (dh, patch, off, rho, npatch, rmax); `rmax` is the largest distance from a
patch-element vertex to its nearest centre (ω ⊂ ∪ B(a_i, rmax)).
"""
function direct_dh(pot::Coulomb{D}, m::SimplexMesh{D}, c::Vector{Float64},
                   S::Float64, p::Float64; per_centre::Bool = false,
                   r_loc::Float64 = Inf,
                   qs::QuadSet{D} = QuadSet(D; q_sing = D == 2 ? 3 : 2)) where {D}
    ne = nelem(m); nc = length(pot.Z)
    dmin = zeros(ne); near = zeros(Int, ne); toff = zeros(ne); dmaxc = zeros(ne)
    Threads.@threads for e in 1:ne
        dloc = Inf; iloc = 1
        for i in 1:nc
            di = dist_point_simplex(m, e, pot.A[:, i])
            if di < dloc
                dloc = di; iloc = i
            end
        end
        dmin[e] = dloc; near[e] = iloc
        dmaxc[e] = maxdist_point_simplex(m, e, pot.A[:, iloc])
        toff[e] = 0.5 * m.hK[e] * sup_dev(pot, m, e, c[e])
    end
    perm = sortperm(dmin)
    # suffix maximum of the off-patch term in sorted order
    sufmax = zeros(ne + 1)
    for j in ne:-1:1
        sufmax[j] = max(sufmax[j+1], toff[perm[j]])
    end
    nmand = count(==(0.0), dmin)            # elements touching a centre
    acc = zeros(nc)
    best = (dh = Inf, patch = NaN, off = NaN, rho = NaN, npatch = 0, rmax = NaN)
    rm = 0.0
    for j in 1:ne
        e = perm[j]
        if per_centre && dmaxc[e] > r_loc
            break                            # ω must stay inside B(a_i, r_loc)
        end
        rm = max(rm, dmaxc[e])
        acc[near[e]] += lp_dev_integral(pot, m, e, c[e], p; qs = qs)
        j < nmand && continue
        pt = per_centre ? S^2 * maximum(acc)^(1 / p) : S^2 * sum(acc)^(1 / p)
        pt > best.dh && break                # patch term only grows
        val = pt + sufmax[j+1]
        if val < best.dh
            best = (dh = val, patch = pt, off = sufmax[j+1],
                    rho = j < ne ? dmin[perm[j+1]] : Inf, npatch = j, rmax = rm)
        end
    end
    return best
end
