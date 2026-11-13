# Why the new mesh recovers the perturbation rate

This directory is an isolated numerical extension of the release. The supplied
`paper_revised.tex`, particularly the graded-mesh assumption, the direct
perturbation estimate, and the numerical Convergence subsection, is the
mathematical reference.

## The required exponent in two dimensions

For the paper's default exponents p = 4/3 and r = 8, the singular-patch
contribution satisfies

    d_patch <= 2 Z (3 pi)^(3/4) S_8^2 sqrt(rho).

Consequently rho_h = beta h^4 makes this contribution O(h^2). A radius
proportional to h^2 would only imply O(h), and a fixed radius allows the
stagnation described in the revised paper. The new default is beta = 1e-6;
it is recorded in each mesh report and can be changed explicitly.
There is no fixed lower cutoff replacing beta h^4.

The mesh-independent embedding constants remain the paper's values:
S_8 = 1.466 on [-5,5]^2 and S_8 = 1.462 on [-7,7] x [-5,5].

## A shape-regular core and an explicit expanded patch

The generator uses a Cartesian core around each nucleus and conforming
homothetic rectangular rings outside it. The core is scaled so its farthest
corner is rho_h from that nucleus. With n = ceil(4/h), its 2n by 2n Cartesian
subdivisions contain the nucleus as a vertex. The radial ring ratios are
bounded by 1 + 1/n. Logarithmic radial spacing ends exactly at the outer
rectangle; this avoids a final arbitrarily thin layer.

For the two-centre problem, the domain is divided at x = 0. The left and
right rectangles have centres at the two nuclei; their common interface
has matching subdivisions. No Delaunay/Qhull calculation, absolute
deduplication floor, or arbitrary fan of increasingly thin nucleus triangles
is needed.

The entire small Cartesian core is an admissible union-of-elements patch
for the direct perturbation estimate and supplies a geometric O(h^2)
competitor. The numerical patch optimizer may select a smaller patch; it
need not contain the entire designed core. The mesh report records the
designed core radius; the computation result separately records the radius
and number of elements of the patch actually selected for d_h.

This expanded-patch definition matters. The supplied original manuscript
defined omega_0 in its graded-mesh assumption to be only the nucleus-star;
the updated manuscript explicitly allows the expanded cell-union patch. Requiring the original (G1) on every triangle outside only the
nucleus-star forces increasingly narrow central fan
triangles in a conforming family: for a uniformly shape-regular central
triangle, an edge opposite the nucleus has length comparable to its distance
from the nucleus. The adjacent off-star triangle then has diameter/distance
bounded below independently of h, so it cannot satisfy diameter <= theta h
distance with a fixed theta as h tends to zero.

The implementation therefore uses the more general direct-estimate route
already documented in src/perturbation.jl: any cell-union patch containing
the singularities is admissible. Outside the expanded Cartesian core, the
reported diagnostic is

    theta_off_core = max_K diameter(K) / (h dist(K, nearest nucleus)).

The core lies in the closed ball of radius beta h^4, and theta_off_core
stays bounded under refinement. The direct-estimate proof gives
second-order patch and off-patch contributions for this full-core
comparison estimate. The computed optimizer searches distance prefixes,
which need not contain the exact Cartesian core as one of their candidates.
Its rate is justified by the enlarged-prefix argument below, rather than
by claiming its minimum is no larger than the exact full-core estimate.
The actual nucleus-star is contained in this core and hence satisfies the
shrinking-radius requirement too. This implementation does not claim
literal nucleus-star (G1) for the other triangles inside the expanded core.

For multiple nuclei the code evaluates the actual multi-centre potential
and the patch/off-patch expression; it does not reuse a single-centre
closed-form constant as if the other nucleus were absent.

## Why the distance-prefix optimizer retains the rate

Let rho_h cover every designed comparison core, and write
r_K = dist(K, {a_j}). The union

    P_h = union {K : r_K <= rho_h}

contains every core cell and is a candidate in the implemented search:
choose the prefix ending after the complete equality group at r_K = rho_h.
For every additional, noncore cell in this prefix, (G1) gives

    max_(x in K) |x - a_nearest(K)| <= r_K + h_K
                                      <= rho_h (1 + theta h).

Thus this prefix satisfies the shrinking-radius condition with the same
power of h and a uniform enlarged coefficient beta*(1+theta*h_0), for
0<h<=h_0. Outside this prefix, (G1) still holds because it already holds
outside the smaller designed cores. The resulting prefix comparison
estimate is O(h^2), so the computed minimum over distance prefixes is
O(h^2) as well. Pruning after the accumulated patch term exceeds the
incumbent is safe: that term only increases, while every remaining
candidate also has a nonnegative off-patch contribution.

The embedding constant must remain valid on this enlarged prefix. Our
2D constants are global rectangle constants, independent of the patch.
The 3D two-centre code uses fixed unit balls around each nucleus, and
rho_h*(1+theta*h)<1 on all reported levels, so the enlarged prefix remains
within those balls. Its local-radius stopping rule therefore does not
exclude the rate comparison prefix.

The full-core geometry-only estimate is an independent second-order
comparison. The prefix optimizer is not universally bounded above by
that exact full-core estimate; any such dominance in the recorded runs
is a numerical observation, not the search-family argument.

Also, on a bounded domain (G1)--(G2) imply h_max=O(h): outside the cores,
h_K<=theta*h*r_K<=C_Omega*h, and inside h_K<=2*rho_h. No separate
maximum-element-size assumption is needed for C_h^2=O(h^2).

## What must be checked on the actual mesh

Each mesh report includes node and triangle counts, core radius, measured
core-radius/h^4, theta_off_core, minimum angle, positive minimum element
area, interior free edges, nonmanifold edges, nodes outside the rectangle,
and total-area defect. Interior free edges and nonmanifold edges must be
zero. The geometric diagnostics, rather than a nominal sizing function,
show what was actually generated.

Each computation records d_h_patch, d_h_off, d_h, Gamma_h, A_h, C_h, the
bounds and eigensolver residuals. analyze.py computes local EOCs using

    log(value_coarse / value_fine) / log(h_coarse / h_fine).

The CSV reports both nominal-h EOCs and actual-h_max EOCs (the latter
are named EOC_hmax_*). The target is order two in nominal h for d_h,
Gamma_h, A_h and C_h^2.
The target patch radius has order four. The number of annular layers grows
like h^(-1) log(1/h), giving an extra logarithm in the triangle count;
second order in h should not be relabelled as an exact first-order law in
the number of triangles.

## Small elements require numerically stable local calculations

The new cells can be much smaller than the previous Qhull floor. Two
existing calculations become inaccurate at off-origin nuclei if implemented
in global coordinates:

* A quadrature point reconstructed as sum(lambda_i x_i), followed by
  subtraction of a nucleus at x = +/-2, loses relative precision. Evaluate
  the distance from centre a directly as sum(lambda_i (x_i - a)).
* The active VFEM ECR assembler uses Bernstein coefficients. Its quadratic
  enrichment normalization subtracts global squared-coordinate moments:
  beta = m_cell - mean(m_edges). On tiny off-origin cells this subtraction
  loses the cell-scale quadratic information through cancellation. Translate
  and uniformly scale each triangle before calling the Bernstein coefficient
  routines; the local ECR space is unchanged. Gradients transform with the
  inverse scale, mass integrals with the physical area. A legacy global
  monomial implementation has the related Vandermonde-conditioning problem,
  but that is not the active assembly path used by this study.

The stable local routines are isolated in this directory; the release's
original source files need not be changed.

## Finite-domain and whole-space convergence are different checks

On a fixed rectangle, the corrected lower bound converges to the Neumann
eigenvalue lambda_1^(Omega,N). A conforming Neumann Ritz value U_N supplies
a same-domain upper comparison. The width U_N - L_opt is the primary
eigenvalue convergence diagnostic.

The Dirichlet upper value U_D belongs to a different boundary condition:
U_D - L_opt approaches the Dirichlet-Neumann truncation gap and need not
vanish. For the one-centre whole-space problem the exact ground-state value
is -1. Its gaps -1 - L_opt and U_D + 1 also include truncation effects.
No exact whole-space eigenvalue is assumed for the two-centre problem.
Increasing the box size is a separate study if whole-space truncation error
becomes the dominant contribution.

All computations here use floating-point arithmetic and quadrature.
Residual and quadrature checks give numerical validation. Interval
arithmetic and verified matrix eigenvalue bounds would additionally be
required before calling the results rigorous certified enclosures.


## Independent geometric perturbation competitor

Run geometric_certificate.py to obtain geometric_certificate.json and
geometric_certificate.csv for every generated mesh. This independently
reads nodes.dat, elements.dat and core_elements.dat. It verifies positive
orientation, the presence of each nucleus, and containment of every
nucleus-star in the designated core, then recomputes distances and radii.

For every noncore element it bounds the oscillation through the Coulomb
gradient, without sampling V:

    d_off_upper = max_K h_K^2 / 2 * sum_i Z_i / dist(a_i,K)^2.

Cover the full core by disjoint balls B(a_i,rho), with rho the maximum of
the target radius and the measured full-core radius. Minkowski's and
Jensen's inequalities give the following patch competitor for p = 4/3:

    d_patch_upper = 2 S_8^2 sum_i Z_i [
        2 pi/(2-p) rho^(2-p)
        + sum_(j != i) pi rho^2 / (|a_i-a_j|-rho)^p
    ]^(1/p).

The first term integrates the singularity in its own ball. The remaining
terms bound that same potential on the other nuclei's balls. This
certificate uses neither cell-average quadrature nor numerical integration
of |V-c_h|^p. It reports both contributions, their sum, normalized values
d_h_upper/h^2, and local EOCs. It is an independent a priori second-order
check for an admissible patch; the optimized quadrature-based d_h is a
separate reported quantity. The geometry itself is evaluated in Float64,
so the file name does not imply an interval-arithmetic certificate.


## Canonical 3D H2+ construction and global edge cap

The canonical 3D meshes are generated by gen_mesh_3d.py with
--global-cap 1. The octant [0,8] x [0,6] x [0,6] contains the nucleus
(2,0,0). Its homothetic cuboid core has covering radius

    rho_h = 0.0015625 * h^2.

With n=ceil(4/h), the Cartesian octant core uses 2n subdivisions in x
(split at the nucleus) and n in y and z. Homothetic cuboid shells use
radial ratios at most 1+1/n. Each convex brick is tetrahedralized by a
pulling rule using global vertex order, so adjacent bricks choose the
same diagonal on every shared face. Reflecting the octant across all
three coordinate planes produces the full box and exact mesh symmetries.

Before reflection, refine_cap_3d.py selects the globally longest edge
whose length exceeds h and bisects every incident tetrahedron along that
same edge. This gives a conforming refined mesh without changing any
small-core cells. Since a child is contained in its parent, its diameter
cannot increase and its distance to any nucleus cannot decrease. Hence
(G1) is preserved, and the unchanged core preserves (G2). The final
maximum tetrahedron diameter is at most h.

The edge cap also preserves uniform shape regularity. The selected edge
is longest in every incident tetrahedron. Its new midpoint-to-opposite-
vertex edges have length at most sqrt(3)/2 times that parent diameter.
Along a descendant chain, after at most six bisections eliminating the
original overlong edges, the diameter contracts by at least that factor.
The initial diameter/h ratio is uniformly bounded, so the number of
bisections needed for the global cap is bounded independently of h.
Finite-depth bisection of the uniformly shape-regular initial family
preserves a mesh-independent positive shape bound. On the three reported
meshes, the actual maximum depth is six and normalized tetrahedron
quality remains above 0.0204.

The canonical directories are meshes/3d_h2plus_cap_h1.0,
meshes/3d_h2plus_cap_h0.6666666666666666, and
meshes/3d_h2plus_cap_h0.5. Their tetrahedron counts are 333376, 1153168,
and 2816016, with measured h_max 0.999683, 0.666652, and 0.499969.
Uncapped and exploratory meshes are retained separately and are not the
canonical paper data.

## Local multicentre patch bound in 3D

For patches omega_j inside B(a_j,1), use the paper's local constant
S6_local=sqrt(2)*K3. The support balls B(a_j,2) for the nuclei at +/-2
have disjoint interiors, so

    sum_j integral_(omega_j) |V-c_h| u^2
      <= S6_local^2 * max_j ||V-c_h||_(L^(3/2)(omega_j))
         * ||u||_(H1(Omega))^2.

The maximum comes from summing the squared local H1 norms on the
disjoint support balls. It is not a global box embedding constant.
For the two unit-charge nuclei separated by four, covering each full
core by a ball of radius rho<1 and applying Minkowski and Jensen gives

    d_patch_local_upper = 2*S6_local^2 * [
        (8*pi/3 * rho^(3/2))^(2/3)
        + (4*pi/3 * rho^3)^(2/3)/(4-rho)
    ].

This is O(rho)=O(h^2), and the smooth cross-nucleus term has higher order.
The geometric off-core majorant uses the sum of both Coulomb gradients.
The reports use bounding-box distances, which are lower bounds for true
point-to-tetrahedron distances, so their gradient majorants are
conservative. These full-core estimates are saved in
meshes/geometric_3d_cap_checks.json and remain distinct from the optimized
distance-prefix estimate, as explained above.

## Independent validation of the 3D meshes

Run test_mesh_3d.py --prefix 3d_h2plus_cap_h --levels 1.0
0.6666666666666666 0.5 --output meshes/mesh_validation_3d_cap.json
from this directory. It independently reads serialized files, checks
positive orientation, exact box volume, core-radius scaling, containment
of all nuclear-star tetrahedra in the cores, and positive normalized
quality. On the coarsest artifact it also recomputes the full facet
census and checks exact reflected node, tetrahedron and facet topology.
The saved result is meshes/mesh_validation_3d_cap.json and the log is
logs/test_mesh_3d_cap.log. Independent release-kernel perturbation checks
are stored as perturbation_check.json in the coarse and fine cap mesh
directories. They agree with the stable production kernels to the
reported quadrature accuracy.
